"""推論後端：Hailo-8（.hef）或 CPU 上的 YOLO（.pt）。

兩個 class 介面相同，其他模組只透過這個介面用模型：
    names           {class_id: class_name}
    __call__(cells) 吃一串 BGR 影像（大小可以不一樣），回傳 probs[N, C]
    close()         釋放硬體 / 資源

用 config.yaml 裡 model: 的副檔名選後端，不另加設定 key。
"""

from pathlib import Path

import cv2
import numpy as np

from .config import CLASS_NAMES, Config


class HailoClassifier:
    """
    Hailo-8 AI HAT 推論。

    前處理刻意做得跟 ultralytics 的 classify 一模一樣（短邊 resize → 置中裁切 →
    BGR→RGB，不做 mean/std 正規化），這樣 .hef 和 .pt 的判讀才會一致；
    實測 210 格 argmax 全同、機率平均差 0.002。HEF 的 softmax 已經包在模型裡，
    輸出直接就是機率。

    VDevice / 網路 / 串流在 __init__ 全部開好並保持開啟，每次推論直接餵；
    每次重新 configure 要好幾百毫秒，會把 Hailo 的速度優勢吃掉。
    """

    def __init__(self, hef_path: Path):
        try:
            from hailo_platform import (HEF, ConfigureParams, FormatType,
                                        HailoStreamInterface, InferVStreams,
                                        InputVStreamParams, OutputVStreamParams,
                                        VDevice)
        except ImportError as e:
            raise ImportError(
                "找不到 hailo_platform。它是 apt 的 python3-hailort 帶的，裝在系統 Python；"
                "venv 必須用 `uv venv --system-site-packages` 建立才看得到（見 README）。"
            ) from e

        self.names = dict(enumerate(CLASS_NAMES))
        self._hef = HEF(str(hef_path))
        in_info = self._hef.get_input_vstream_infos()[0]
        out_info = self._hef.get_output_vstream_infos()[0]
        self._in_name = in_info.name
        self._out_name = out_info.name
        # HEF 的輸入尺寸是編譯時固定的（NHWC），config 的 grid.imgsz 對它沒作用
        self.imgsz = int(in_info.shape[0])
        if int(out_info.shape[-1]) != len(CLASS_NAMES):
            raise ValueError(
                f"HEF 輸出 {out_info.shape[-1]} 類，但 CLASS_NAMES 有 {len(CLASS_NAMES)} 個")

        self._vdevice = VDevice()
        params = ConfigureParams.create_from_hef(self._hef, interface=HailoStreamInterface.PCIe)
        self._network = self._vdevice.configure(self._hef, params)[0]
        in_params = InputVStreamParams.make(self._network, format_type=FormatType.UINT8)
        out_params = OutputVStreamParams.make(self._network, format_type=FormatType.FLOAT32)
        self._activation = self._network.activate()
        self._activation.__enter__()
        self._pipeline = InferVStreams(self._network, in_params, out_params)
        self._pipeline.__enter__()

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        size = self.imgsz
        h, w = img.shape[:2]
        scale = size / min(h, w)
        img = cv2.resize(img, (max(size, round(w * scale)), max(size, round(h * scale))),
                         interpolation=cv2.INTER_LINEAR)
        h, w = img.shape[:2]
        top, left = (h - size) // 2, (w - size) // 2
        img = img[top:top + size, left:left + size]
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __call__(self, cells: list[np.ndarray]) -> np.ndarray:
        batch = np.stack([self._preprocess(c) for c in cells]).astype(np.uint8)
        out = self._pipeline.infer({self._in_name: batch})[self._out_name]
        return np.asarray(out, dtype=np.float32).reshape(len(cells), -1)

    def close(self) -> None:
        for obj in (self._pipeline, self._activation):
            try:
                obj.__exit__(None, None, None)
            except Exception:
                pass
        self._vdevice.release()


class YoloClassifier:
    """ultralytics YOLO 分類模型在 CPU 上推論（沒有 Hailo HAT 時的備援）。"""

    def __init__(self, pt_path: Path, imgsz: int, torch_threads: int):
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(torch_threads)   # 留 CPU 給讀取與編碼，避免畫面卡頓
        self._model = YOLO(str(pt_path))
        self.imgsz = imgsz
        self.names = self._model.names

    def __call__(self, cells: list[np.ndarray]) -> np.ndarray:
        preds = self._model(cells, imgsz=self.imgsz, verbose=False)
        return np.stack([p.probs.data.cpu().numpy() for p in preds]).astype(np.float32)

    def close(self) -> None:
        pass


def load_classifier(cfg: Config):
    """依 model: 的副檔名建立對應後端。"""
    suffix = cfg.model.suffix.lower()
    if suffix == ".hef":
        clf = HailoClassifier(cfg.model)
        print(f"模型：{cfg.model.name}（Hailo-8，輸入 {clf.imgsz}×{clf.imgsz}）")
    elif suffix == ".pt":
        clf = YoloClassifier(cfg.model, cfg.grid.imgsz, cfg.camera.torch_threads)
        print(f"模型：{cfg.model.name}（CPU，imgsz {clf.imgsz}，{cfg.camera.torch_threads} 執行緒）")
    else:
        raise ValueError(f"不支援的模型格式 {suffix}：只認 .hef（Hailo）或 .pt（CPU）")
    return clf
