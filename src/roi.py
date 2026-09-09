"""框選要分析的路面範圍，並把結果存起來重複使用。"""

import json
from pathlib import Path

import cv2
import numpy as np

from .config import PROJECT_ROOT, ROI_WINDOW_TITLE, WindowConfig

ROI_STORE = PROJECT_ROOT / "roi.json"

Roi = tuple[int, int, int, int]


def camera_key(camera_id: int) -> str:
    """相機的 roi.json 鍵值。每顆鏡頭各存一組，cam0 和 cam1 可以框不同範圍。"""
    return f"camera{camera_id}"


def video_key(video_path: Path) -> str:
    """影片檔的 roi.json 鍵值。用檔名而非完整路徑，搬動資料夾後仍然對得上。"""
    return f"video:{video_path.name}"


def _load_store() -> dict:
    if not ROI_STORE.exists():
        return {}
    try:
        with open(ROI_STORE, encoding="utf-8") as f:
            return json.load(f) or {}
    except (json.JSONDecodeError, OSError) as e:
        print(f"[ROI] {ROI_STORE.name} 讀取失敗（{e}），當作沒有存過")
        return {}


def save_roi(key: str, roi: Roi, width: int, height: int) -> None:
    """
    存成 0~1 的比例而不是像素座標。

    這樣之後改解析度（例如相機從 720p 換成 1080p、或換一支不同尺寸的影片）
    同一份框選還能直接用，不必重框。
    """
    store = _load_store()
    x1, y1, x2, y2 = roi
    store[key] = [x1 / width, y1 / height, x2 / width, y2 / height]
    with open(ROI_STORE, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2)
    print(f"[ROI] 已存到 {ROI_STORE.name}（下次不用再框，除非鏡頭位置有動過）")


def load_roi(key: str, width: int, height: int) -> Roi | None:
    """讀回指定來源的範圍並換算成目前解析度的像素座標；沒存過就回傳 None。"""
    store = _load_store()
    if key not in store:
        return None
    try:
        fx1, fy1, fx2, fy2 = store[key]
    except (ValueError, TypeError):
        print(f"[ROI] {key} 的設定格式不對，當作沒有存過")
        return None
    x1 = max(0, min(width, int(round(fx1 * width))))
    y1 = max(0, min(height, int(round(fy1 * height))))
    x2 = max(0, min(width, int(round(fx2 * width))))
    y2 = max(0, min(height, int(round(fy2 * height))))
    if x2 - x1 < 1 or y2 - y1 < 1:
        print(f"[ROI] {key} 的範圍太小，當作沒有存過")
        return None
    return x1, y1, x2, y2


def select_roi(first_frame: np.ndarray, win_cfg: WindowConfig) -> Roi | None:
    """
    開視窗讓使用者用滑鼠拖出要分析的路面範圍，回傳原始解析度下的 (x1, y1, x2, y2)。
    按 C 取消、或沒框到東西時回傳 None（呼叫端會改用全畫面）。

    畫面比螢幕大時會先等比例縮小再顯示，框完把座標依縮放比例換算回原始解析度，
    所以縮放只影響操作時看到的大小、不影響實際分析範圍的精度。
    """
    h, w = first_frame.shape[:2]
    scale = min(1.0, win_cfg.roi_max_w / w, win_cfg.roi_max_h / h)
    if scale < 1.0:
        view = cv2.resize(first_frame, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_AREA)
        print(f"畫面 {w}×{h} 太大，縮到 {view.shape[1]}×{view.shape[0]} 顯示"
              f"（座標會自動換算回原尺寸）")
    else:
        view = first_frame

    print("請用滑鼠拖拉選取分析範圍，按 Enter 確認，按 C 取消（全畫面）")
    x, y, bw, bh = cv2.selectROI(ROI_WINDOW_TITLE, view,
                                 showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(ROI_WINDOW_TITLE)
    cv2.waitKey(1)          # 讓 Qt 有機會真正把視窗關掉，否則視窗會殘留在畫面上

    if bw == 0 or bh == 0:
        return None

    # 換算回原始解析度，並夾在畫面範圍內，避免四捨五入後超界
    x1 = max(0, min(w, int(round(x / scale))))
    y1 = max(0, min(h, int(round(y / scale))))
    x2 = max(0, min(w, int(round((x + bw) / scale))))
    y2 = max(0, min(h, int(round((y + bh) / scale))))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return x1, y1, x2, y2


def resolve_roi(first_frame: np.ndarray, key: str, force_ui: bool,
                win_cfg: WindowConfig, width: int, height: int) -> tuple[Roi, bool]:
    """
    決定這次要分析的範圍，回傳 ((x1, y1, x2, y2), 是否為全畫面)。

    加了 --ui 就重新框；沒加就沿用 roi.json 存的。
    如果這個來源從來沒框過，一樣會跳出視窗讓你框——沒框過就直接分析全畫面的話，
    會把天空、路邊也當成路面拿去分類，跑出一堆沒用的結果。
    框完自動存檔，之後就不會再問。

    影片模式和即時模式共用，兩邊行為才會一致。
    """
    saved = None if force_ui else load_roi(key, width, height)
    if saved is not None:
        print(f"沿用上次框選的範圍: ({saved[0]}, {saved[1]}) → ({saved[2]}, {saved[3]})"
              f"　[{key}]")
        return saved, saved == (0, 0, width, height)

    if not force_ui:
        print(f"[ROI] {key} 還沒框過，跳出視窗讓你框一次（之後就不用了）")

    roi = select_roi(first_frame, win_cfg)
    if roi is None:
        print("未選取範圍，這次使用全畫面（沒有存檔）")
        return (0, 0, width, height), True

    print(f"分析範圍: ({roi[0]}, {roi[1]}) → ({roi[2]}, {roi[3]})")
    save_roi(key, roi, width, height)
    return roi, roi == (0, 0, width, height)
