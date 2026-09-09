"""讀取 config.yaml，並把設定整理成程式好用的形式。"""

from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# 0: dry_asphalt_severe  → 紅
# 1: dry_asphalt_slight  → 黃
# 2: dry_asphalt_smooth  → 綠
COLORS = [
    (0, 0, 255),    # 紅 - severe
    (0, 255, 255),  # 黃 - slight
    (0, 255, 0),    # 綠 - smooth
]

# 視窗標題只能用 ASCII：OpenCV 5.0 的 Qt 後端內部是用視窗名稱去查找視窗，
# 名稱含中文時查不到、回傳空指標，setMouseCallback 就會炸掉：
#   cv2.error: (-27:Null pointer) NULL window handler in function 'setMouseCallbackImpl'
# 中文操作說明改用 print 印在終端機。
ROI_WINDOW_TITLE = "Select ROI - drag, ENTER to confirm, C to cancel"
LIVE_WINDOW_TITLE = "Road Classification (press q to quit)"


@dataclass
class GridConfig:
    rows: int
    cols: int
    imgsz: int
    ema_alpha: float
    switch_margin: float


@dataclass
class CameraConfig:
    id: int
    width: int
    height: int
    fps: int
    exposure: str
    awb: str
    torch_threads: int


@dataclass
class WindowConfig:
    roi_max_w: int
    roi_max_h: int


@dataclass
class RunOptions:
    """
    「這次怎麼跑」——由命令列決定，不放在 config.yaml。
    這樣同一件事不會有兩個地方能設定而互相打架。
    """
    live: bool      # True = 接 CSI 鏡頭，False = 跑影片檔
    ui: bool        # True = 開視窗重新框選路面範圍；False = 沿用 roi.json 存的
    save: bool      # 是否把結果錄成影片


@dataclass
class Config:
    """「跑起來用什麼參數」——全部來自 config.yaml。"""
    model: Path
    video: Path
    output_dir: Path
    segment_seconds: float
    grid: GridConfig
    camera: CameraConfig
    window: WindowConfig


def _resolve(path_str: str) -> Path:
    """設定檔裡的相對路徑一律相對於專案根目錄，這樣在哪個目錄下執行都一樣。"""
    p = Path(path_str)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(path: Path | str = DEFAULT_CONFIG_PATH,
                live: bool = False) -> Config:
    """
    讀取 config.yaml。live 只用來決定要不要檢查影片檔存不存在
    （接相機時根本用不到 video 那個設定）。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到設定檔: {path}")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    g = raw.get("grid", {})
    c = raw.get("camera", {})
    w = raw.get("window", {})

    cfg = Config(
        model=_resolve(raw.get("model", "model/asphalt_prep_v2_best.pt")),
        video=_resolve(raw.get("video", "")),
        output_dir=_resolve(raw.get("output_dir", "output_videos")),
        segment_seconds=float(raw.get("segment_seconds", 60)),
        grid=GridConfig(
            rows=int(g.get("rows", 3)),
            cols=int(g.get("cols", 5)),
            imgsz=int(g.get("imgsz", 128)),
            ema_alpha=float(g.get("ema_alpha", 0.45)),
            switch_margin=float(g.get("switch_margin", 0.1)),
        ),
        camera=CameraConfig(
            id=int(c.get("id", 0)),
            width=int(c.get("width", 1280)),
            height=int(c.get("height", 720)),
            fps=int(c.get("fps", 25)),
            exposure=str(c.get("exposure", "normal")),
            awb=str(c.get("awb", "auto")),
            torch_threads=int(c.get("torch_threads", 2)),
        ),
        window=WindowConfig(
            roi_max_w=int(w.get("roi_max_w", 1600)),
            roi_max_h=int(w.get("roi_max_h", 900)),
        ),
    )

    # 早點擋掉明顯寫錯的值，不然要等模型載入完、跑到一半才出現莫名其妙的錯誤
    if cfg.grid.rows < 1 or cfg.grid.cols < 1:
        raise ValueError("grid.rows / grid.cols 必須至少是 1")
    if cfg.segment_seconds <= 0:
        raise ValueError("segment_seconds 必須大於 0")
    if not cfg.model.exists():
        raise FileNotFoundError(f"找不到模型檔: {cfg.model}")
    if not live and not cfg.video.exists():
        raise FileNotFoundError(f"找不到影片檔: {cfg.video}（config.yaml 的 video 設定）")
    return cfg
