"""CSI 鏡頭讀取（透過 rpicam-vid 驅動 libcamera）。"""

import subprocess

import cv2
import numpy as np

from .config import CameraConfig


class Camera:
    """
    讀 IMX219 這類 CSI 鏡頭，每次 read() 回傳一張 BGR 畫面。

    為什麼不用 cv2.VideoCapture：
    CSI 鏡頭的感光元件送出來的是原始拜爾馬賽克資料（SBGGR10），要經過去馬賽克、
    黑階校正、白平衡、色彩矩陣、gamma 等 ISP 處理，才會變成能看的彩色畫面。
    USB 攝影機自己內建 ISP 所以吐出來就能用，CSI 鏡頭則是把這件事交給主機端，
    在 Pi 5 上由 libcamera 負責。
    /dev/video0 那個 rp1-cfe 只是「接收原始資料的硬體入口」，給的是還沒處理的 RAW，
    所以 OpenCV 打得開裝置卻讀不到可用畫面（實測 isOpened()=True 但 read() 回 False）。
    因此改用 rpicam-vid 驅動 libcamera，讓它把處理好的 YUV420 吐到 stdout。

    曝光相關參數沿用 imx_video/rpi5_dual_camera_capture.py 實測過的組合。
    """

    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        self.frame_size = cfg.width * cfg.height * 3 // 2   # YUV420 每幀位元組數
        cmd = [
            "rpicam-vid",
            "--camera", str(cfg.id),
            "--width", str(cfg.width),
            "--height", str(cfg.height),
            "--framerate", str(cfg.fps),
            "--level", "4.2",               # 官方建議高 fps 要設這個
            "--denoise", "cdn_off",         # 關掉顏色降噪，大幅降低 CPU 負擔
            "--codec", "yuv420",            # 不編碼，直接吐原始畫面給 Python 處理
            "--exposure", cfg.exposure,     # 自適應曝光，不鎖定固定快門/增益
            "--awb", cfg.awb,
            "-n",                           # 不開 rpicam 自己的預覽視窗，省 CPU
            "--timeout", "0",               # 0 = 一直錄到我們主動結束
            "-o", "-",
        ]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL)

    def read(self) -> np.ndarray | None:
        """讀一幀並轉成 BGR；相機結束或資料不完整時回傳 None。"""
        buf = self.proc.stdout.read(self.frame_size)
        if len(buf) < self.frame_size:
            return None
        yuv = np.frombuffer(buf, np.uint8).reshape(self.cfg.height * 3 // 2, self.cfg.width)
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
