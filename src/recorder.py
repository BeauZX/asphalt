"""分段錄影：把畫面依序寫成 001.mp4、002.mp4…，不覆蓋既有檔案。"""

from pathlib import Path

import cv2
import numpy as np


def next_segment_index(output_dir: Path) -> int:
    """
    掃描 output_dir 底下已經存在的 NNN.mp4，回傳下一個可用的編號。

    這樣重複執行程式時不會蓋掉先前錄好的片段，編號會從現有的最大值往後接續。
    只認純數字檔名，其他名稱的 mp4 會被忽略、不影響編號。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    used = [int(p.stem) for p in output_dir.glob("*.mp4") if p.stem.isdigit()]
    return max(used) + 1 if used else 1


class SegmentWriter:
    """
    每滿 segment_seconds 秒就自動換一個輸出檔，影片和即時模式共用。

    務必記得呼叫 close()——mp4 的索引資訊（moov atom）是在關檔那一刻才寫進去的，
    沒關就結束的話，已經寫入的畫面全都讀不出來，檔案等於報廢。
    呼叫端把它包在 try/finally 裡，Ctrl+C 也能留下可以播放的片段。
    """

    def __init__(self, output_dir: Path, fps: float, frame_size: tuple[int, int],
                 segment_seconds: float):
        self.output_dir = output_dir
        self.fps = fps
        self.frame_size = frame_size
        self.seg_frames = max(1, int(round(fps * segment_seconds)))
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.next_idx = next_segment_index(output_dir)
        self.writer = None
        self.path = None
        self.frames_in_seg = 0
        print(f"分段輸出到: {output_dir}（每 {segment_seconds:.0f} 秒一段，"
              f"從 {self.next_idx:03d}.mp4 開始編號）")

    def write(self, frame: np.ndarray) -> None:
        if self.writer is None or self.frames_in_seg >= self.seg_frames:
            self._rotate()
        self.writer.write(frame)
        self.frames_in_seg += 1

    def _rotate(self) -> None:
        self._close_current()
        self.path = self.output_dir / f"{self.next_idx:03d}.mp4"
        self.writer = cv2.VideoWriter(str(self.path), self.fourcc, self.fps, self.frame_size)
        self.next_idx += 1
        self.frames_in_seg = 0

    def _close_current(self) -> None:
        if self.writer is not None:
            self.writer.release()
            print(f"已存檔: {self.path.name}")
            self.writer = None

    def close(self) -> None:
        self._close_current()
