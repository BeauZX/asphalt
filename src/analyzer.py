"""即時模式用的背景分析執行緒。"""

import sys
import threading
import time

import numpy as np

from .config import GridConfig
from .grid import GridTracker, classify_grid


class GridAnalyzer(threading.Thread):
    """
    在背景持續分析「最新的一幀」，主緒隨時可以取走最近一次算好的網格結果。

    為什麼需要它：CPU 推論時分析一次網格要 1~2 秒。如果照影片模式那樣
    「讀一幀 → 分析 → 顯示」，畫面就會變成每 1~2 秒才動一下，根本沒辦法看。
    拆成背景緒之後，主緒維持相機的原始幀率顯示與錄影，網格顏色則用
    「最近一次算好的結果」填補，等分析緒算完再換上新的。
    換成 Hailo 之後單次分析只要幾十毫秒，但主緒仍不該同步等：
    等待時間會直接變成掉幀。

    submit() 只保留最新一幀、舊的直接丟掉：分析若比相機慢，
    排隊處理會越積越舊，畫面上標的顏色會離現況越來越遠。
    寧可跳過中間那些幀，也要讓每次分析都是當下的路況。
    """

    def __init__(self, classifier, grid_cfg: GridConfig):
        super().__init__(daemon=True)
        self.classifier = classifier
        self.cfg = grid_cfg
        self.tracker = GridTracker(grid_cfg.rows, grid_cfg.cols,
                                   grid_cfg.ema_alpha, grid_cfg.switch_margin)

        self._pending = None
        self._lock = threading.Lock()
        self._new_frame = threading.Event()
        # 不能取名 _stop：會蓋掉 threading.Thread 內部的 _stop() 方法，join() 會壞掉
        self._stop_evt = threading.Event()

        self._grid = None
        self._grid_lock = threading.Lock()
        self.update_count = 0       # 已完成幾次分析
        self.last_ms = 0.0          # 最近一次分析花了幾毫秒

    def submit(self, roi_crop: np.ndarray) -> None:
        """主緒呼叫：丟最新一幀進來分析（會覆寫掉還沒被處理的舊幀）。"""
        with self._lock:
            self._pending = roi_crop
        self._new_frame.set()

    def latest(self) -> list[list[tuple]] | None:
        """主緒呼叫：取最近一次的網格結果；第一次分析還沒完成時回傳 None。"""
        with self._grid_lock:
            return self._grid

    def stop(self) -> None:
        self._stop_evt.set()
        self._new_frame.set()       # 叫醒可能正在等待的 wait()

    def run(self) -> None:
        while not self._stop_evt.is_set():
            self._new_frame.wait(timeout=1.0)
            if self._stop_evt.is_set():
                break
            with self._lock:
                frame = self._pending
                self._pending = None
            self._new_frame.clear()
            if frame is None:
                continue

            t0 = time.perf_counter()
            try:
                probs, names = classify_grid(self.classifier, frame,
                                             self.cfg.rows, self.cfg.cols)
            except Exception as e:
                # 單次推論失敗不該讓整個錄影中斷，記錄後繼續等下一幀
                print(f"[分析] 推論失敗: {e}", file=sys.stderr)
                time.sleep(0.5)
                continue

            grid = self.tracker.update(probs, names)
            with self._grid_lock:
                self._grid = grid
            self.update_count += 1
            self.last_ms = (time.perf_counter() - t0) * 1000.0
