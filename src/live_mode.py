"""mode: live — 接 CSI 鏡頭即時辨識。"""

import sys
import time

import cv2
from ultralytics import YOLO

from .analyzer import GridAnalyzer
from .camera import Camera
from .config import Config, LIVE_WINDOW_TITLE, RunOptions
from .grid import draw_grid_overlay
from .recorder import SegmentWriter
from .roi import camera_key, resolve_roi


def _draw_status(frame, fps_shown: float, cfg: Config, analyzer: GridAnalyzer) -> None:
    """左上角疊一行狀態：實際幀率、網格尺寸、已分析次數與單次耗時。"""
    text = (f"{fps_shown:4.1f} fps | grid {cfg.grid.rows}x{cfg.grid.cols} "
            f"#{analyzer.update_count} ({analyzer.last_ms:.0f} ms)")
    for color, thickness in (((0, 0, 0), 3), ((255, 255, 255), 1)):   # 先黑描邊再白字
        cv2.putText(frame, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, thickness, cv2.LINE_AA)


def run(cfg: Config, opts: RunOptions) -> None:
    import torch
    torch.set_num_threads(cfg.camera.torch_threads)  # 留 CPU 給讀取與編碼，避免畫面卡頓

    model = YOLO(str(cfg.model))
    cam = Camera(cfg.camera)
    print(f"相機 cam{cfg.camera.id} 已啟動：{cfg.camera.width}×{cfg.camera.height} "
          f"@ {cfg.camera.fps}fps（曝光 {cfg.camera.exposure} / 白平衡 {cfg.camera.awb}）")

    analyzer = None
    seg = None
    interrupted = False
    frame_idx = 0
    t_start = time.perf_counter()

    try:
        first_frame = cam.read()
        if first_frame is None:
            raise RuntimeError(
                f"讀不到 cam{cfg.camera.id} 的畫面，請確認鏡頭有接好"
                f"（可用 rpicam-hello --list-cameras 檢查）")

        # 框選路面範圍：跟影片模式用同一個 resolve_roi()，操作方式完全一致
        (rx1, ry1, rx2, ry2), roi_is_full = resolve_roi(
            first_frame, camera_key(cfg.camera.id), opts.ui, cfg.window,
            cfg.camera.width, cfg.camera.height)

        analyzer = GridAnalyzer(model, cfg.grid)
        analyzer.start()

        if opts.save:
            seg = SegmentWriter(cfg.output_dir, cfg.camera.fps,
                                (cfg.camera.width, cfg.camera.height), cfg.segment_seconds)

        # 每隔幾幀才餵一幀給分析緒。分析遠慢於相機，餵太密只是白做工被覆寫掉
        submit_every = max(1, cfg.camera.fps // 4)
        fps_shown = 0.0
        t_start = last_t = time.perf_counter()
        last_n = 0
        print("開始即時辨識。" + ("按 q 或 " if opts.ui else "按 ") + "Ctrl+C 結束。")

        while True:
            frame = cam.read()
            if frame is None:
                print("\n相機串流結束（可能是 CSI 排線接觸不良）", file=sys.stderr)
                break

            roi_crop = frame[ry1:ry2, rx1:rx2]
            if frame_idx % submit_every == 0:
                # copy() 是必要的：frame 下一輪會被覆寫，而分析緒可能還在用這塊記憶體
                analyzer.submit(roi_crop.copy())

            grid = analyzer.latest()
            if grid is not None:
                annotated = draw_grid_overlay(roi_crop.copy(), grid,
                                              cfg.grid.rows, cfg.grid.cols)
                if roi_is_full:
                    frame = annotated
                else:
                    frame[ry1:ry2, rx1:rx2] = annotated
                    # 白框標出分析範圍，方便確認框對地方了
                    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 255, 255), 1)

            now = time.perf_counter()
            if now - last_t >= 1.0:
                fps_shown = (frame_idx - last_n) / (now - last_t)
                last_t, last_n = now, frame_idx
            _draw_status(frame, fps_shown, cfg, analyzer)

            if seg:
                seg.write(frame)

            if opts.ui:
                cv2.imshow(LIVE_WINDOW_TITLE, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1

    except KeyboardInterrupt:
        interrupted = True
        print("\n偵測到 Ctrl+C，正在收尾存檔...")
    finally:
        if analyzer:
            analyzer.stop()
            analyzer.join(timeout=5)
        if seg:
            seg.close()
        cam.close()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    elapsed = time.perf_counter() - t_start
    avg = frame_idx / elapsed if elapsed > 0 else 0
    print(f"{'已中斷。' if interrupted else ''}共 {frame_idx} 幀 / {elapsed:.0f} 秒"
          f"（平均 {avg:.1f} fps）"
          + (f"，網格更新 {analyzer.update_count} 次" if analyzer else ""))
    if seg:
        print(f"影片分段輸出到: {cfg.output_dir}")
