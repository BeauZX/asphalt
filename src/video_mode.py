"""mode: video — 讀影片檔逐幀分析。"""

import cv2
from ultralytics import YOLO

from .config import Config, LIVE_WINDOW_TITLE, RunOptions
from .grid import GridTracker, classify_grid, draw_grid_overlay
from .recorder import SegmentWriter
from .roi import resolve_roi, video_key


def run(cfg: Config, opts: RunOptions) -> None:
    model = YOLO(str(cfg.model))

    cap = cv2.VideoCapture(str(cfg.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {cfg.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("無法讀取影片第一幀")

    (rx1, ry1, rx2, ry2), _is_full = resolve_roi(
        first_frame, video_key(cfg.video), opts.ui, cfg.window, orig_w, orig_h)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)     # 框選時已經讀掉第一幀，倒回開頭重跑

    tracker = GridTracker(cfg.grid.rows, cfg.grid.cols,
                          cfg.grid.ema_alpha, cfg.grid.switch_margin)
    seg = SegmentWriter(cfg.output_dir, fps, (orig_w, orig_h),
                        cfg.segment_seconds) if opts.save else None

    frame_idx = 0
    interrupted = False
    show = opts.ui      # 有開視窗才一邊處理一邊顯示

    # 包在 try/finally 裡：不管正常跑完、Ctrl+C 中斷、還是中途出錯，
    # 都會走到 finally 把影片正常收檔（原因見 SegmentWriter 的說明）
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            roi_crop = frame[ry1:ry2, rx1:rx2]
            probs, names = classify_grid(model, roi_crop,
                                         cfg.grid.rows, cfg.grid.cols, cfg.grid.imgsz)
            grid = tracker.update(probs, names)

            roi_annotated = draw_grid_overlay(roi_crop.copy(), grid,
                                              cfg.grid.rows, cfg.grid.cols)
            frame[ry1:ry2, rx1:rx2] = roi_annotated

            if seg:
                seg.write(frame)

            if show:
                cv2.namedWindow(LIVE_WINDOW_TITLE, cv2.WINDOW_NORMAL)
                cv2.imshow(LIVE_WINDOW_TITLE, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"Processed {frame_idx} frames...")

    except KeyboardInterrupt:
        interrupted = True
        print("\n偵測到 Ctrl+C，正在把目前這一段存檔...")
    finally:
        cap.release()
        if seg:
            seg.close()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    if interrupted:
        done_sec = frame_idx / fps if fps else 0
        print(f"已中斷。共處理 {frame_idx} 幀（相當於影片的前 {done_sec:.1f} 秒），"
              f"已存的片段都可以正常播放。")
    else:
        print(f"Done. {frame_idx} frames processed.")
    if seg:
        print(f"影片分段輸出到: {cfg.output_dir}")
