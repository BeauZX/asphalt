# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Road surface degradation grading. A YOLO **classification** model (3 classes:
`dry_asphalt_severe` / `dry_asphalt_slight` / `dry_asphalt_smooth`) is applied
to each cell of a grid overlaid on the road area, then colour-coded (red/yellow/green)
and recorded to segmented video files.

Runs on a **Raspberry Pi 5** with a **Hailo-8 AI HAT** and two IMX219 CSI
cameras attached. Target use is in-vehicle road surveying. Inference runs on the
Hailo (`model/asphalt_cls.hef`); the original `.pt` is kept as a CPU fallback.

## Commands

```bash
cd /home/beau/Desktop/asphalt

.venv/bin/python asphalt_main.py --live --ui   # camera, re-frame the road ROI
.venv/bin/python asphalt_main.py --live        # camera, reuse saved ROI, no window
.venv/bin/python asphalt_main.py --ui          # video file, re-frame
.venv/bin/python asphalt_main.py               # video file, reuse saved ROI
.venv/bin/python asphalt_main.py --help
```

Add `--no-save` to display without recording. `q` (when a window is open) or
`Ctrl+C` stops; both paths close the video file properly.

Always use `.venv/bin/python` — the system `python3` has no ultralytics/torch.
There is no test suite, linter, or build step.

Rebuilding the venv: `uv venv --system-site-packages && uv sync`. The flag is
mandatory (see below); a bare `uv sync` on a missing `.venv` creates one without it.

## The platform constraints that shape the design

Every one of these was found by measurement on this specific machine. They are
the reason the code looks the way it does; don't "simplify" them away.

**`hailo_platform` only exists in the system Python 3.13.** It comes from the apt
package `python3-hailort` (cp313 `.so` in `/usr/lib/python3/dist-packages`), is
not on PyPI, and must match the installed `libhailort` version exactly. The venv
is therefore built with `--system-site-packages` (recorded in
`.venv/pyvenv.cfg` as `include-system-site-packages = true`) and
`pyproject.toml` pins `python-preference = "only-system"` so uv never substitutes
a downloaded 3.13 that cannot see the system packages. Everything else
(torch, ultralytics, opencv, numpy, pyyaml) is installed by uv into the venv and
shadows the system copies.

**Backend is chosen by the `model:` file extension** (`src/backend.py`
`load_classifier()`): `.hef` → `HailoClassifier`, `.pt` → `YoloClassifier`.
Both expose `names`, `__call__(cells) -> probs[N, C]`, `close()`. No separate
config key, so the same thing is never set in two places. The HEF carries no
class names, so `CLASS_NAMES` in `src/config.py` is the source of truth for
index→name; its order must match the `.pt`'s (alphabetical from training).

**HEF preprocessing must mirror ultralytics classify exactly**: resize shorter
side to 256, centre crop, BGR→RGB, raw uint8 (no mean/std). The HEF input is
`256×256×3 UINT8 NHWC`, output is a `softmax(3)` already in the graph. Verified
on 210 cells from the sample video: argmax agrees 100% with the `.pt`, mean
absolute probability difference 0.002. `grid.imgsz` and `camera.torch_threads`
only affect the `.pt` path.

**Keep the Hailo pipeline open for the process lifetime.** `HailoClassifier`
enters `activate()` and `InferVStreams` in `__init__` and exits them in
`close()`; reconfiguring per call costs hundreds of ms and would erase the
speedup. Both mode modules call `classifier.close()` in `finally`, and live mode
does so only after `analyzer.join()` because the analysis thread is the one
using the device. HailoRT writes `hailort.log` into the cwd on every run; it is
gitignored.

**No GPU.** `torch.cuda.is_available()` is always False. The `.pt` fallback is
CPU inference on 4 cores.

**On CPU, grid cell count — not resolution — is the bottleneck.** Measured with
NCNN and PyTorch alike: 4×10=40 cells takes ~1.2 s per pass, 3×5=15 cells ~0.6 s.
Dropping `imgsz` from 256 to 128 barely helps at 40 cells (per-cell fixed
overhead dominates) but does help once the cell count is low. NCNN export was
tried and only gave ~1.3× — not worth the extra dependency. On the Hailo the same
passes take ~30 ms and ~90 ms, so cell count is no longer a practical limit.

**Live mode must not analyse synchronously.** On CPU (~1.4 s per analysis) a
read→analyse→display loop produces a video that updates once per second and a
preview that is unusable; on the Hailo (~35 ms) a synchronous wait would still
drop frames. `GridAnalyzer` runs inference on a background thread while the main
loop displays and records at full camera rate, filling in with the most recent
grid result. `submit()` deliberately keeps only the newest frame and drops the
rest — queuing would make the displayed colours drift further and further behind
reality. With the Hailo the grid refresh (~4/s) is now bounded by `submit_every`
in `live_mode.py`, not by inference.

**CSI cameras cannot be read with `cv2.VideoCapture`.** IMX219 emits raw Bayer
(SBGGR10); the ISP pipeline that turns it into a usable image lives in libcamera.
`/dev/video0` (`rp1-cfe`) is only the raw-data intake — OpenCV reports
`isOpened()=True` but `read()` returns False. `src/camera.py` therefore spawns
`rpicam-vid --codec yuv420 -o -` and reads YUV420 frames from its stdout.

**OpenCV window titles must be ASCII.** OpenCV 5.0's Qt backend looks windows up
by name; a non-ASCII title makes the lookup return NULL and `setMouseCallback`
raises `(-27:Null pointer) NULL window handler`. Titles live in `src/config.py`
(`ROI_WINDOW_TITLE`, `LIVE_WINDOW_TITLE`) and are English on purpose. Chinese
instructions go to stdout instead.

**`QT_QPA_PLATFORM=xcb` must be set before `cv2` is imported.** The desktop runs
Wayland but OpenCV ships only `libqxcb.so`, so window creation fails outright.
The assignment lives in `src/__init__.py` precisely because importing any `src.*`
module triggers it before that module imports cv2. Do not move it into a function
or a module that imports cv2 at the top.

**`SegmentWriter.close()` is mandatory.** The mp4 moov atom is written at close;
skipping it makes every already-written frame unreadable. Both mode modules wrap
their loop in `try/except KeyboardInterrupt/finally` so Ctrl+C still yields
playable segments.

**OpenCV here has no H.264.** Only `mp4v`, `XVID` and `MJPG` open successfully;
`avc1`/`H264` fail. That is why output files are large. ffmpeg with libx264 is
available on the system if piping is ever needed.

## Architecture

`asphalt_main.py` only parses argv, loads config, and dispatches. Both mode
modules take `(cfg, opts)`.

The split between the two settings sources is deliberate, so the same thing is
never configurable in two places:

- **`RunOptions`** (from argv: `--live`, `--ui`, `--no-save`) — *how this run behaves*
- **`Config`** (from `config.yaml`) — *what parameters it uses*

`config.yaml` has no `mode`/`ui`/`save` keys; adding them back would recreate that
conflict.

Shared pieces both modes route through, so their behaviour stays identical:

- `src/backend.py` — `load_classifier(cfg)` returns the Hailo or CPU classifier;
  the rest of the code only sees the common interface.
- `src/grid.py` — `classify_grid()`, `draw_grid_overlay()`, and `GridTracker`,
  which holds the EMA smoothing plus hysteresis switching. Without it, cells flicker
  between visually similar classes. Video mode calls `update()` per frame; live mode
  calls it per analysis. Same logic, different cadence.
- `src/roi.py` — `resolve_roi()` is the single entry point. It reuses the ROI saved
  in `roi.json` unless `--ui` was passed, and opens the framing window if this source
  was never framed (analysing the full frame would feed sky and roadside into the
  classifier).
- `src/recorder.py` — `SegmentWriter` rotates output every `segment_seconds` and
  scans `output_dir` at startup so numbering continues past existing files instead
  of overwriting them.

`roi.json` stores **0–1 fractions**, not pixels, so a resolution change does not
invalidate a framing. Keys are per source (`camera0`, `video:asphalt.MP4`), so the
two cameras and each video file keep separate regions.

`src/config.py` validates before either mode module is imported — model file exists,
video file exists (video mode only), grid dimensions sane. Model loading takes tens
of seconds, so validating first means a typo in the YAML surfaces immediately.
That is also why the mode modules are imported inside the `if` in `main()`.

## Camera settings

Exposure settings in `config.yaml` mirror those in
`/home/beau/Desktop/imx_video/rpi5_dual_camera_capture.py` (a separate,
unrelated dual-camera recording script the user owns — **reference only, never
modify it**). Adaptive AEC/AGC with no fixed shutter/gain, `--awb auto`, and
`--denoise cdn_off` for CPU. `exposure: sport` is the documented switch if motion
blur smears the road texture at speed.

`camera.torch_threads` caps YOLO's core usage (`.pt` only) so recording and
encoding keep their share; raising it makes the preview stutter.

## Measured performance

Hailo (`.hef`): live mode 25 fps display/record, ~34 ms per 3×5 analysis, grid
refresh ~0.25 s (bounded by `submit_every`). Video mode on the sample footage
(`asphalt.MP4`, 2704×1520 @ 239.76 fps, 3635 frames) runs ~11 fps with recording,
~21 fps without — the `mp4v` encode is now the cost, so the full file takes ~5–6 min.

CPU (`.pt`): live mode ~24.7 fps, grid colours refresh ~1.4 s, no thermal
throttling at ~57 °C. At 30 km/h that lag is ~12 m of road, so displayed colours
describe road that is already behind the vehicle — reduce `grid.rows`/`grid.cols`
if that matters. Video mode on the sample footage takes roughly 84 minutes.

Known unfixed issues, listed in README.md: no frame skipping in video mode,
output inherits the 240 fps source rate (so it appears frozen on a 60 Hz
display), and files are large because of the `mp4v` codec.

## Working style

The user wants changes discussed and explicitly approved before files are edited.
Investigation, measurement and running commands are fine beforehand — those make
the discussion concrete — but hold off on writing until they agree.
