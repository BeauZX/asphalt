# 路面劣化分級

用 YOLO 分類模型把路面切成網格逐格判讀劣化程度，疊上顏色標註後錄影存檔。
可以跑影片檔，也可以接樹莓派的 CSI 鏡頭即時辨識。
推論跑在 Hailo-8 AI HAT 上（`.hef`），沒插 HAT 時可改用 CPU 跑 `.pt`。

分級與顏色：

| 類別 | 意義 | 顏色 |
|---|---|---|
| `dry_asphalt_severe` | 嚴重劣化 | 🔴 紅 |
| `dry_asphalt_slight` | 輕微劣化 | 🟡 黃 |
| `dry_asphalt_smooth` | 平整 | 🟢 綠 |

---

## 執行指令

先切到專案目錄：

```bash
cd /home/beau/Desktop/asphalt
```

### 接相機即時辨識

```bash
# 開視窗，用滑鼠框選路面範圍，框完按 Enter 開始
.venv/bin/python asphalt_main.py --live --ui

# 沿用上次框的範圍，不開視窗（車上無頭執行用這個）
.venv/bin/python asphalt_main.py --live
```

### 跑影片檔

影片路徑在 `config.yaml` 的 `video:` 設定。

```bash
# 開視窗重新框選
.venv/bin/python asphalt_main.py --ui

# 沿用上次框的範圍
.venv/bin/python asphalt_main.py
```

### 其他

```bash
# 只顯示不存檔
.venv/bin/python asphalt_main.py --live --ui --no-save

# 看說明
.venv/bin/python asphalt_main.py --help
```

**結束程式**：按 `q`（有開視窗時）或 `Ctrl+C`。
兩種方式都會把目前錄到的片段正常收檔，不會壞掉。

> **注意**：要用 `.venv/bin/python`，不要用系統的 `python3`。
> 系統 Python 沒有裝 ultralytics 和 torch，直接跑會報 `ModuleNotFoundError`。
> 如果終端機提示字元前面有 `(pothole)` 代表 venv 已啟動，那時打 `python3 asphalt_main.py` 也可以。

---

## 參數怎麼調

「這次怎麼跑」用命令列（`--live` / `--ui` / `--no-save`），
「跑起來用什麼參數」改 [`config.yaml`](config.yaml)：

| 設定 | 說明 |
|---|---|
| `model` | 模型路徑。副檔名決定後端：`.hef` 走 Hailo-8，`.pt` 走 CPU |
| `video` | 影片檔路徑（不加 `--live` 時才用到） |
| `output_dir` | 影片輸出資料夾 |
| `segment_seconds` | 每幾秒切一段 |
| `grid.rows` / `grid.cols` | 網格幾列幾欄 |
| `grid.imgsz` | 推論解析度（只對 `.pt` 有效，`.hef` 固定 256） |
| `grid.ema_alpha` | 時序平滑，越小越平滑、反應越慢 |
| `grid.switch_margin` | 切換類別所需的機率優勢，避免顏色閃爍 |
| `camera.id` | 用哪顆鏡頭（0 或 1） |
| `camera.width/height/fps` | 相機解析度與幀率 |
| `camera.exposure` | `sport` / `normal` / `long`，車速快糊掉就改 `sport` |
| `camera.torch_threads` | YOLO 用幾顆核心，調高畫面會卡（只對 `.pt` 有效） |

改完存檔，重新執行即生效。

---

## 框選範圍會記住

框選的路面範圍存在 `roi.json`，**只要框過一次，之後不用再框**。

- 存的是 0~1 的比例，所以改了解析度也不用重框，會自動換算
- cam0、cam1、每支影片檔各存一組，不會互相覆蓋
- 沒框過的來源，就算沒加 `--ui` 也會自動跳出視窗讓你框一次
- 鏡頭位置動過想重框：加 `--ui`

---

## 輸出

影片存到 `output_videos/`，依序編號：

```
output_videos/
  001.mp4
  002.mp4
  003.mp4
```

重複執行會**從現有的最大編號往後接**，不會蓋掉之前錄的。

---

## 程式結構

```
asphalt_main.py     進入點：解析命令列參數，分派到對應模式
config.yaml         參數設定
roi.json            框選的路面範圍（程式自動產生）
src/
  config.py         讀 config.yaml 並驗證
  backend.py        推論後端：Hailo-8（.hef）/ CPU YOLO（.pt），介面相同
  grid.py           網格切分、分類、上色、時序平滑
  analyzer.py       即時模式的背景分析執行緒
  roi.py            框選路面範圍 + roi.json 存讀
  camera.py         CSI 鏡頭讀取（透過 rpicam-vid）
  recorder.py       分段錄影
  video_mode.py     影片模式主流程
  live_mode.py      即時模式主流程
```

---

## 效能參考（Raspberry Pi 5 8GB + Hailo-8 AI HAT，實測）

| 項目 | Hailo-8（`.hef`） | CPU（`.pt`） |
|---|---|---|
| 3×5 網格單次分析 | 約 30 ms | 約 0.6 秒 |
| 4×10 網格單次分析 | 約 90 ms | 約 1.2 秒 |
| 即時模式畫面幀率 | 25 fps | 約 24.7 fps |
| 網格顏色更新 | 約 0.25 秒一次 | 約 1.4 秒一次 |
| 影片模式（2704×1520） | 約 11 fps（不存檔 21 fps） | 約 0.7 fps |

`.hef` 和 `.pt` 的判讀結果一致（210 格比對 argmax 全同，機率平均差 0.002）。

用 Hailo 時格數已不是瓶頸，可以放心加大 `grid.rows` / `grid.cols`。
即時模式網格更新頻率目前被「每幾幀餵一次分析」卡在約每秒 4 次，
不是被推論速度卡住。

用 CPU 時**瓶頸是「格數」不是解析度**，格數砍半、時間就砍半；
車速 30 km/h 時 1.4 秒約等於前進 12 公尺，正常車速建議把網格調小（例如 2×4）。

---

## 已知限制

- **影片模式沒做跳幀**：`asphalt.MP4` 是 240fps、3635 幀，用 Hailo 跑完約 5~6 分鐘，
  時間主要花在 `mp4v` 編碼 2704×1520 的畫面。
- **輸出影片可能看起來像靜止**：輸出沿用來源幀率，240fps 的影片在
  60Hz 螢幕上只能播 1/4 速度。
- **檔案偏大**：OpenCV 只支援 `mp4v` 這類舊編碼器，沒有 H.264，
  壓縮效率差。

---

## 環境需求

- Raspberry Pi OS（Bookworm 以上）
- `rpicam-apps`：`sudo apt install rpicam-apps`（即時模式用）
- Hailo：`sudo apt install hailo-all`（裝 HailoRT、驅動和 `python3-hailort`）

### 建立 Python 環境

```bash
uv venv --system-site-packages
uv sync
```

**`--system-site-packages` 不能省。** Hailo 的 Python 套件 `hailo_platform` 是 apt 裝在
系統 Python 3.13 的，不在 PyPI 上；venv 要加這個旗標才看得到它。
`pyproject.toml` 已設定 `python-preference = "only-system"`，uv 一定會用系統的
`/usr/bin/python3.13`，不會自己下載另一份（下載的那份看不到系統套件）。

如果之後刪掉 `.venv` 重建，記得還是要先 `uv venv --system-site-packages` 再 `uv sync`，
直接 `uv sync` 建出來的 venv 會少這個旗標，跑 `.hef` 時會報找不到 `hailo_platform`。
