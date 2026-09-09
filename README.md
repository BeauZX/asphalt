# 路面劣化分級

用 YOLO 分類模型把路面切成網格逐格判讀劣化程度，疊上顏色標註後錄影存檔。
可以跑影片檔，也可以接樹莓派的 CSI 鏡頭即時辨識。

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
> 如果終端機提示字元前面有 `(asphalt)` 代表 venv 已啟動，那時打 `python3 asphalt_main.py` 也可以。

---

## 參數怎麼調

「這次怎麼跑」用命令列（`--live` / `--ui` / `--no-save`），
「跑起來用什麼參數」改 [`config.yaml`](config.yaml)：

| 設定 | 說明 |
|---|---|
| `model` | 模型權重路徑 |
| `video` | 影片檔路徑（不加 `--live` 時才用到） |
| `output_dir` | 影片輸出資料夾 |
| `segment_seconds` | 每幾秒切一段 |
| `grid.rows` / `grid.cols` | 網格幾列幾欄 |
| `grid.imgsz` | 推論解析度 |
| `grid.ema_alpha` | 時序平滑，越小越平滑、反應越慢 |
| `grid.switch_margin` | 切換類別所需的機率優勢，避免顏色閃爍 |
| `camera.id` | 用哪顆鏡頭（0 或 1） |
| `camera.width/height/fps` | 相機解析度與幀率 |
| `camera.exposure` | `sport` / `normal` / `long`，車速快糊掉就改 `sport` |
| `camera.torch_threads` | YOLO 用幾顆核心，調高畫面會卡 |

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
  grid.py           網格切分、分類、上色、時序平滑
  analyzer.py       即時模式的背景分析執行緒
  roi.py            框選路面範圍 + roi.json 存讀
  camera.py         CSI 鏡頭讀取（透過 rpicam-vid）
  recorder.py       分段錄影
  video_mode.py     影片模式主流程
  live_mode.py      即時模式主流程
```

---

## 效能參考（Raspberry Pi 5 8GB，實測）

| 項目 | 數值 |
|---|---|
| 即時模式畫面幀率 | 約 24.7 fps |
| 網格顏色更新 | 約 1.4 秒一次 |
| 3×5 網格單次分析 | 約 0.6 秒（CPU 空閒時） |
| 4×10 網格單次分析 | 約 1.2 秒 |

Pi 5 沒有 GPU，全部靠 CPU 推論。**瓶頸是「格數」不是解析度**，
格數砍半、時間就砍半，所以要更即時就調小 `grid.rows` / `grid.cols`。

車速 30 km/h 時，1.4 秒約等於前進 12 公尺 ——
畫面上標的顏色其實是十幾公尺前那段路的判讀結果。慢速巡檢沒問題，
正常車速建議把網格再調小（例如 2×4）。

---

## 已知限制

- **跑影片檔很慢**：`asphalt.MP4` 是 240fps、3635 幀，跑完約 84 分鐘。
  還沒做跳幀處理。
- **輸出影片可能看起來像靜止**：輸出沿用來源幀率，240fps 的影片在
  60Hz 螢幕上只能播 1/4 速度。
- **檔案偏大**：OpenCV 只支援 `mp4v` 這類舊編碼器，沒有 H.264，
  壓縮效率差。

---

## 環境需求

- Raspberry Pi OS（Bookworm 以上）
- `rpicam-apps`：`sudo apt install rpicam-apps`（即時模式用）
- Python 套件：`uv sync`
