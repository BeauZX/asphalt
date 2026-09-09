#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路面劣化分級 — 進入點。

執行方式:
    python3 asphalt_main.py --live --ui    接相機，開視窗重新框選路面範圍
    python3 asphalt_main.py --live         接相機，沿用上次框的範圍（不開視窗）
    python3 asphalt_main.py --ui           跑影片檔，開視窗重新框選
    python3 asphalt_main.py                跑影片檔，沿用上次框的範圍
    加 --no-save 則只顯示不存檔

框選的範圍會存進 roi.json（存的是 0~1 比例，換解析度也不用重框），
所以只要框過一次，之後加不加 --ui 都能跑。從沒框過的來源會自動跳出視窗讓你框。

命令列決定「這次怎麼跑」，config.yaml 決定「跑起來用什麼參數」
（模型路徑、網格大小、相機解析度、曝光、分段秒數…）。

程式碼拆在 src/ 底下：
    src/config.py     讀 config.yaml
    src/grid.py       網格切分、分類、上色、時序平滑
    src/analyzer.py   即時模式的背景分析執行緒
    src/roi.py        框選路面範圍 + roi.json 存讀
    src/camera.py     CSI 鏡頭讀取（rpicam-vid）
    src/recorder.py   分段錄影
    src/video_mode.py 影片模式主流程
    src/live_mode.py  即時模式主流程
"""

import argparse
import sys

from src.config import RunOptions, load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="路面劣化分級：可跑影片檔，也可接 CSI 鏡頭即時辨識",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""範例:
  python3 asphalt_main.py --live --ui   接相機，開視窗重新框選路面範圍
  python3 asphalt_main.py --live        接相機，沿用上次框的範圍（不開視窗）
  python3 asphalt_main.py --ui          跑影片檔，開視窗重新框選
  python3 asphalt_main.py               跑影片檔，沿用上次框的範圍

其他參數（網格大小、相機解析度、曝光…）請改 config.yaml
""")
    parser.add_argument("--live", action="store_true",
                        help="接 CSI 鏡頭即時辨識（不加則是跑 config.yaml 指定的影片檔）")
    parser.add_argument("--ui", action="store_true",
                        help="開視窗重新框選路面範圍，並即時顯示畫面"
                             "（不加則沿用 roi.json 存的範圍）")
    parser.add_argument("--no-save", action="store_true",
                        help="只顯示不存檔")
    args = parser.parse_args()

    opts = RunOptions(live=args.live, ui=args.ui, save=not args.no_save)

    try:
        cfg = load_config(live=opts.live)
    except (FileNotFoundError, ValueError) as e:
        print(f"設定錯誤: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"模式: {'相機即時辨識' if opts.live else '影片檔'} | "
          f"視窗: {'開（重新框選）' if opts.ui else '關（沿用上次框選）'} | "
          f"存檔: {'是' if opts.save else '否'} | "
          f"網格: {cfg.grid.rows}×{cfg.grid.cols} (imgsz {cfg.grid.imgsz})")

    # 這兩個模組會載入 ultralytics / torch，要好幾秒，
    # 所以放在設定檢查通過之後才 import，設定寫錯時能馬上得到回饋
    if opts.live:
        from src import live_mode
        live_mode.run(cfg, opts)
    else:
        from src import video_mode
        video_mode.run(cfg, opts)


if __name__ == "__main__":
    main()
