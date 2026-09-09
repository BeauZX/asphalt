"""
路面劣化分級 — 內部模組。

這裡放的是「必須在 cv2 被載入之前」執行的環境設定。
因為任何一支模組要用 cv2 都得先經過這個 package，把設定放這裡最保險——
不管是從 asphalt_main.py 進來，還是單獨 import src.grid 來測試，都會先跑到。
"""

import os

# OpenCV 內建的 Qt 只附了 X11 版的視窗元件（libqxcb.so），沒有 Wayland 版。
# 樹莓派桌面預設跑 Wayland，直接開窗會失敗：
#   qt.qpa.plugin: Could not find the Qt platform plugin "wayland"
#   cv2.error: (-27:Null pointer) NULL window handler
# Wayland 底下有 XWayland 相容層，指定走 X11 就能正常顯示。
# 用 setdefault 是為了讓使用者仍能從外部覆寫，例如 QT_QPA_PLATFORM=offscreen 跑無頭測試。
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
