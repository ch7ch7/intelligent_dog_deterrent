# 攝影機物件偵測

使用 OpenCV 讀取攝影機影像，透過 PyCoral 載入 TensorFlow Lite 模型，取得物件類別、分數與位置框，並顯示辨識結果。

| 檔案 | 用途 |
| --- | --- |
| `edge_detect.py` | 攝影機輸入、模型推論與結果顯示 |
| `efficientdet_edgetpu.tflite` | 模型檔 |
| `label.txt` | people、dog 的類別名稱對應 |

## 處理流程

讀取標籤 → 載入模型 → 配置張量 → 讀取攝影機 → 調整輸入影像 → `engine.invoke()` → 取得物件偵測結果。

程式使用攝影機索引 0，以及相對路徑的模型與標籤檔。套件相依包含 OpenCV、NumPy、Pillow、PyCoral 與 requests；未確認完整版本組合，尚未重新執行驗證。
