# 區域判斷與觸發邏輯

`final_model_test.py` 結合攝影機物件偵測、手動框選範圍、區域重疊比例計算，以及計時與觸發執行緒。

| 檔案 | 用途 |
| --- | --- |
| `final_model_test.py` | 偵測區域與觸發流程的開發程式 |
| `efficientdet.tflite` | 程式指定載入的模型 |
| `labels.txt` | 類別名稱對應 |
| `efficientdet_edgetpu.log` | Edge TPU 模型編譯紀錄 |

## 實作狀態

這份程式仍有未完成的邏輯：

- 計時的時間差方向、狀態變數名稱及執行緒重設需要修正。
- `shocking_dog()` 目前只輸出訊息，未包含 GPIO 呼叫。
- `obj.id == 3` 與現有標籤內容對應為 card，尚未與人／狗辨識整合。
- `get_iou()` 實際計算交集占預測框的比例，並非標準 IoU。

此目錄保留原始程式內容，不能直接作為完整驅離系統執行。
