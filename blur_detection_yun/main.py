import cv2
import numpy as np
import blur_detection  # 假設 evaluate_face_blur_array 寫在這個檔案中

if __name__ == "__main__":
    # 載入 DNN 模型
    prototxt_path = "./models/deploy.prototxt"
    model_path = "./models/res10_300x300_ssd_iter_140000.caffemodel"
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    # 讀取一張圖並轉成 array（你可以用任意圖像來源）
    image = cv2.imread("222.jpg")
    
    # ✅ 正確呼叫模組中的函式
    blur_score = blur_detection.evaluate_face_blur_array(image, net)

    if blur_score is not None:
        print(f"✅ 模糊度評分（人臉平均）：{blur_score:.2f}")
    else:
        print("❌ 圖中未偵測到人臉")
