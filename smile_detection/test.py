import cv2
from model import predict_smile_ratio, predict_smile_probs

img = cv2.imread("image.png")  # 換成你要測試的圖片
if img is None:
    print("無法讀取圖片")
else:
    result = predict_smile_probs(img)
    print("每種類別機率：")
    for k, v in result.items():
        print(f"{k}: {v:.2%}")
    ratio = predict_smile_ratio(img)
    print(f"笑容比例：{ratio:.2f}")
