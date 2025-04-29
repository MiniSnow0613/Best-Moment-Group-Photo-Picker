import cv2
from retinaface import RetinaFace

img_path = '001.jpg'

img_bgr = cv2.imread(img_path)

if img_bgr is None:
    print("無法讀取圖片：", img_path)
    exit()

results = RetinaFace.detect_faces(img_path)

print("偵測到的臉部數量：", len(results))

for key in results:
    face = results[key]
    x1, y1, x2, y2 = face['facial_area']
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("RetinaFace Result", img_bgr)
key = cv2.waitKey()
if key == ord("q"):
   print("exit")
cv2.destroyAllWindows()
