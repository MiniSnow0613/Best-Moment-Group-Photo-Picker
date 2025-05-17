from detector import detect_and_crop_faces
import cv2

img_path = 'grup_image.jpg'  # 換成你要測試的圖片
faces, boxes = detect_and_crop_faces(img_path)

print(f"共偵測到 {len(faces)} 張人臉")

for i, face in enumerate(faces):
    cv2.imshow(f"Face {i}", face)
    cv2.imwrite(f"face_{i}.jpg", face)  # 儲存裁切後的圖片

cv2.waitKey(0)
cv2.destroyAllWindows()
