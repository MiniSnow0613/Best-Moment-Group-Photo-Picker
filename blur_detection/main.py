from model import compute_blur_score, blur_score_to_100

face_img = "grup_image.jpg"
blur_raw = compute_blur_score(face_img)
blur_score = blur_score_to_100(blur_raw)

print(f"模糊原始分數：{blur_raw:.4f}")
print(f"模糊清晰度（0~100）：{blur_score:.2f}")
