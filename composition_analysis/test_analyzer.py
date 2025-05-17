from analyzer import score_photo

img_path = "grup_image.jpg"  # 換成你要測試的圖片
score = score_photo(img_path, area_threshold_ratio=0)
print(f"構圖分數: {score:.2f}")
