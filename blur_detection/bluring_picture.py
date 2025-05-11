import cv2
import numpy as np
import os
import shutil  # ✅ 用來刪除整個資料夾內容

def compute_blur_score(gray_img):
    laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
    return laplacian.var()

def clear_folder(folder_path):
    """清空資料夾內容"""
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # 刪除檔案或符號連結
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # 刪除資料夾
            except Exception as e:
                print(f"⚠️ 刪除失敗：{file_path} → {e}")
    else:
        os.makedirs(folder_path)

def blur_experiment(image_path, output_folder="blur_levels"):
    clear_folder(output_folder)  # ✅ 每次執行前先清空資料夾

    image = cv2.imread(image_path)
    if image is None:
        print("❌ 圖片讀取失敗")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(f"🔍 原圖 Laplacian 模糊分數：{compute_blur_score(gray):.2f}")

    blur_levels = [1, 3, 5, 7, 11, 15, 21]

    for k in blur_levels:
        blurred_color = image.copy() if k == 1 else cv2.GaussianBlur(image, (k, k), 0)
        gray_blurred = cv2.cvtColor(blurred_color, cv2.COLOR_BGR2GRAY)
        score = compute_blur_score(gray_blurred)

        output_path = os.path.join(output_folder, f"blur_k{k}_score{score:.1f}.jpg")
        cv2.imwrite(output_path, blurred_color)

        print(f"💡 模糊 kernel={k} → 模糊分數={score:.2f}")

    print(f"\n✅ 所有彩色模糊圖已儲存於：{output_folder}/")

if __name__ == "__main__":
    blur_experiment("222.jpg")
