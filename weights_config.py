import cv2

def get_face_weights(face_imgs):
    weights = []
    
    # 對每一個人臉遍歷，顯示人臉圖片以及滑桿，要求使用者設定權重
    for idx, face_img in enumerate(face_imgs):
        window_name = f"Set Weight for Face {idx}"
        cv2.namedWindow(window_name)
        cv2.imshow(window_name, face_img)

        # 占位函數，使滑桿變動時不做任何操作
        def nothing(x):
            pass

        # 創建一個滑桿，設定人臉權重(0~100)
        cv2.createTrackbar('Weight x100', window_name, 100, 100, nothing)  # 預設 1.0
        print(f"請使用滑桿設定第 {idx} 張人臉的權重，按 Enter 確認")

        # 等待使用者使用滑桿設定權重並按下 Enter 鍵
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter 鍵
                weight_val = cv2.getTrackbarPos('Weight x100', window_name)
                weight = weight_val / 100.0
                weights.append(weight)
                break

        cv2.destroyWindow(window_name)

    return weights
