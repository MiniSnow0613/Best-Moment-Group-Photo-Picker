import cv2
import os
import shutil

def compute_blur_score(gray_face):
    laplacian = cv2.Laplacian(gray_face, cv2.CV_64F)
    return laplacian.var()

def clear_folder(folder_path):
    """清空指定資料夾內容"""
    if os.path.exists(folder_path):
        for f in os.listdir(folder_path):
            file_path = os.path.join(folder_path, f)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    else:
        os.makedirs(folder_path)

def evaluate_face_blur_dnn(image_path, net, output_dir, conf_threshold=0.5):
    image = cv2.imread(image_path)
    if image is None:
        return None

    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 [104.0, 177.0, 123.0], False, False)

    net.setInput(blob)
    detections = net.forward()

    scores = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x1, y1, x2, y2) = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            face_roi = gray[y1:y2, x1:x2]

            if face_roi.size == 0:
                continue

            score = compute_blur_score(face_roi)
            scores.append(score)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{score:.1f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(out_path, image)

    if not scores:
        return None

    return sum(scores) / len(scores)

def process_folder(folder_path, output_dir="output"):
    print("✅ 使用 OpenCV DNN 人臉偵測")

    prototxt_path = "./models/deploy.prototxt"
    model_path = "./models/res10_300x300_ssd_iter_140000.caffemodel"
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    results = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(folder_path, filename)
            score = evaluate_face_blur_dnn(path, net, output_dir)
            if score is not None:
                results.append((filename, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results

if __name__ == "__main__":
    folder = "./blur_levels"
    output = "./output"

    clear_folder(output)  # ✅ 執行前清空 output/

    scores = process_folder(folder, output)

    print("📸 人臉區模糊度排序結果（由清晰到模糊）:")
    for fname, score in scores:
        print(f"{fname}: {score:.2f}")

    if scores:
        best, val = scores[0]
        print(f"\n✅ 最清晰人臉的合照是：{best}（分數：{val:.2f}）")
