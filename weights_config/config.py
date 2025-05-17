import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2

def get_face_weights_gui(face_imgs):

    print("請設定每張人臉的權重")

    # 主視窗
    root = tk.Tk()
    root.title("設定每張人臉的權重")

    sliders = [] # 滑桿
    per_row = 5  # 每列最大人臉量

    # 建立圖片與滑桿
    for idx, face_img in enumerate(face_imgs):
        frame = ttk.Frame(root, padding=10)
        frame.grid(row=idx // per_row, column=idx % per_row)

        img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        pil_img = pil_img.resize((150, 200))
        tk_img = ImageTk.PhotoImage(pil_img)

        label = ttk.Label(frame, text=f"Face {idx}")
        label.pack()
        image_label = ttk.Label(frame, image=tk_img)
        image_label.image = tk_img
        image_label.pack()

        scale = tk.Scale(frame, from_=0, to=100, orient='horizontal')
        scale.set(80)  # 預設 80分
        scale.pack()
        sliders.append(scale)

    # 儲存每張人臉權重
    result = []

    def submit():
        for s in sliders:
            result.append(s.get() / 100.0)
        root.destroy()

    # 確認按鈕
    total_rows = (len(face_imgs) - 1) // per_row + 1
    confirm_btn = ttk.Button(root, text="確認", command=submit)
    confirm_btn.grid(row=total_rows, column=0, columnspan=per_row, pady=10)

    root.mainloop()
    return result
