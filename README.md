# Best-Moment-Group-Photo-Picker

## Overview

This project implements an automated photo quality evaluation system that analyzes group photos by detecting faces and assessing multiple attributes such as smile intensity, eye openness, gaze direction, image composition, and sharpness. The goal is to help users quickly select the best photos from group shots.

---

## Prerequisites

### Development Environment

* Python 3.10

### Required Packages

Install all dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Key dependencies include (versions may vary):

```
mediapipe==0.10.21
numpy==1.26.4
opencv-contrib-python==4.11.0.86
opencv-python==4.7.0
Pillow==9.4.0
requests==2.32.3
retina-face==0.0.17
scikit-learn==1.6.1
scipy==1.15.2
torch==2.7.0
torchvision==0.22.0
ultralytics==8.3.120
```

---

## Usage

1. Prepare multiple group photos you want to evaluate and place them into the folder named `group_photos`.
2. Run the script `main.py`.
3. For each photo, the program will automatically detect and crop every face, then display each cropped face one by one.
4. For each detected face, adjust the **weight** according to its importance. A higher weight means the face’s expression has more influence on the photo’s overall score.
5. Sometimes the system may detect strangers or unwanted faces; in this case, set the weight to **0** to exclude that face from contributing to the score.
6. After adjusting the weights for all faces in the photo, click **Confirm** to proceed.
7. Repeat the adjustment process for every photo in the folder.
8. Once all photos have been processed, the program will output the score for each photo.
9. You can then select the photo with the highest score as the final choice.

---

## Hyperparameters

* Smile score weight: 0.20
* Eye openness weight: 0.40
* Gaze score weight: 0.15
* Blur score weight: 0.15
* Composition score weight: 0.10

  *(These weights can be adjusted in the main script.)*

---

## Experiment Results

We conducted an evaluation using 10 different group sets, each containing 5 photos. For each set, participants voted to select the most visually appealing group photo as the ground truth standard.

We then scored all photos using both our proposed multi-factor model and a baseline method relying solely on smile detection. By comparing the scores with the human voting results, we calculated the accuracy of selecting the best photo.

The baseline method achieved an accuracy of 46%, whereas our proposed model improved this to 76%, demonstrating a significant performance gain.

This result indicates that incorporating multiple facial attributes and image quality metrics leads to more reliable and human-aligned photo quality assessment compared to using smile detection alone.

---
