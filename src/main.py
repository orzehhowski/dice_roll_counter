import cv2
import numpy as np

for i in range(1, 4):

    # Load image
    img = cv2.imread(f'data/input/{i}.jpeg')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Preprocess
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    _, img_thresh = cv2.threshold(img_blur, 130, 255, cv2.THRESH_BINARY_INV)

    cv2.imwrite(f"data/output/thresh{i}.jpg", img_thresh)

    # Find contours (dice)
    contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:  # filter out small noise
            continue

        if area > 100000: # filter big areas 
            continue

        # Draw bounding box
        x, y, w, h = cv2.boundingRect(cnt)
        roi = img_thresh[y:y+h, x:x+w]

        # Draw on original image
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imwrite(f"data/output/{i}.jpg", img)