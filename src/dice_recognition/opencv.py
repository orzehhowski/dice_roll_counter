import cv2
import numpy as np
import os

# (I decided to set those as env vars because they strongly depend on photo spot conditions)
# expected x / y of detected dice object - depends on shadow direction 
EXPECTED_ASPECT_RATIO = 1
# what difference from expected value we accept
ASPECT_RATIO_DELTA = 0.2

MIN_DICE_AREA = 5000
MAX_DICE_AREA = 100000

INPUT_DATA_DIR = "data/input"
OUTPUT_DATA_DIR = "data/output"

# Merge overlapping boxes
def merge_boxes(boxes):
    merged = []
    for b in boxes:
        x1, y1, x2, y2 = b
        aspect_ratio = abs(x2 - x1) / abs(y2 - y1)
        merged_flag = False
        # try merging only if aspect ratio doesn't match the expected
        if abs(aspect_ratio - EXPECTED_ASPECT_RATIO) > ASPECT_RATIO_DELTA:
            for i, (mx1, my1, mx2, my2) in enumerate(merged):
                aspect_ratio_merged = abs(mx2 - mx1) / abs(my2 - my1)
                if abs(aspect_ratio_merged - EXPECTED_ASPECT_RATIO) > ASPECT_RATIO_DELTA:
                    if not (x2 < mx1 or x1 > mx2 or y2 < my1 or y1 > my2):  # overlap
                        merged[i] = (min(x1,mx1), min(y1,my1), max(x2,mx2), max(y2,my2))
                        merged_flag = True
                        break
                    
        if not merged_flag:
            merged.append(b)
    return merged

def process_data():
    for i in range(1, len(os.listdir(INPUT_DATA_DIR)) + 1):
        # Load image
        img = cv2.imread(f'{INPUT_DATA_DIR}/{i}.jpg')
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Preprocess
        img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
        img_thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 8)
        
        # closing edges - removing small missing parts
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        img_closed = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)

        cv2.imwrite(f"{OUTPUT_DATA_DIR}/preprocessed{i}.jpg", img_closed)

        # Find contours (dice)
        contours, _ = cv2.findContours(img_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if MIN_DICE_AREA < area < MAX_DICE_AREA:
                boxes.append((x, y, x+w, y+h))

        merged_boxes = merge_boxes(boxes)

        for box in merged_boxes:

            # Draw on original image
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        cv2.imwrite(f"{OUTPUT_DATA_DIR}/{i}.jpg", img)
