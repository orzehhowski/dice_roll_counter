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

MIN_PIP_RADIUS = 8
MAX_PIP_RADIUS = 30

INPUT_DATA_DIR = "data/input"
OUTPUT_DATA_DIR = "data/output"

# detect pips inside the box, returning them as list of tuples: (abs_x, abs_y, r)
def pip_detector(img, box, min_r, max_r):
	x1, y1, x2, y2 = box

	range = img[y1:y2, x1:x2]
	
	circles = cv2.HoughCircles(
		range,
		cv2.HOUGH_GRADIENT_ALT,
		dp=1.2,
		minDist=2 * min_r,
		param1=300,
		param2=0.7,
		minRadius=min_r,
		maxRadius=max_r
	)

	detected_circles = []
	if circles is not None:
			circles = np.uint16(np.around(circles))
			for (cx, cy, r) in circles[0, :]:
					# Convert to original image coordinates
					abs_cx, abs_cy = x1 + cx, y1 + cy
					detected_circles.append((abs_cx, abs_cy, r))

	return detected_circles

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


		for box in boxes:
			pips = pip_detector(img_closed, box, MIN_PIP_RADIUS, MAX_PIP_RADIUS)
			
			# Draw on original image
			for pip in pips:
				# Draw the circle and center
				abs_cx, abs_cy, r = pip
				cv2.circle(img, (abs_cx, abs_cy), r, (0, 255, 0), 2)
				#cv2.circle(img, (abs_cx, abs_cy), 2, (0, 0, 255), 3)
			cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

		cv2.imwrite(f"{OUTPUT_DATA_DIR}/{i}.jpg", img)
