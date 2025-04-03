import cv2
import numpy as np

def process_image(image_path):
    # Read the image in BGR color space
    img = cv2.imread(image_path)

    # Preprocessing
    img_resized = cv2.resize(img, (900, 900))
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # Egg Identification
    circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, 50, param1=30, param2=40, minRadius=50, maxRadius=70)
    if circles is None:
        return img_resized, 0, 0, 0  # No circles detected
    detected_circles = np.uint16(np.around(circles))
    mask = np.ones_like(img_gray) * 255

    egg_images = []
    egg_masks = []
    egg_coords = []

    for x, y, r in detected_circles[0, :]:
        cv2.circle(mask, (x, y), r, (0, 255, 0), -1)
        egg_images.append(img_gray[y-r:y+r, x-r:x+r])
        egg_masks.append(mask[y-r:y+r, x-r:x+r])
        x, y, w, h = x-r, y-r, r*2, r*2
        egg_coords.append((x, y, w, h))

    # Individual Egg Extraction
    eggs_cropped = []
    for egg_img, egg_mask in zip(egg_images, egg_masks):
        egg_mask = cv2.bitwise_not(egg_mask)
        egg_mask = cv2.erode(egg_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=3)
        egg_crop = cv2.bitwise_and(egg_img, egg_mask)
        eggs_cropped.append(egg_crop)

    # Abnormality Detection
    def abnormal(egg, style):
        if style == 1:
            egg_abn = cv2.Canny(egg, 150, 100)
        elif style == 2:
            egg_abn = cv2.adaptiveThreshold(egg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 7)
        return egg_abn

    egg_abnormals = []
    for egg_img, egg_mask in zip(eggs_cropped, egg_masks):
        egg_abn = abnormal(egg_img, 1)
        egg_mask = cv2.bitwise_not(egg_mask)
        egg_mask = cv2.erode(egg_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=9)
        egg_abn = cv2.bitwise_and(egg_abn, egg_mask)
        egg_abnormals.append(egg_abn)

    # Watershed Algorithm for Overlapping Eggs
    binary_egg_img = np.zeros_like(img_gray)
    for x, y, r in detected_circles[0, :]:
        cv2.circle(binary_egg_img, (x, y), r, 255, -1)

    dist_transform = cv2.distanceTransform(binary_egg_img, cv2.DIST_L2, 5)
    _, markers = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    markers = np.uint8(markers)
    markers = cv2.add(markers, binary_egg_img)
    markers = cv2.watershed(img_resized, markers.astype(np.int32))
    img_resized[markers == -1] = [0, 0, 255]

    # Classification
    cracked_eggs_count = 0
    medium_impurity_eggs_count = 0
    img_output = img_resized.copy()

    for egg_img, egg_coord in zip(egg_abnormals, egg_coords):
        x, y, w, h = egg_coord
        count = np.count_nonzero(egg_img)
        canny_thres = (70, 20)
        adaptive_thres = (300, 200)
        large_impure, medium_impure = canny_thres
        if count > large_impure:
            cv2.rectangle(img_output, (x, y), (x+w, y+h), (255, 0, 0), 3)
            cracked_eggs_count += 1
        elif count > medium_impurity_eggs_count:
            cv2.rectangle(img_output, (x, y), (x+w, y+h), (255, 255, 0), 3)
            medium_impurity_eggs_count += 1

    total_detected_eggs = len(egg_coords)
    good_eggs_count = total_detected_eggs - cracked_eggs_count - medium_impurity_eggs_count

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_output, f'Cracked Eggs: {cracked_eggs_count}', (10, 30), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img_output, f'Medium Impurity Eggs: {medium_impurity_eggs_count}', (10, 70), font, 1, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img_output, f'Good Eggs: {good_eggs_count}', (10, 110), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return img_output, cracked_eggs_count, medium_impurity_eggs_count, good_eggs_count
#venv\Scripts\activate
# set FLASK_APP=app.py