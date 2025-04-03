import cv2
import numpy as np
import matplotlib.pyplot as plt

#img_path1 = 'C:/Users/prath/OneDrive/Documents/cracked-egg-detection-master/cracked-egg-detection-master/images/test_img2.jpg'
#img_path1 = 'C:/Users/prath/OneDrive/Documents/cracked-egg-detection-master/cracked-egg-detection-master/images/test_img3.jpg'
img_path1 = 'C:/Users/prath/OneDrive/Documents/cracked-egg-detection-master/cracked-egg-detection-master/images/test_img1.jpg'

img = cv2.imread(img_path1)

# functionsSS
def plot30(egg_images, title):
    plt.figure(title)
    for index, egg_img in enumerate(egg_images):
        plt.subplot(3, 10, index+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(egg_img, cmap='gray')
        plt.title('{0}'.format(index+1))

### SECTION 1: PREPROCESSING
# resize for quicker processing 
img_resized = cv2.resize(img, (900, 900))
# rgb version
img_resized_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
# convert to grayscale for processing
img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

### SECTION 2: EGG IDENTIFICATION FROM 900x900
# hough circles
circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, 50, param1=30, param2=40, minRadius=50, maxRadius=70)
detected_circles = np.uint16(np.around(circles))
mask = np.ones_like(img_gray) * 255  # Set the mask to white

egg_images = []  # each single egg image
egg_masks = []   # mask of egg with circle. egg is background due to cv2.circle fill
egg_coords = []  # coordinates of the egg box. to be used for cv2.rectangle in output
for x, y, r in detected_circles[0, :]:

    # Create a mask for each detected circle
    cv2.circle(mask, (x, y), r, (0, 255, 0), -1)
    egg_images.append(img_gray[y-r:y+r, x-r:x+r])
    egg_masks.append(mask[y-r:y+r, x-r:x+r])
    # coordinates of the box
    x, y, w, h = x-r, y-r, r*2, r*2
    egg_coords.append((x, y, w, h))

# SECTION 3: INDIVIDUAL EGG EXTRACTION
# cropping out the egg using the mask
eggs_cropped = []
for egg_img, egg_mask in zip(egg_images, egg_masks):
    # change egg to foreground
    egg_mask = cv2.bitwise_not(egg_mask)
    # erode egg mask to cover guaranteed egg region only
    egg_mask = cv2.erode(egg_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=3)
    # apply mask to the egg
    egg_crop = cv2.bitwise_and(egg_img, egg_mask)

    eggs_cropped.append(egg_crop)

# SECTION 4: ABNORMALITY DETECTION
def abnormal(egg, style):
    if style == 1:
        # apply canny edge detector to find abnormalities
        egg_abn = cv2.Canny(egg, 150, 100)
    elif style == 2:
        # adaptive thres
        egg_abn = cv2.adaptiveThreshold(egg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 7)
    return egg_abn

egg_abnormals = []
for egg_img, egg_mask in zip(eggs_cropped, egg_masks):
    # use either canny edge or adaptive threshold method to detect abnormalities
    egg_abn = abnormal(egg_img, 1)    

    # change egg to foreground
    egg_mask = cv2.bitwise_not(egg_mask)
    # erode egg mask to cover egg region only
    egg_mask = cv2.erode(egg_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=9)
    # remove outline
    egg_abn = cv2.bitwise_and(egg_abn, egg_mask)

    egg_abnormals.append(egg_abn)

# SECTION 5: WATERSHED ALGORITHM FOR OVERLAPPING EGGS
# Convert image to binary for watershed
binary_egg_img = np.zeros_like(img_gray)
for x, y, r in detected_circles[0, :]:
    cv2.circle(binary_egg_img, (x, y), r, 255, -1)

# Distance transform and normalize
dist_transform = cv2.distanceTransform(binary_egg_img, cv2.DIST_L2, 5)
_, markers = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
markers = np.uint8(markers)

# Add unknown regions
markers = cv2.add(markers, binary_egg_img)
markers = cv2.watershed(img_resized_rgb, markers.astype(np.int32))  # Convert markers to int32
img_resized_rgb[markers == -1] = [0, 0, 255]  # Mark boundaries

# SECTION 6: CLASSIFICATION - CALCULATE ABNORMALITY LEVEL OF EGGS
# abnormality correlates to the sum of white pixels
count = []
cracked_eggs_count = 0
medium_impurity_eggs_count = 0
img_output = img_resized_rgb.copy()
for egg_img, egg_coord in zip(egg_abnormals, egg_coords):
    x, y, w, h = egg_coord
    count.append(np.count_nonzero(egg_img))
    # different count thres for canny and adaptive thres methods
    canny_thres = (70, 20)
    adaptive_thres = (300, 200)
    large_impure, medium_impure = canny_thres  # change this depending on canny_thres or adaptive_thres method
    if count[-1] > large_impure:
        cv2.rectangle(img_output, (x, y), (x+w, y+h), (255, 0, 0), 3)
        cracked_eggs_count += 1
    elif count[-1] > medium_impurity_eggs_count:
        cv2.rectangle(img_output, (x, y), (x+w, y+h), (255, 255, 0), 3)
        medium_impurity_eggs_count += 1

# Calculate good eggs
total_detected_eggs = len(egg_coords)
good_eggs_count = total_detected_eggs - cracked_eggs_count - medium_impurity_eggs_count

# Add text annotations
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img_output, f'Cracked Eggs: {cracked_eggs_count}', (10, 30), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
cv2.putText(img_output, f'Medium Impurity Eggs: {medium_impurity_eggs_count}', (10, 70), font, 1, (255, 255, 0), 2, cv2.LINE_AA)
cv2.putText(img_output, f'Good Eggs: {good_eggs_count}', (10, 110), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

# Print the number of cracked eggs, medium impurity eggs, and good eggs
print(f"Number of Cracked Eggs: {cracked_eggs_count}")
print(f"Number of Medium Impurity Eggs: {medium_impurity_eggs_count}")
print(f"Number of Good Eggs: {good_eggs_count}")

# PLOT: OUTPUT WITH ANNOTATIONS
plt.figure('output')
plt.imshow(img_output)
plt.axis('off')  # Hide axes
plt.show()
