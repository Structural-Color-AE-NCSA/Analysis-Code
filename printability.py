# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 12:34:34 2025

@author: jalom
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Mask the bottom region of the image to exclude the base or irrelevant parts
# @input: image: The image to be dealt
#         bottom_percent: cut bottom part as only center matters
# @return: A mask (The cut bottom part) which is either black(0) or white (255)
def mask_bottom_region(image, bottom_percent=0.2):
    height, width = image.shape[:2]
    mask = np.ones((height, width), dtype=np.uint8) * 255
    bottom_height = int(height * bottom_percent)
    mask[height - bottom_height:, :] = 0
    return mask


# Improved contour filtering with an area filter
# @input: contours: The list of contours that detected
#         min_area: the default smallest contour
# @return: the contour with largest area
def getBiggestContourWithAreaFilter(contours, min_area=500):
    filtered_contours = [c for c in contours if cv.contourArea(c) > min_area]
    if len(filtered_contours) == 0:
        return None
    return max(filtered_contours, key=cv.contourArea)


# @input: binaryImage: The mask from previous func
#         mode: way to get contour, set to default
#         method: way to get points in contour
# @return: contour list used for getbiggest func
def getContours(binaryImage, mode='TREE', method=cv.CHAIN_APPROX_NONE):
    try:
        if mode == 'LIST':
            contours, _ = cv.findContours(binaryImage, cv.RETR_CCOMP, method)
        else:
            contours, _ = cv.findContours(binaryImage, cv.RETR_TREE, method)
        return contours
    except Exception as e:
        print(f"Error in getContours: {e}")
        return None


# @input: the contour
#         y value for the horizontal line
# @return: list of intersecting x value
def get_horizontal_intersections(contour, y_line):
    # initialize the list to store all intersecting x value
    intersections = []
    # usually in contour we have numpy with form (n,1,2) to store all points
    # first : means that we pick all points within the contour
    # 0 means we ignore second part
    # third : means pick all (x,y)
    pts = contour[:, 0, :]
    num_points = pts.shape[0]
    # iterate all points
    for i in range(num_points):
        p1 = pts[i]
        p2 = pts[(i + 1) % num_points]  # ensure the last point meets with the first one
        y1, y2 = p1[1], p2[1]
        # decide if the edge of two points will intersect with horizontal line
        # proceed if the product < 0, indicating a valid intersection
        if (y1 - y_line) * (y2 - y_line) < 0:
            # calculate the x value of the intersection point
            x = p1[0] + (y_line - y1) * (p2[0] - p1[0]) / (y2 - y1)
            intersections.append(x)
        # deal with case if both y1, y2 are on the line
        elif y1 == y_line or y2 == y_line:
            intersections.extend([p1[0], p2[0]])
    return intersections


# @input: image_file: the image we need to analyze
#         bottom_percent: the percent that we want to mask at the beginning
#         threshold_val: the threshold that we use to disguish between background and targe image. 50 TBD!!!
def process_image(image_file, bottom_percent=0.0, threshold_val=50):
    # step 1: read image and mask bottom
    img_bgr = cv.imread(image_file)
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {image_file}")
    bottom_mask = mask_bottom_region(img_bgr, bottom_percent=bottom_percent)
    masked_bgr = cv.bitwise_and(img_bgr, img_bgr, mask=bottom_mask)

    # step 2:  deal with the image and turn it into clear, simplified image
    img_rgb = cv.cvtColor(masked_bgr, cv.COLOR_BGR2RGB)
    gray_img = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
    _, bin_img = cv.threshold(gray_img, threshold_val, 255, cv.THRESH_BINARY)

    # step 3: detect and get the exact contour
    contours = getContours(bin_img)
    contour = getBiggestContourWithAreaFilter(contours, min_area=5000)
    if contour is None:
        raise ValueError("No valid contour found. Adjust threshold or check the image.")

    # step 4: get the coordinates of all points, especailly the upper and lower bound of y
    pts = contour[:, 0, :]
    topmost_y = np.min(pts[:, 1])
    bottommost_y = np.max(pts[:, 1])
    # calculate total height
    total_height = bottommost_y - topmost_y
    if total_height <= 0:
        raise ValueError("Invalid contour height (possibly a single line?).")

    # From here we compute the height of each segment and get their according y values
    # seg_height = total_height / (n + 1)
    horizontal_lines = list(range(topmost_y, bottommost_y + 1))

    # 5. For each horizontal line, get its all x values and draw the lines
    line_lengths = []
    img_with_lines = img_rgb.copy()  # copy of original image and we will modify on this version

    # iterate the horizontal height values and get the x values for each height y
    for y in horizontal_lines:
        xs = get_horizontal_intersections(contour, y)
        # if the number of x values is smaller than 2, meaning it can not form a valid segment. So we label the length to be 0
        if len(xs) < 2:
            line_lengths.append(0)
            continue
        # if it is a valid line, then we store it.
        left_x, right_x = min(xs), max(xs)
        line_length = right_x - left_x
        line_lengths.append(line_length)

        # draw the line in red within range of the contour
        cv.line(
            img_with_lines,
            (int(left_x), int(y)),
            (int(right_x), int(y)),
            (255, 0, 0),
            2
        )

    # draw the exact contour(green)
    cv.drawContours(img_with_lines, [contour], -1, (0, 255, 0), 3)

    # step 6: show the final score

    # plt.figure(figsize=(10, 8))
    # plt.imshow(img_with_lines)
    # plt.title("Contour with Horizontal Segments Inside")
    # plt.axis('off')
    # plt.show()

    # 7. draw the histogram
    # plt.figure(figsize=(12, 6))
    # indices = np.arange(len(horizontal_lines))
    # plt.bar(indices, line_lengths, color='skyblue', label='Segment Length')
    # plt.xlabel("Horizontal Line Index (from bottom to top)")
    # plt.ylabel("Length (pixels)")
    # plt.title("Horizontal Intersection Lengths Within Contour")

    # Step 8: Calculate mean and standard deviation
    lengths_array = np.array(line_lengths)
    mean_length = np.mean(lengths_array)
    std_length = np.std(lengths_array)  # printability score t be saved
    print(f"std_length: {std_length}")
    return std_length

    # draw the line of average and standard deviation
    # plt.axhline(mean_length, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_length:.2f}')
    # # use fill_between to depict the sd
    # plt.fill_between(indices, mean_length - std_length, mean_length + std_length,
    #                  color='green', alpha=0.3, label=f'Std Dev = {std_length:.2f}')
    # plt.legend()
    # plt.show()

    # print out info of each segment
    ## for test: print(f"Line length{total_height}")
    # for idx, length in enumerate(line_lengths, start=1):
    #     print(f"Line {idx}: y = {horizontal_lines[idx-1]:.2f}, Length = {length:.2f} pixels")
    # # Step 9: save the result as csv file
    #     df = pd.DataFrame({
    #         'y': horizontal_lines,
    #         'length': line_lengths
    #     })
    #     csv_filename = "segmentation_lengths.csv"
    #     df.to_csv(csv_filename, index=False)
    #     print(f"Results saved to {csv_filename}")
# change the file name and the number of inserction parts here
# if __name__ == "__main__":
#     image_file = "Green_Sanghyun.jpg"
#     process_image(image_file)