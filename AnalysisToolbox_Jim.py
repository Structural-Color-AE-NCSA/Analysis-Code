import time
import AmaresConfig as config
from Messaging import *
import traceback

import numpy as np
import matplotlib.pyplot as plt
import cv2

from PIL import Image

from skimage import data, img_as_float, img_as_uint, segmentation, color, exposure
from skimage.color import rgb2gray, rgb2hsv
from skimage.transform import rescale, resize
from skimage.morphology import reconstruction
import copy


def get_vertical_edge_variation(contours, desired_width):
    try:
        # finds the vertical edges of a leading segment
        # vertical edges begin at the bottom of FOV and end at max(y) - 0.5*desired_width
        # * also need to handle multiple contours

        # combine contours to create a single array of points (if needed)
        if len(contours) > 1:
            all_points = np.concatenate(contours)
        else:
            all_points = contours[0]

        # reformat array
        all_points = [[point[0][0], point[0][1]] for point in all_points]

        # find avg_x to use as default x value for spaces between vertically separated contours
        avg_x = np.average([point[0] for point in all_points])

        # get max y coordinate of contours (remember y = 0 is at the TOP of FOV)
        max_y = min([point[1] for point in all_points]) + 0.5 * desired_width

        # get left and right edges
        left_edge = np.stack([point for point in all_points if (point[0] <= avg_x and point[1] > max_y)])
        right_edge = np.stack([point for point in all_points if (point[0] > avg_x and point[1] > max_y)])

        # get average of left and right edges
        avg_left_edge = np.average([point[0] for point in left_edge])
        avg_right_edge = np.average([point[0] for point in right_edge])

        # get average variation of left and right edges from vertical straight lines
        avg_var_left_edge = np.average([abs(point[0] - avg_left_edge) for point in left_edge])
        avg_var_right_edge = np.average([abs(point[0] - avg_right_edge) for point in right_edge])
        avg_var = np.average([avg_var_left_edge, avg_var_right_edge])
        return round(avg_var, 4), left_edge, right_edge
    except:
        addToLog(traceback.format_exc())


def showImages(img1, img2, xlabel1=0):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4),
                         sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(img1, cmap=plt.cm.gray, vmin=0, vmax=1)
    
    ax[1].imshow(img2, cmap=plt.cm.gray, vmin=0, vmax=1)
    #ax[1].set_xlabel(label.format(xlabel1))
   
    plt.show()


def gaussianBlur(img,kernel_size=3):
    try:
        blurred = cv2.GaussianBlur(img,(kernel_size,kernel_size),cv2.BORDER_DEFAULT)
        return(blurred)
    except:
        addToLog(traceback.format_exc())


def showImage(img_data, title="Image"):
    # used in debug mode
    plt.imshow(img_data)
    plt.title(title)
    plt.show()


def denoise(img, strength, templateWindowSize = 7, searchWindowSize = 21):
    try:
        quiet = cv2.fastNlMeansDenoisingColored(img, None, strength, strength,
                                                templateWindowSize, searchWindowSize)
        return(quiet)
    except:
        addToLog(traceback.format_exc())


def do_Rescale(image, current_scale, factor=1):
    try:
        image_rescaled = cv2.resize(image, dsize=(int(image.shape[1]*factor),int(image.shape[0]*factor)),
                                     interpolation=cv2.INTER_CUBIC)
        new_scale = int(current_scale * factor)
        return(image_rescaled, new_scale)
    except:
        addToLog(traceback.format_exc())


def scaledRound(num):
    # scales a number to it's highest place value
    # e.g. scaledRound(3234) = 3000
    try:
        if num >= 1:
            num = int(num)
            return(round(num,-(len(str(num))-1)))
        else:
            return(1)
    except:
            addToLog(traceback.format_exc())


def doScaleBar(img, scale, show=False):
    # scale = pix/µm
    # draws a scalebar onto 'img'
    try:
        height, width = img.shape[0:2]
        font_scale = height / 1500
        # scalebar width
        sb_width = int(0.2 * scaledRound(width/scale))
        imginfo = "width = " + str(round(width/scale, 1)) +\
                  " um, height = " + str(round(height/scale, 1)) + " um"

        cv2.putText(img, imginfo, (10, 25), cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255),
                    lineType=cv2.LINE_AA)
        cv2.rectangle(img, (width-int(1.2*sb_width*scale), int(0.025*height)),
                      (width - int(0.2*sb_width*scale), int(0.025*height)+10), (255, 255, 255), -1)
        cv2.putText(img, str(int(sb_width)) + " um",
                    (width-int(1.0*sb_width*scale), int(0.025*height)+30),
                    cv2.FONT_HERSHEY_DUPLEX, font_scale*1.4, (255, 200, 100),
                    lineType=cv2.LINE_AA)
        if show:
            showImage(img, "Scalebar Added")
    except:
        addToLog(traceback.format_exc())


def getBackgroundSpecs(img_data, crop=0.1, buffer=30, use_left=True, use_best_side=False, dark_bg=False, debug=False):
    #       - gets average background pixel, brightest background pixel, and dimmest
    #         background pixel for left and right regions for an image of a centered
    #        vertical line
    #       - 'crop' defines the percent of image width used to define left and right
    #         background regions
    #       - buffer adds or subtracts an integer to lower and upper values
    try:
        img = cv2.GaussianBlur(img_data, (15, 15), 0)
        leftBound = int(img.shape[1]*crop)
        rightBound = int(img.shape[1]*(1-crop))
        height = img.shape[0]
        leftRegion = img[0:height, 0:leftBound]
        rightRegion = img[0:height, rightBound:img.shape[1]]

        if use_best_side:
            if not dark_bg:
                if np.average(leftRegion) > np.average(rightRegion):
                    bothRegions = leftRegion
                else:
                    bothRegions = rightRegion
            elif dark_bg:
                if np.average(leftRegion) < np.average(rightRegion):
                    bothRegions = leftRegion
                else:
                    bothRegions = rightRegion
        else:
            if use_left:
                bothRegions = np.concatenate((leftRegion, rightRegion), axis=1)
            else:
                bothRegions = rightRegion
                
        # get average background pixel:
        r,g,b = cv2.split(bothRegions)
        rAvg = np.average(r)
        gAvg = np.average(g)
        bAvg = np.average(b)
        avgBgPix = np.array([rAvg, gAvg, bAvg])
        avgBgPix = np.clip(avgBgPix, 0, 255).astype('uint8')
        # get largest background values:
        rMax = (np.amax(r)+buffer)
        gMax = (np.amax(g)+buffer)
        bMax = (np.amax(b)+buffer)
        maxBgPix = np.array([rMax, gMax, bMax])
        maxBgPix = np.clip(maxBgPix, 0, 255).astype('uint8')
        # get dimmest background values:
        rMin = (np.amin(r)-buffer)
        gMin = (np.amin(g)-buffer)
        bMin = (np.amin(b)-buffer)
        minBgPix = np.array([rMin, gMin, bMin])
        minBgPix = np.clip(minBgPix, 0, 255).astype('uint8')
        return avgBgPix, maxBgPix, minBgPix
    except:
        addToLog(traceback.format_exc(), debug=debug)


def getBackgroundSpecsGrey(img_data, crop=0.1, buffer=25, use_left=False, use_best_side=False, dark_bg=False, debug=False):
    # - Gets average background greyscale value
    # - 'crop' defines the percent of image width used to define left and right
    # background regions
    # buffer adds or subtracts an integer to lower and upper values
    try:
        img = cv2.GaussianBlur(img_data, (15, 15), 0)
        leftBound = int(img.shape[1]*crop)
        rightBound = int(img.shape[1]*(1-crop))
        height = img.shape[0]
        leftRegion = img[0:height, 0:leftBound]
        rightRegion = img[0:height, rightBound:img.shape[1]]
        if use_best_side:
            if not dark_bg:
                if np.average(leftRegion) > np.average(rightRegion):
                    bothRegions = leftRegion
                else:
                    bothRegions = rightRegion
            elif dark_bg:
                if np.average(leftRegion) < np.average(rightRegion):
                    bothRegions = leftRegion
                else:
                    bothRegions = rightRegion
        else:
            if use_left:
                bothRegions = np.concatenate((leftRegion, rightRegion), axis=1)
            else:
                bothRegions = rightRegion
        # get average background pixel greyscale value:
        avgBgPix = np.average(bothRegions)
        return avgBgPix-buffer
    except:
        addToLog(traceback.format_exc(), debug=debug)


def measureContrast(img_data):
    try:
        img_grey = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
        contrast = img_grey.std()/img_grey.mean()
        return contrast
    except Exception as ex:
        addToLog(traceback.format_exc())


def npAvg(a, b):
    try:
        a = a.astype('float32')
        b = b.astype('float32')
        return ((a+b)/2).astype('uint8')
    except:
        addToLog(traceback.format_exc())
        

def flattenBG(img_data, avg_bg_color, new_bg_color=[0,0,0], black_bg=True, tolerance=0.1, show=False, debug=False):
    # avg_bg_color is in [R,G,B] format where R, G, and B are from 0..255
    # black_bg: black background (bg) or not (i.e. or white)?
    # tolerance: if abs(1 - pixelval/targetval) < tolerance then set to new_bg_color
    try:
        img = img_data.copy()
        for y in range(img.shape[1]):
            for x in range(img.shape[0]):
                if ((abs(1 - (img[x, y][0]/avg_bg_color[0])) < tolerance) and
                        (abs(1 - (img[x, y][1]/avg_bg_color[1])) < tolerance) and
                        (abs(1 - (img[x, y][2]/avg_bg_color[2])) < tolerance)):
                    img[x, y] = np.array(new_bg_color).astype('uint8')
        if show:
            showImage(img, "Flattened Image")
        return img
    except:
        addToLog(traceback.format_exc(), debug=debug)
            

def removeBoundaryContours(contours, input_img, left=True, right=True, top=False):
    # removes contours that touch edges specified
    try:
        newContours = []
        # loops through list of np.arrays containing contour coordinates:
        for contour in contours:                    
            isgood = True
            # loops through all coordinates in each contour:
            for index in range(contour.shape[0]):
                # checks each "x" coordinate or column index for edge touching
                if left:
                    # check for left edge:
                    if contour[index,0][0] <= 1:
                        isgood = False
                        break      
                if right:
                    # check for right edge:
                    if contour[index,0][0] >= (input_img.shape[1]-1):
                        isgood = False
                        break
                if top:
                    # if there is more than one contour:
                    if len(contours) > 1:
                        # check for top edge:
                        if contour[index,0][1] <= 2:
                            isgood = False
                            break
            if isgood:
                # if passed all edge checks:
                newContours.append(contour)
        return newContours
    except:
        addToLog(traceback.format_exc())


def removeBoundaryBlobs(binary_image, contours, left=True, right=True, top=True, bottom=True):
    # uses contours to subtract boundary blobs from binary_image and returns resulting image
    # loops through list of np.arrays containing contour coordinates:
    try:
        for contour in contours:
            isgood = True
            # loops through all coordinates in each contour:
            for index in range(contour.shape[0]):
                # checks each "x" coordinate or column index for edge touching
                if left:
                    # check for left edge:
                    if contour[index, 0][0] <= 2:
                        isgood = False
                        break
                if right:
                    # check for right edge:
                    if contour[index, 0][0] >= (binary_image.shape[1]-2):
                        isgood = False
                        break
                if top:
                    # if there is more than one contour:
                    if len(contours) > 1:
                        # check for top edge:
                        if contour[index, 0][1] <= 2:
                            isgood = False
                            break
                if bottom:
                    # check for bottom edge:
                    if contour[index, 0][1] >= (binary_image.shape[0]-2):
                        isgood = False
                        break
            if not isgood:
                # if failed one of the edge checks then subtract blob using contour
                cv2.drawContours(binary_image, contour, -1, (0, 0, 0), -1)
        return binary_image
    except:
        addToLog(traceback.format_exc())


def removeSmallContours(contours, area=100):
    # removes contours with small areas (as defined by 'area' in pix^2)
    try:
        newContours = []
        for contour in contours:
            if cv2.contourArea(contour) > area:
                newContours.append(contour)
        return newContours
    except:
        addToLog(traceback.format_exc())


def getBiggestContour(contours, num_contours=1):
    # returns the largest contour(s)
    try:
        if len(contours) == 1:
            return contours
        elif len(contours) > 1 and num_contours == 1:
            biggest = contours[0]
            for contour in contours:
                if cv2.contourArea(contour) > cv2.contourArea(biggest):
                    biggest = contour
            return (biggest,)
        elif len(contours) > 1 and len(contours) > num_contours > 1:
            biggest = list(contours[:num_contours])
            for contour in contours[num_contours:]:
                areas = [cv2.contourArea(cont) for cont in biggest]
                if cv2.contourArea(contour) > min(areas):
                    _ = biggest.pop(areas.index(min(areas)))
                    biggest.append(contour)
            return tuple(biggest)
        elif num_contours > len(contours) > 1:
            return contours
        else:
            return contours
    except:
        addToLog(traceback.format_exc())


def removeSmallBlobs(binary_image, contours, area=100):
    # removes small blobs using identified contours and returns resulting IMAGE (not contours)
    try:
        for contour in contours:
            if cv2.contourArea(contour) <= area:
                # drawContours(image, contours, contour_idx (-1 = all), color, thickness (-1 = filled)
                cv2.drawContours(binary_image, contour, -1, (0, 0, 0), -1)
        return binary_image
    except:
        addToLog(traceback.format_exc())


def dilate(img_data, kernel_size=5, iterations=1):
    # dilates binary shapes
    # img_data must be binary
    try:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated = cv2.dilate(img_data, kernel, iterations=iterations)
        return dilated
    except:
        addToLog(traceback.format_exc())


def erode(img_data,kernel_size=5,iterations=1):
    # erodes binary shapes
    # img_data must be binary
    try:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        eroded = cv2.erode(img_data, kernel, iterations=iterations)
        return eroded
    except:
        addToLog(traceback.format_exc())


def equalizeColorHist(color_img, method='normal', show=False):
    # converts to YUV and equalizes the luminance (Y) component
    # alternative methods: 'clahe', 'none'
    try:
        if method != 'none':
            yuv_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2YUV)
        if method == 'normal':
            yuv_img[:, :, 0] = cv2.equalizeHist(yuv_img[:, :, 0])
        elif method == 'clahe':
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            yuv_img[:, :, 0] = clahe.apply(yuv_img[:, :, 0])
        else:
            pass
        if method == 'none':
            output_img = color_img
        else:
            output_img = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR)
        if show:
            showImage(output_img, "Equalized Histogram")
        return output_img
    except:
        addToLog(traceback.format_exc())


def getBinary(input_img, thresh=50, upper=255, show=False, erode_size=2, erode_iter=1, dilate_size=0, dilate_iter=0, debug=False):
    # Analyzes and returns thresholded binary image
    try:
        binary_img = cv2.threshold(input_img, thresh, upper, cv2.THRESH_BINARY)[1]
        try:
            eroded_img = erode(binary_img, kernel_size=erode_size, iterations=erode_iter)
            dilated_img = dilate(eroded_img, kernel_size=dilate_size, iterations=dilate_iter)
            if show:
                showImage(dilated_img, "Binary Image")
            addToLog("Binary image created. Type is " + str(type(binary_img)))
            return dilated_img
        except:
            addToLog("Problem eroding binary image: " + str(type(binary_img)))
            addToLog(traceback.format_exc())
            return binary_img
    except:
        addToLog(traceback.format_exc())


def getPixelArea(binaryImage, scale, color=255, debug=False):  # scale in pix/mm
    try:
        # counts pixels having designated color
        color_pixels = 0
        for y in range(0, binaryImage.shape[0] - 0):
            for x in range(0, binaryImage.shape[1] - 0):
                if binaryImage[y, x] > 0:
                    color_pixels += 1
        area = color_pixels / (scale**2)
        return area
    except:
        if not debug:
            addToLog(traceback.format_exc())
        else:
            print(traceback.format_exc())
            print("binaryImage.shape[1]" + str(binaryImage.shape[1]))
        return 0


def getContours(binaryImage, mode='TREE', method=cv2.CHAIN_APPROX_SIMPLE):
    # returns filtered contours of binary image
    # RETR_LIST for all, RETR_TREE for external only(?), CCOMP for
    try:
        # get contours (works for different versions of openCV)
        if mode == 'LIST':  # gets all of the contours regardless of hierarchy(?)
            try:
                contours, _ = cv2.findContours(binaryImage, cv2.RETR_CCOMP, method)
            except:
                _, contours, _ = cv2.findContours(binaryImage, cv2.RETR_CCOMP, method)

        else:
            try:
                contours, _ = cv2.findContours(binaryImage, cv2.RETR_TREE, method)
            except:
                _, contours, _ = cv2.findContours(binaryImage, cv2.RETR_TREE, method)
        return contours
    except:
        return None
        addToLog(traceback.format_exc())


def getROIRectangle(binaryImage, contours, desired_width, mode='fixed', min_y=-1, vertical_shift=1):
    # Returns coordinates and size of target ROI in scaled pixel units
    #          - remember (0,0) is top-left of image
    #          - ultimately, the target shape is drawn based on the coordinates and
    #            dimensions of this rectangle
    #
    #        currently, there are 3 modes:
    #            1. 'fixed' mode - the target shape is at a fixed location in the FOV
    #                    - fixed y-location will be height/2 - vertical_shift (remember
    #                    (0,0) is top-left)
    #            2. 'flex_y' mode - the target shape is fixed in the center of the FOV
    #            but the top of the target shape can move vertically to fit the
    #            printed feature
    #                    - 'min_y' is user-specified and sets the MINIMUM rectangle
    #                    height. Recall (0,0) is top-left, so the rectangle's base
    #                    is found by min_y + height (it's all upside-down wtf).
    #            3. 'flex_xy' mode - the target shape can move horizontally and
    #            vertically to fit the printed feater
    try:
        # first we need the image dimensions:
        height, width = binaryImage.shape[0:2] 
        vertical_shift = int(vertical_shift)

        if mode == 'square':
            x = round((width / 2) - (desired_width / 2))
            y = round((height / 2) - (desired_width / 2))
            w = desired_width
            h = desired_width

        if mode == 'fixed':
            x = round((width/2) - (desired_width/2))
            y = round(height/2) - vertical_shift
            w = desired_width
            h = round(height/2) + vertical_shift

        if mode == 'flex_x':
            #  initialize extents values before iterating over contour values
            y = round(height/2) - vertical_shift
            w = desired_width
            h = round(height/2) + vertical_shift
            min_x = width           # minimum x-value initializes at maximum possible value
            max_x = 0
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                min_x, max_x = min(min_x, x), max(max_x, x+w)
            # centers the rectangle on the feature:
            x = int((min_x + max_x) / 2 - (desired_width / 2))
            y = round(height / 2) - vertical_shift
            w = desired_width
            h = round(height / 2) + vertical_shift

        if mode == 'flex_y':
            # initialize extents values before iterating over contour values
            if min_y == -1:         # if user didn't specify min_y
                min_y = height
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                min_y = min(min_y, y)
            x = round((width/2) - (desired_width/2))
            y = min_y
            w = desired_width
            h = height - min_y

        if mode == 'flex_xy':
            # initialize extents values before iterating over contour values
            if min_y == -1:         # if user didn't specify min_y
                min_y = height
            min_x = width           # minimum x-value initializes at maximum possible value
            max_x = 0
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                min_y = min(min_y, y)
                min_x, max_x = min(min_x, x), max(max_x, x+w)
            # centers the rectangle on the feature:
            x = int((min_x + max_x)/2 - (desired_width/2)) 
            y = min_y
            w = desired_width
            h = height - min_y         
        return x, y, w, h

    except:
        # use some crappy default values:
        x = int(width / 2) - int(desired_width/2)
        y = int(height / 2)
        w = desired_width
        h = int(height / 2)
        addToLog(traceback.format_exc())
        return x, y, w, h


def drawSquare(img, top_left, bottom_right, color=(50, 100, 255), thickness=3):
    # top-left and bottom-right are of the format (x,y)
    # 'thickness' is outline thickness in pixels
    try:
        thickness = int(thickness)
        if thickness > 0:
            # draw outline of rectangle
            cv2.rectangle(img, top_left, bottom_right, color, thickness)
        else:
            # draw a filled rectangle
            thickness = -1
            cv2.rectangle(img, top_left, bottom_right, color, thickness)
        return img
    except:
        addToLog(traceback.format_exc())


def getMultiBoundingRectangle(contours):
    # input is list of contours
    # returns rectangle that bounds multiple contours
    try:
        lefts = []
        rights = []
        tops = []
        bottoms = []
        # remember that y-axis is flipped (0 = top)
        for contour in contours:
            x_c, y_c, w_c, h_c = cv2.boundingRect(contour)
            lefts.append(x_c)
            rights.append(x_c + w_c)
            tops.append(y_c)
            bottoms.append(y_c + h_c)
        x = min(lefts)
        w = max(rights) - x
        y = min(tops)
        h = max(bottoms) - y
        return x, w, y, h
    except:
        addToLog(traceback.format_exc())


def getCentroid(img):
    try:
        grayscale = toGrayscale(img)
        binary = getBinary(grayscale,
                           thresh=config.imageAnalysis["BINARY_THRESHOLD"],
                           erode_size=config.imageAnalysis['BINARY_ERODE_SIZE'],
                           erode_iter=config.imageAnalysis['BINARY_ERODE_ITER'],
                           dilate_size=config.imageAnalysis['BINARY_DILATE_SIZE'],
                           dilate_iter=config.imageAnalysis['BINARY_DILATE_ITER'],
                           upper=255,
                           show=False)

        contours = getContours(binary)
        contours = removeSmallContours(contours, config.imageAnalysis['SMALLEST_CONTOUR_AREA'])
        if len(contours) > 1:
            biggest = getBiggestContour(contours)
            x, y, w, h = getMultiBoundingRectangle(biggest)
        elif len(contours) == 1:
            x, y, w, h = cv2.boundingRect(contours[0])
        else:
            x, y, w, h = 0, 0, 0, 0
        return x, y, w, h
    except:
        pass


def drawVerticalSlot(img, top_left, bottom_right, color=(50, 100, 255), thickness=3):
    # top-left and bottom-right are of the format (x,y) and don't include rounded ends
    # 'thickness' is outline thickness in pixels
    try:
        thickness = int(thickness)
        radius = int((bottom_right[0]-top_left[0])/2)
        bottom_left = (top_left[0], bottom_right[1])
        top_right = (bottom_right[0], top_left[1])
        top_arc_center = (int((top_left[0]+top_right[0])/2), top_left[1])
        bot_arc_center = (int((top_left[0]+top_right[0])/2), bottom_right[1])
        if thickness > 0:
            # draw outline of filleted rectangle
            cv2.line(img, top_left, bottom_left, color, thickness)
            cv2.ellipse(img, bot_arc_center, (radius, radius), 0, 0, 180,
                        color, thickness)
            cv2.line(img, bottom_right, top_right, color, thickness)
            cv2.ellipse(img, top_arc_center, (radius, radius), 0, 180, 360,
                        color, thickness)
        else:
            # draw a filled rectangle and circle
            thickness = -1
            cv2.rectangle(img, top_left, bottom_right, color, thickness)
            cv2.circle(img, top_arc_center, radius, color, thickness)
            cv2.circle(img, bot_arc_center, radius, color, thickness)
        return img
    except:
        addToLog(traceback.format_exc())


def drawTargetOutline(img, target_fp, color=(255, 0, 0), thickness=3):
    # img is PIL image
    # target_img is filapath to target image
    # 'thickness' is outline thickness in pixels
    try:
        thickness = int(thickness)
        # convert PIL to OpenCV:
        open_cv_img = np.array(img)
        # load target image:
        target_img = cv2.imread(target_fp)
        # get outline of target image:
        target_img = getBinary(toGrayscale(target_img))
        contours = getContours(target_img)
        for contour in contours:
            cv2.drawContours(open_cv_img, [contour], 0, color, thickness)
        return Image.fromarray(open_cv_img)
    except:
        addToLog(traceback.format_exc())


def drawTopFilletRectangle(img, top_left, bottom_right, color=(50, 100, 255), thickness=3):
    # top-left and bottom-right are of the format (x,y)
    # 'thickness' is outline thickness in pixels
    try:
        thickness = int(thickness)
        radius = int((bottom_right[0]-top_left[0])/2)
        top_left = (top_left[0], top_left[1]+radius)
        bottom_left = (top_left[0], bottom_right[1])
        top_right = (bottom_right[0], top_left[1])
        arc_center = (int((top_left[0]+top_right[0])/2), top_left[1])
        if thickness > 0:
            # draw outline of filleted rectangle
            cv2.line(img, top_left, bottom_left, color, thickness)
            cv2.line(img, bottom_left, bottom_right, color, thickness)
            cv2.line(img, bottom_right, top_right, color, thickness)
            cv2.ellipse(img, arc_center, (radius, radius), 0, 180, 360,
                        color, thickness)
        else:
            # draw a filled rectangle and circle
            thickness = -1
            cv2.rectangle(img, top_left, bottom_right, color, thickness)
            cv2.circle(img, arc_center, radius, color, thickness)
        return img
    except:
        addToLog(traceback.format_exc())


def topFilletRectArea(top_left, bottom_right):
    # calculates and returns the area of a top-filleted rectangle that fits
    # within the specified rectangle; radius of fillet is half the width of
    # the rectangle
    try:
        radius = int((bottom_right[0]-top_left[0])/2)
        width = bottom_right[0] - top_left[0]
        height = bottom_right[1] - top_left[1] - radius
        semicircleArea = (np.pi * radius * radius)/2
        rectangleArea = width * height
        return semicircleArea + rectangleArea
    except:
        addToLog(traceback.format_exc())


def verticalSlotArea(top_left, bottom_right):
    # calculates and returns the area of a vertical slot
    try:
        radius = int((bottom_right[0]-top_left[0])/2)
        width = bottom_right[0] - top_left[0]
        height = bottom_right[1] - top_left[1]
        semicircleArea = (np.pi * radius * radius)/2
        rectangleArea = width * height
        return 2*semicircleArea + rectangleArea
    except:
        addToLog(traceback.format_exc())


def toGrayscale(input_img, show=False, debug=False):
    try:
        output_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)
        if show:
            showImage(output_img, "Grayscale")
        return output_img
    except:
        addToLog(traceback.format_exc(), debug=debug)


def drawFilledContours(input_img, contours, show=False):
    try:
        if len(contours) > 0:
            for contour in contours:
                cv2.drawContours(input_img, [contour], 0, 255, -1)
            output_img = cv2.bitwise_not(input_img)
            #cv2.imwrite(config.paths["campaigndatapath"] + "outImg_" + str(config.data['Expt#']) + ".jpg", output_img)
            if show:
                showImage(~output_img, "Blobs Drawn")
            return ~output_img
        else:
            return input_img
    except:
        addToLog(traceback.format_exc())


def dataOverlay(input_img, data_text="(data text)", descriptors=['Score 1', 'Score 2', 'Score 3'],
                expt_num=0, result="-1", result2="-1", result3="-1", text_color=(255, 255, 255)):
    # annotates image with relevant data
    try:
        font = cv2.FONT_HERSHEY_DUPLEX
        height = input_img.shape[0]
        width = input_img.shape[1]
        font_scale = height/1000*config.imageAnalysis['DATA_OVERLAY_FONT_SCALE']
        y_loc = 30
        text = data_text + " Expt# " + str(expt_num)
        cv2.putText(input_img, text, (10, height-y_loc), font, font_scale, text_color, 1, cv2.LINE_AA) # 25,255,25
        if str(result)[0:2].isalpha():
            text = str(result)
        else:
            text = (descriptors[0] + " = " + str(round(float(result), 4)))
        if result2 != "-1":
            if str(result2).isnumeric():
                text += "; " + descriptors[1] + " = " + str(round(float(result2), 4))
            else:
                text += "; " + str(result2)
        if result3 != "-1":
            if str(result3)[0:2].isalpha():
                text += "; " + str(result3)
            else:
                text += "; " + descriptors[2] + " = " + str(round(float(result3), 4))
        cv2.putText(input_img, text, (10, height - y_loc + 15), font, font_scale, text_color, 1, cv2.LINE_AA)
#        extra_text = ("Thresh = " + str(config.imageAnalysis['HONEY_BINARY_THRESHOLD']))
#        cv2.putText(input_img,extra_text, (int(width*0.8), height-10), font, font_scale, (100,255,0), 1, cv2.LINE_AA)
#        extra_text = ("Light = " + str(config.tool_vars['alignLightVal']))
#        cv2.putText(input_img,extra_text, (int(width*0.8), height-30), font, font_scale, (100,255,0), 1, cv2.LINE_AA)
    except:
        addToLog(traceback.format_exc())


def bilateralFilter(input_img, d=-1, sigmaColor=250, sigmaSpace=75, show=False, debug=False):
    #    reduces noise but maintains 'fairly' sharp edges
    #    - 'd' is the diameter of each pixel neighborhood that is
    #      used during filtering. If it is non-positive, it is
    #      computed from sigmaSpace.
    #    - 'sigmaColor': Filter sigma in the color space. A larger
    #      value of the parameter means that farther colors within
    #      the pixel neighborhood (see sigmaSpace) will be mixed
    #      together, resulting in larger areas of semi-equal color.
    #    - 'sigmaSpace': A larger value of the parameter means that
    #      farther pixels will influence each other as long as their
    #      colors are close enough (see sigmaColor ).
    try:
        output_img = cv2.bilateralFilter(input_img, d, sigmaColor, sigmaSpace)
        if show:
            showImage(output_img, "Bilateral Filter")
        return output_img
    except:
        addToLog(traceback.format_exc(), debug=debug)
