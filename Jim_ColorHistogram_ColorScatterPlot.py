# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 22:38:00 2023

Accepts image and then spits out 
HSV histogram and color scatterplot

I THINK THE CIRCLE FUNCTION IS WORKING!!!!!!!!

@author: J.A. LOMAS
"""

import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
import traceback  # for exception reporting
import pandas as pd
from math import pi

def getContours(binaryImage, mode='TREE', method=cv.CHAIN_APPROX_SIMPLE):
    # returns filtered contours of binary image
    # RETR_LIST for all, RETR_TREE for external only(?), CCOMP for
    try:
        # get contours (works for different versions of openCV)
        if mode == 'LIST':  # gets all of the contours regardless of hierarchy(?)
            try:
                contours, _ = cv.findContours(binaryImage, cv.RETR_CCOMP, method)
            except:
                _, contours, _ = cv.findContours(binaryImage, cv.RETR_CCOMP, method)

        else:
            try:
                contours, _ = cv.findContours(binaryImage, cv.RETR_TREE, method)
            except:
                _, contours, _ = cv.findContours(binaryImage, cv.RETR_TREE, method)
        return contours
    except:
        return None
        print(traceback.format_exc())


def getBiggestContour(contours, num_contours=1):
    # returns the largest contour(s)
    try:
        if len(contours) == 1:
            return contours
        elif len(contours) > 1 and num_contours == 1:
            biggest = contours[0]
            for contour in contours:
                if cv.contourArea(contour) > cv.contourArea(biggest):
                    biggest = contour
            return (biggest,)
        elif len(contours) > 1 and len(contours) > num_contours > 1:
            biggest = list(contours[:num_contours])
            for contour in contours[num_contours:]:
                areas = [cv.contourArea(cont) for cont in biggest]
                if cv.contourArea(contour) > min(areas):
                    _ = biggest.pop(areas.index(min(areas)))
                    biggest.append(contour)
            return tuple(biggest)
        elif num_contours > len(contours) > 1:
            return contours
        else:
            return contours
    except:
        print(traceback.format_exc())


#--------------------------------------------------imageFile is some RGB image

imageFile = os.path.join('Xiao_1um.png')

pix2 = cv.imread(imageFile)

#-------------------------------------------------Convert the BRG image to RGB
img = cv.cvtColor(pix2, cv.COLOR_BGR2RGB)

try:
    # convert to grayscale before thresholding 
    #(we *could* do color thresholding as well if necessary)
    
    grayscale_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    
    # threshold *inverted* image (first number is cutoff value, 
    # second number assigned to all pixels exceeding cutoff)
    
    bin_img = cv.threshold(grayscale_img, 
                           25, 
                           255, 
                           cv.THRESH_BINARY_INV)[1]
    
    # get list of contours:
    cntrs = getContours(cv.bitwise_not(bin_img))
    
    # keep biggest contour only:
    cntr = getBiggestContour(cntrs)
    
    # show contour:
    img_cntr = img.copy()
    cv.drawContours(img_cntr, cntr, -1, color=(255, 0, 0), thickness=3)
    fig0 = plt.imshow(img_cntr)
    
    # create blank mask:
    mask = np.full(bin_img.shape, 0, dtype=np.uint8)
    
    # use biggest contour to define ROI area:
    cv.drawContours(mask, cntr, -1, (255, 255, 255), cv.FILLED)

    #---------------------------------------------Convert the RGB image to HSV
    pix2 = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    # --------------------------------------------------Splitting HSV channels
    h, s, v = cv.split(pix2)

except:
    print(traceback.format_exc()) 

#--------------------------------Making some empty matrices of the same size

pix_test = np.zeros((pix2.shape[0],pix2.shape[1]))
pix_mid = np.zeros((pix2.shape[0],pix2.shape[1]))

#----------------------------------------Creating a list of all the HSV values
flat_pix2=pix2.reshape((pix2.shape[1]*pix2.shape[0],3))
flat_mask = mask.reshape((mask.shape[1] * mask.shape[0]), 1)

#----------------------------------------Creating a list of all the RGB values
flat_img=img.reshape((img.shape[1]*img.shape[0],3))

#--------------------------------------------------------------------------
#----------------------------------------------------FILTERING-------------
#--------------------------------------------------------------------------
dictionary_HSV = {
    'H':flat_pix2[:,0],
    'S':flat_pix2[:,1],
    'V':flat_pix2[:,2],
    'filter':flat_mask[:,0]}

df = pd.DataFrame(dictionary_HSV)


dictionary_RGB = {
    'R':flat_img[:,0],
    'G':flat_img[:,1],
    'B':flat_img[:,2],
    'filter':flat_mask[:,0]}

dfimg = pd.DataFrame(dictionary_RGB)


#------------------------------------Now filter just the image portion
#------------------------------------use "!=" instead of "==" to get background
df2 = df.loc[df['filter'] == 255]
df3 = dfimg.loc[dfimg['filter'] == 255]

#---------------------------------------------------------------------------
#-----------------------------------------------HISTOGRAM ------------------
#---------------------------------------------------------------------------
from scipy.stats import circmean, circstd

#-------------------------------------------------------------------------RGB
R_DIST = df3['R']
R_mu = np.mean(R_DIST)
R_sig = np.std(R_DIST)

G_DIST = df3['G']
G_mu = np.mean(G_DIST)
G_sig = np.std(G_DIST)

B_DIST = df3['B']
B_mu = np.mean(B_DIST)
B_sig = np.std(B_DIST)

fig1, axes = plt.subplots(1, 3, sharey=False, tight_layout=True)

# We can set the number of bins with the *bins* keyword argument.
axes[0].hist(R_DIST, bins=256)
axes[0].set_title('R_dist\n'
                 fr'$\mu={R_mu:.0f}$, $\sigma={R_sig:.0f}$')

axes[1].hist(G_DIST, bins=256)
axes[1].set_title('G_dist\n'
                 fr'$\mu={G_mu:.0f}$, $\sigma={G_sig:.0f}$')

axes[2].hist(B_DIST, bins=256)
axes[2].set_title('B_dist\n'
                 fr'$\mu={B_mu:.0f}$, $\sigma={B_sig:.0f}$')

#----------------------------------------------------------------------H Data
#----------------------------------circular mean and stdev since H is a circle
H_DIST = df2['H'].round(0).astype(int)
#H_DIST = H_DIST*2
rads = np.deg2rad(H_DIST)
circmn = circmean(rads*2)
h_mu = np.rad2deg(circmn)


crcstd = circstd(rads*2)
h_sig = np.rad2deg(crcstd)

#-----------------------------------------------------------------------S Data
S_DIST = df2['S']
s_mu = np.mean(S_DIST)
s_sig = np.std(S_DIST)



#-----------------------------------------------------------------------V Data
V_DIST = df2['V']

#------------For filtering just the V values from the thresholding filter
#FILTERED_V_DIST = [i for i,j in zip(V_DIST, flat_mask) if j == 255]

v_mu = np.mean(V_DIST)
v_sig = np.std(V_DIST)



fig2, axs = plt.subplots(1, 3, sharey=False, tight_layout=True)

# We can set the number of bins with the *bins* keyword argument.
axs[0].hist(H_DIST, bins=360)
axs[0].set_title('H_dist\n'
                 fr'$\mu={h_mu:.0f}$, $\sigma={h_sig:.0f}$')

axs[1].hist(S_DIST, bins=256)
axs[1].set_title('S_dist\n'
                 fr'$\mu={s_mu:.0f}$, $\sigma={s_sig:.0f}$')

axs[2].hist(V_DIST, bins=256)
axs[2].set_title('V_dist\n'
                 fr'$\mu={v_mu:.0f}$, $\sigma={v_sig:.0f}$')

#-----------------------------------------------------------------------------   
#-----------------------------------------------SCATTER PLOT -----------------
#-----------------------------------------------------------------------------


#----------------------------------------------------------Define size of dots
area = 1.5

#------------------------------------Making sure color values are not squished
graphbounds = pd.Series([0, pi/2, pi])
satbounds = pd.Series([255, 255, 255])

rads = pd.concat([rads, graphbounds])
S_DIST = pd.concat([S_DIST, satbounds])


#--------------------------------------------------------------PLOTTING STARTS
fig3 = plt.figure()
axe = fig3.add_subplot(projection='polar')
axe.set_yticks([100, 200, 255])
axe.errorbar(circmn,s_mu, 
             xerr= crcstd,yerr= s_sig,
             capsize=7,
             fmt= '^',
             c='k')
c = axe.scatter(rads*2, 
               S_DIST,
               c=rads, 
               s=area,
               cmap='hsv', 
               alpha=1)


#----------------------------------OpenCV only has H values between 0 and 180
#ax.set_thetamin(0)
#ax.set_thetamax(180)









