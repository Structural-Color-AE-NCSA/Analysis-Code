# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 22:38:00 2023

Accepts image and then spits out 
HSV histogram and color scatterplot

-FIX STATISTICS

@author: J.A. LOMAS
"""

import cv2 as cv
import os
import numpy as np
 

#------------------------------------------------imageFile is some RGB image

imageFile = os.path.join('Green_Sanghyun.jpg')

pix2 = cv.imread(imageFile)

#-----------------------------------------------Convert the BRG image to RGB
img = cv.cvtColor(pix2, cv.COLOR_BGR2RGB)

#-----------------------------------------------Convert the RGB image to HSV
pix2 = cv.cvtColor(img, cv.COLOR_RGB2HSV)


#-----------------------------------------------------Splitting HSV channels
h, s, v = cv.split(pix2)

#--------------------------------Making some empty matrices of the same size

pix_test = np.zeros((pix2.shape[0],pix2.shape[1]))
pix_mid = np.zeros((pix2.shape[0],pix2.shape[1]))

#----------------------------------------Creating a list of all the HSV values
flat_pix2=pix2.reshape((pix2.shape[1]*pix2.shape[0],3))


#---------------------------------------------------------------------------   
#-----------------------------------------------HISTOGRAM ------------------
#---------------------------------------------------------------------------
from scipy.stats import circmean, circstd
import matplotlib.pyplot as plt

H_DIST = flat_pix2[:,0]
rads = np.deg2rad(H_DIST)
circmean = circmean(rads,
                    high = np.pi, 
                    low=0)
h_mu = np.rad2deg(circmean)


circstd = circstd(rads, 
                    high = np.pi, 
                    low=0)
h_sig = np.rad2deg(circstd)

S_DIST = flat_pix2[:,1]
s_mu = np.mean(S_DIST)
s_sig = np.std(S_DIST)

V_DIST = flat_pix2[:,2]
v_mu = np.mean(V_DIST)
v_sig = np.std(V_DIST)



fig1, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)

# We can set the number of bins with the *bins* keyword argument.
axs[0].hist(H_DIST, bins=180)
axs[0].set_title('H_dist\n'
                 fr'$\mu={h_mu:.0f}$, $\sigma={h_sig:.0f}$')

axs[1].hist(S_DIST, bins=256)
axs[1].set_title('S_dist\n'
                 fr'$\mu={s_mu:.0f}$, $\sigma={s_sig:.0f}$')

axs[2].hist(V_DIST, bins=256)
axs[2].set_title('V_dist\n'
                 fr'$\mu={v_mu:.0f}$, $\sigma={v_sig:.0f}$')

#---------------------------------------------------------------------------   
#-----------------------------------------------SCATTER PLOT ---------------
#---------------------------------------------------------------------------


#--------------------------------------------------------Define size of dots
area = 1.5

fig2 = plt.figure()
ax = fig2.add_subplot(projection='polar')

c = ax.scatter(np.pi*H_DIST/180, S_DIST,
               c=np.pi*H_DIST/180, s=area,
               cmap='hsv', alpha=0.75)

#----------------------------------OpenCV only has H values between 0 and 180
ax.set_thetamin(0)
ax.set_thetamax(180)

plt.show()








