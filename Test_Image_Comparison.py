#!/usr/bin/env python
# coding: utf-8

# In[201]:


import skimage 
from skimage import io
from skimage import color
from skimage import data
from pylab import *
from skimage import draw
from skimage import exposure
from skimage import transform as tr
from skimage import measure

import cv2


# In[223]:


org_seis = cv2.imread('D:\\OfficeWork\\imagesforscript\\OrgPorject_Seis.jpg')
imp_seis = cv2.imread('D:\\OfficeWork\\imagesforscript\\Imported_Seis.jpg')

#imp_seis = tr.resize( imp_seis, (org_seis.shape[0],org_seis.shape[1]) )


# In[224]:


print("Original Imported image size : ", imp_seis.shape)
print("Imported image size after resize : ", imp_seis_resize.shape)
print("Original image size : ", org_seis.shape)
print(type(imp_seis_resize.shape[1]))
print(type(org_seis.shape[1]))


# In[233]:


#FUNCTIONS
def mse( org_image, imp_image ):
    height = org_image.shape[0]
    width = org_image.shape[1]
    diff = cv2.subtract(org_image, imp_image,dtype=cv2.CV_32F)
    err = np.sum( diff**2 )
    mse = err/(float(height*width))
    return mse

def image_reSize( image, ratio ):
    length = int( image.shape[1] * ratio )
    width = int( image.shape[0] * ratio )
    dim = ( length, width )
    return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

def plot_gray(image):
    plt.figure(figsize=(16,10))
    return plt.imshow(image, cmap='Greys_r')

def bw_scanner(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    T = threshold_local(gray, 21, offset = 5, method = "gaussian")
    return (gray > T).astype("uint8") * 255

def seismicCropImage( image ):
    resize_ratio =  0.5
    seis = image_reSize( image, resize_ratio )
    seis_gray = cv2.cvtColor( seis, cv2.COLOR_BGR2GRAY )
    # Get rid of noise with Gaussian Blur filter
    blurred = cv2.GaussianBlur( seis_gray, (7,7), 0 )
    
    # Detect white regions
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(blurred,rectKernel)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11,11))
    eroded = cv2.erode(dilated,rectKernel)
    
    #Blurred image outlining
    blurred_edged = cv2.Canny(blurred, 100, 200, apertureSize=3)
    
    eroded_edged = cv2.Canny(eroded, 100, 200, apertureSize=3)
    
    contours, hierarchy = cv2.findContours(eroded_edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image_with_contours = cv2.drawContours(seis.copy(), contours, -1, (255,0,0), 1)
    
    seismic_window_contour = sorted(contours, key = cv2.contourArea, reverse = True)[:1]
    image_with_seismic_window_contour = cv2.drawContours(seis.copy(), seismic_window_contour, -1, (0,255,0), 1)
    #print("Image with Seismic Window Contour")
    #plot_gray(image_with_seismic_window_contour)
    
    for c in seismic_window_contour:
        x,y,w,h = cv2.boundingRect(c)
        actual_seismic_image = seis_gray[y:y+h, x:x+w]
        break
    
    #cv2.imshow('Actual Seismic Image',actual_seismic_image)
    #plot_gray(actual_seismic_image)
    
    return actual_seismic_image


# In[234]:


org_seis_sec = seismicCropImage(org_seis)
plot_gray(org_seis_sec)


# In[235]:


imp_seis_sec = seismicCropImage(imp_seis)
plot_gray(imp_seis_sec)


# In[236]:


#converting image to GrayScale
#org_seis_gray = cv2.cvtColor( org_seis_sec, cv2.COLOR_BGR2GRAY )
#imp_seis_gray = cv2.cvtColor( imp_seis_sec, cv2.COLOR_BGR2GRAY )

#Resize Images
print("Before : Original Seismic Shape")
print(org_seis_sec.shape)

print("Before : Imported Seismic Shape")
print(imp_seis_sec.shape)

org_seis_sec = tr.resize( org_seis_sec, (imp_seis_sec.shape[0],imp_seis_sec.shape[1]) )

print("After : Original Seismic Shape")
print(org_seis_sec.shape)

print("After : Imported Seismic Shape")
print(imp_seis_sec.shape)

print("Plotting Original Seismic after processing")
plot_gray(org_seis_sec)

print("Plotting Imported Seismic section after processing")
plot_gray(imp_seis_sec)

mse_comp = mse(org_seis_sec,imp_seis_sec)
print("MSE : " + str(mse_comp) )


# In[ ]:




