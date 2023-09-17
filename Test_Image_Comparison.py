#!/usr/bin/env python
# coding: utf-8

# In[1]:


import skimage 
from skimage import io
from skimage import color
from skimage import data
from pylab import *
from skimage import draw
from skimage import exposure
from skimage.filters import threshold_local
from skimage import transform as tr
from skimage import measure

import cv2


# In[3]:


org_seis = cv2.imread('D:\\OfficeWork\\imagesforscript\\OrgPorject_Seis.jpg')
imp_seis = cv2.imread('D:\\OfficeWork\\imagesforscript\\Imported_Seis.jpg')

#TestCases
personal_seis = cv2.imread('D:\\OfficeWork\\imagesforscript\\FromPersonalLaptop.jpg')
inl500_seis = cv2.imread('D:\\OfficeWork\\imagesforscript\\inl500.jpg')
OriginalSeismic_TestCase_seis = cv2.imread('D:\\OfficeWork\\imagesforscript\\OriginalSeismic_TestCase.jpg')

inl425_4DSMF_mirrored_seis = cv2.imread('D:\\OfficeWork\\imagesforscript\\inl425_4DSMF_mirrored.jpg')


# In[4]:


#FUNCTIONS
def mse( org_image, imp_image ):
    height = org_image.shape[0]
    width = org_image.shape[1]
    diff = cv2.subtract(org_image, imp_image, dtype=cv2.CV_32F )
    err = np.sum( diff**2 )
    mse = err/(float(height*width))
    return mse

def image_reSize( image, ratio ):
    length = int( image.shape[1] * ratio )
    width = int( image.shape[0] * ratio )
    dim = ( length, width )
    return cv2.resize(image, dim, interpolation = cv2.INTER_AREA )

def plot( image ):
    plt.figure(figsize=(16,10))
    return plt.imshow( image )

def bw_scanner(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    T = threshold_local(gray, 21, offset = 5, method = "gaussian")
    return (gray > T).astype("uint8") * 255

def seismicCropImage( image ):
    resize_ratio =  0.5
    seis = image_reSize( image.copy(), resize_ratio )
    seis_gray = cv2.cvtColor( seis, cv2.COLOR_BGR2GRAY )
    # Get rid of noise with Gaussian Blur filter
    blurred = cv2.GaussianBlur( seis_gray, (7,7), 0 )
    
    # Detect white regions
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(blurred,rectKernel)
    
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11,11))
    eroded = cv2.erode(dilated,rectKernel)
    
    eroded_edged = cv2.Canny(eroded, 100, 200, apertureSize=3)
    
    contours, hierarchy = cv2.findContours(eroded_edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image_with_contours = cv2.drawContours(seis.copy(), contours, -1, (255,0,0), 1)
    
    seismic_window_contour = sorted(contours, key = cv2.contourArea, reverse = True)[:1]
    image_with_seismic_window_contour = cv2.drawContours(seis.copy(), seismic_window_contour, -1, (0,255,0), 1)
    
    for c in seismic_window_contour:
        x,y,w,h = cv2.boundingRect(c)
        actual_seismic_image = seis[y:y+h, x:x+w]
        break
    
    return actual_seismic_image

def finalMSEImageAdjustmentAndCalc( testimage, origimage ):    
    testimage_sec = np.float32(testimage.copy())
    testimage_sec = cv2.cvtColor( testimage_sec, cv2.COLOR_BGR2GRAY )
    origimage_resize = tr.resize( origimage, (testimage_sec.shape[0],testimage_sec.shape[1]) )
    mse_comp = mse( origimage_resize, testimage_sec )
    print( "MSE : " + str(mse_comp) )


# In[5]:


plot(org_seis)
org_seis_sec = seismicCropImage(org_seis)
plot(org_seis_sec)
plot(org_seis)


# In[6]:


imp_seis_sec = seismicCropImage(imp_seis)
plot(imp_seis_sec)


# In[7]:


#converting image to GrayScale
org_seis_sec = np.float32(org_seis_sec)
imp_seis_sec = np.float32(imp_seis_sec)

org_seis_sec = cv2.cvtColor( org_seis_sec, cv2.COLOR_BGR2GRAY )
imp_seis_sec = cv2.cvtColor( imp_seis_sec, cv2.COLOR_BGR2GRAY )

#Resize Images
print("Before : Original Seismic Shape")
print(org_seis_sec.shape)

print("Before : Imported Seismic Shape")
print(imp_seis_sec.shape)

org_seis_sec_resize = tr.resize( org_seis_sec, (imp_seis_sec.shape[0],imp_seis_sec.shape[1]) )

print("After : Original Seismic Shape")
print(org_seis_sec_resize.shape)

print("After : Imported Seismic Shape")
print(imp_seis_sec.shape)

print("Plotting Original Seismic after processing")
plot(org_seis_sec)

print("Plotting Imported Seismic section after processing")
plot(imp_seis_sec)

mse_comp = mse( org_seis_sec_resize, imp_seis_sec )
print( "MSE : " + str(mse_comp) )


# In[8]:


inl500_seis_sec = seismicCropImage(inl500_seis)
plot(inl500_seis_sec)
finalMSEImageAdjustmentAndCalc(inl500_seis_sec,org_seis_sec)


# In[9]:


personal_seis_sec = seismicCropImage(personal_seis)
plot(personal_seis_sec)
finalMSEImageAdjustmentAndCalc(personal_seis_sec,org_seis_sec)


# In[10]:


OriginalSeismic_TestCase_seis_sec = seismicCropImage(OriginalSeismic_TestCase_seis)
plot(OriginalSeismic_TestCase_seis_sec)
finalMSEImageAdjustmentAndCalc(OriginalSeismic_TestCase_seis_sec,org_seis_sec)


# In[11]:


inl425_4DSMF_mirrored_seis_sec = seismicCropImage(inl425_4DSMF_mirrored_seis)
plot(inl425_4DSMF_mirrored_seis_sec)
finalMSEImageAdjustmentAndCalc(inl425_4DSMF_mirrored_seis_sec,org_seis_sec)

