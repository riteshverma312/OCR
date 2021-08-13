#GEOMETRICAL RECTIFICATION
from __future__ import print_function
import cv2
import numpy as np


MAX_MATCHES = 500
GOOD_MATCH_PERCENT = 0.15


def alignImages(im1, im2):

  # Convert images to grayscale
  im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
  
  # Detect ORB features and compute descriptors.
  orb = cv2.ORB_create(MAX_MATCHES)
  keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
  
  # Match features.
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = matcher.match(descriptors1, descriptors2, None)
  
  # Sort matches by score
  matches.sort(key=lambda x: x.distance, reverse=False)

  # Remove not so good matches
  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  matches = matches[:numGoodMatches]

  # Draw top matches
  imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
  cv2.imwrite("matches.jpg", imMatches)
  
  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)

  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt
  
  # Find homography
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

  # Use homography
  height, width, channels = im2.shape
  im1Reg = cv2.warpPerspective(im1, h, (width, height))
  
  return im1Reg, h


if __name__ == '__main__':
  
  # Read reference image
  refFilename = "h1.jpg"
  print("Reading reference image : ", refFilename)
  imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

  # Read image to be aligned
  imFilename = "h2.jpg"
  print("Reading image to align : ", imFilename);  
  im = cv2.imread(imFilename, cv2.IMREAD_COLOR)
  
  print("Aligning images ...")
  # Registered image will be resotred in imReg. 
  # The estimated homography will be stored in h. 
  imReg, h = alignImages(im, imReference)
  
  # Write aligned image to disk. 
  outFilename = "aligned.jpg"
  print("Saving aligned image : ", outFilename); 
  cv2.imwrite(outFilename, imReg)

  # Print estimated homography
  print("Estimated homography : \n",  h)




#PREPROCESSING AND DETECTION
import cv2
import numpy as np
import mapper
from matplotlib import pyplot as plt


image=cv2.imread("maam2.jpg")   #read in the image
image=cv2.resize(image,(600,450)) #resizing because opencv does not work well with bigger images
orig=image.copy()


hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower = np.array([0, 0, 218])
upper = np.array([157, 54, 255])
mask = cv2.inRange(hsv, lower, upper)



blurred=cv2.GaussianBlur(mask,(5,5),0)  #(5,5) is the kernel size and 0 is sigma that determines the amount of blur
#cv2.imshow("Blur",blurred)

edged1=cv2.Canny(image,30,50)
cv2.imshow("CANNY",edged1)

edged=cv2.Canny(blurred,30,50)  #30 MinThreshold and 50 is the MaxThreshold
cv2.imshow("Canny",edged)

ret, thresh = cv2.threshold(edged, 50, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('image1', thresh)
cv2.imwrite("h2kcw2/out1.png", thresh)

# Create horizontal kernel and dilate to connect text characters
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,3))
dilate = cv2.dilate(mask, kernel, iterations=5)

# Find contours and filter using aspect ratio
# Remove non-text contours by filling in the contour
cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    ar = w / float(h)
    if ar < 5:
        cv2.drawContours(dilate, [c], -1, (0,0,0), -1)

# Bitwise dilated image with mask, invert, then OCR
result = 255 - cv2.bitwise_and(dilate, mask)
#data = pytesseract.image_to_string(result, lang='eng',config='--psm 6')
#print(data)


cv2.imshow('mask', mask)
#cv2.imshow('dilate', dilate)
#cv2.imshow('result', result)

outFilename = "MASK.jpg"
print("Saving MASK image : ", outFilename);
cv2.imwrite(outFilename, mask)

outFilename1 = "CANNY.jpg"
print("Saving CANNY image : ", outFilename1);
cv2.imwrite(outFilename1, edged)

outFilename2 = "CANNY1.jpg"
print("Saving CANNY image : ", outFilename1);
cv2.imwrite(outFilename2, edged1)

cv2.waitKey(0)
cv2.destroyAllWindows()



#TESSERACT CODE
import cv2
import sys
import pytesseract
#import pytesseract as tess
#tess.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
from PIL import Image
img =Image.open('MASK.jpg')
#text= tess.image_to_string(img)
#print(text)

config = ('-l eng --oem 1 --psm 3')

# Read image from disk
#im = cv2.imread(img, cv2.IMREAD_COLOR)

# Run tesseract OCR on image
text = pytesseract.image_to_string(img, config=config)

# Print recognized text
print(text)

with open("output_data.txt" , "w") as out_file:
    for i in range(len(text)):
        out_string = ""
        out_string += str(text[i])
        out_file.write(out_string)




        
#AUDIO CONVERSION
# Import the required module for text 
# to speech conversion 
from gtts import gTTS 

# This module is imported so that we can 
# play the converted audio 
import os 

# The text that you want to convert to audio
f=open("output_data.txt","r")
if f.mode == 'r':
    contents=f.read()

mytext = contents

# Language in which you want to convert 
language = 'en'

# Passing the text and language to the engine, 
# here we have marked slow=False. Which tells 
# the module that the converted audio should 
# have a high speed 
myobj = gTTS(text=mytext, lang=language, slow=False) 

# Saving the converted audio in a mp3 file named 
# welcome 
myobj.save("welcome.mp3") 

