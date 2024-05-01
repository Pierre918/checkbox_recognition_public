# MEILLEUR RESULTAT DANS CE FICHIER
# https://www.youtube.com/watch?v=iWS9ogMPOI0
###############################################################
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import gcsfs
from typing import Union
from fastapi import FastAPI
from PIL import Image
import json
import io
import base64
from pydantic import BaseModel

app = FastAPI()
class Message(BaseModel):
    image_encoded: str #required

@app.get("/")
def root():
    return {"My API"}

@app.post("/coordinates")
def get_coord(message: Message):
    def decode_message():
        image = base64.b64decode(message.image_encoded)
        image=Image.open(io.BytesIO(image))
        return np.array(image)
    def show_image(img, colorspace=cv2.COLOR_GRAY2RGB):
        plt.figure(figsize=(12,12))
        plt.imshow(cv2.cvtColor(img, colorspace))
        plt.show()

    def draw_contours(contours):
        show_image(cv2.drawContours(img.copy(), contours, -1, (0, 255, 0), 1), cv2.COLOR_BGR2RGB)

    def get_contours(img_bin):
        contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #draw_contours(contours)
        return contours





    def filter_contours(contours, func):
        _contours = [x for x in contours if func(x)]
        #draw_contours(_contours)
        return _contours

    def is_correct_area(contour, min_ratio_area=0.0005, max_ratio_area=0.3):
        height, width = img.shape[:2]
        ratio_area = cv2.contourArea(contour)/(height*width)
        print(ratio_area)
        return ratio_area<max_ratio_area and ratio_area>min_ratio_area
    def approx(contours):
        approxs=[]
        for cnt in contours:
        # Approximer la forme du contour
            epsilon = 0.035 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            # Si l'approximation a quatre points, alors c'est probablement un rectangle
            if len(approx) == 4 and check_angles(approx):
                
                approxs.append(approx)
        #draw_contours(contours)
        print(approxs)
        #draw_contours(approxs)
        return approxs
    def check_angles(approx):
        rectangle_is_ok=True
        for i in range(4):
            point1 = approx[i][0]
            point2 = approx[(i+1)%4][0]
            point3 = approx[(i+2)%4][0]
            vector1 = [point1[0]-point2[0], point1[1]-point2[1]]
            vector2 = [point3[0]-point2[0], point3[1]-point2[1]]

            angle = np.arctan2(vector2[1], vector2[0]) - np.arctan2(vector1[1], vector1[0])
            
            if angle < 0:
                angle += 2*np.pi
            angle =np.degrees(angle)
            print(angle)
            if angle>100:
                print('tej : ',angle)
                rectangle_is_ok=False
                break

        return rectangle_is_ok


    def threshold():
        
        img_bin = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        mean_brightness = np.mean(img_bin)
        is_dark = mean_brightness < 128

        kernel = np.ones((2,2),np.uint8)
        img_bin = cv2.erode(img_bin,kernel,iterations = 1)
        # Image thresholding
        _, img_bin = cv2.threshold(img_bin, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        if not is_dark:
            img_bin = 255 - img_bin
        return img_bin

    def get_center_of_contour(_contours):
        coords=[]
        height, width = img.shape[:2]
        i=0
        for cnt in _contours:
            # Calculez les moments du contour
            M = cv2.moments(cnt)

            # Calculez les coordonnées du centroïde
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            coords.append((cX,cY)) 
            # Dessinez le centroïde sur l'image
            cv2.circle(img, (cX, cY), 2, (i, i, i), -1)

            i+=11
        print('centre sans tri')
        #cv2.imshow('centre sans tri',img)
        #cv2.waitKey()
        return coords
    
    img=decode_message()
    

    img_bin=threshold()
    
    thresholded_contours = get_contours(img_bin)
    thresholded_contours = filter_contours(thresholded_contours, is_correct_area)
    approxs=approx(thresholded_contours)
    coords = get_center_of_contour(approxs)

    #on trie dans le sens de la lecture
    coords = sorted(coords,key = lambda x:x[0]) 
    coords = sorted(coords,key = lambda x:x[1])
    return coords

