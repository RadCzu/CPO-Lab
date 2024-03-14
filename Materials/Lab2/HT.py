# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 22:55:00 2022

@author: Admin


Zapoznaj się z funkcjami cv2.HoughLines i cv2.HoughLinesP.
W trakcie ćwiczenia dobierz wartości parametrów dla tych funkcji oraz
dla metody Canny


"""
import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt 

# %% 
# 1 
 
# wczytanie obrazu w skali szarości 
img = cv2.imread('D:\\envs\\215ICCzujR_env\\Images\\fruits.jpg', cv2.IMREAD_GRAYSCALE) 
 
# obliczenie histogramu za pomocą funkcji histogram() z biblioteki NumPy 
hist_img, bins = np.histogram(a=img, bins=256, range=[0,256]) 
 
# obliczenie dystrybuanty i jej normalizacja 
cdf_img = hist_img.cumsum() 
cdf_img_normalized = cdf_img * float(hist_img.max()) / cdf_img.max()


# %%
# Wczytanie obrazu
img_orig = cv2.imread('images/akumulator.png')
# img_orig = cv2.imread('sudoku.png')
img1 = img_orig.copy()

# Konwersja na skalę szarości
gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

cv2.imshow('Obraz wejściowy', gray)


# %%

# Wyszukiwanie krawędzi operatorem Canny
edges = cv2.Canny(gray,50,200)

cv2.imshow('Obraz Canny', edges)



# %%
# Zastosowanie transformacji Hough'a
# Klasa HoughLines

lines = cv2.HoughLines(edges,1,np.pi/180, 150)

# Rysowanie linii
for line in lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(img1,(x1,y1),(x2,y2),(0,0,255),1)
cv2.imshow('Obraz z HT', img1)


# %%
# Zastosowanie transformacji Hough'a
# Klasa HoughLinesP

img2 = img_orig.copy()
lines2 = cv2.HoughLinesP(edges,1,np.pi/180,40, None, 180, 15)
for x1,y1,x2,y2 in lines2[0]:
    cv2.line(img2,(x1,y1),(x2,y2),(0,0,255),1)

for i in range(1, len(lines2)):
    l = lines2[i][0]
    cv2.line(img2, (l[0], l[1]), (l[2], l[3]), (0,255,0), 1)
    
cv2.imshow('Obraz z HTP', img2)
