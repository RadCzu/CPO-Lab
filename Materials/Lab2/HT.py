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
# 1.1
 
# wczytanie obrazu w skali szarości 
img = cv2.imread('D:\\envs\\215ICCzujR_env\\Images\\fruits.jpg', cv2.IMREAD_GRAYSCALE) 
 
# obliczenie histogramu za pomocą funkcji histogram() z biblioteki NumPy 
hist_img, bins = np.histogram(a=img, bins=256, range=[0,256]) 
 
# obliczenie dystrybuanty i jej normalizacja 
cdf_img = hist_img.cumsum() 
cdf_img_normalized = cdf_img * float(hist_img.max()) / cdf_img.max()


# %% 
# 1.2 
 
# Wykres podglądowy histogramu i dytrybuanty 
plt.rcParams['figure.figsize'] = [10, 5] 
fig, axs = plt.subplots(1, 2, tight_layout=True) 
 
axs[0].plot(np.arange(256), cdf_img_normalized) 
axs[0].plot(np.arange(256), hist_img) 
axs[1].plot(np.arange(256), cdf_img) 
axs[0].legend(('cdf normalized', 'histogram'), loc = 'upper left') 
axs[1].legend(('cdf',), loc = 'upper left') 

# %% 
# 1.3 
# Wyrównanie histogramu za pomocą funkcji cv2.equalizeHist 
img_equ = cv2.equalizeHist(img) 
# obliczenie histogramu za pomocą funkcji histogram() z biblioteki NumPy 
hist_img_equ, bins = np.histogram(a=img_equ, bins=256, range=[0,256]) 
# obliczenie dystrybuanty obrazu img_equ i jej normalizacja 
cdf_img_equ = hist_img_equ.cumsum() 
cdf_img_equ_normalized = cdf_img_equ * float(hist_img_equ.max()) / cdf_img_equ.max() 

# Wykres podglądowy histogramu i dytrybuanty obrazu po wyrównaniu histogramu 
plt.rcParams['figure.figsize'] = [10, 5] 
fig, axs = plt.subplots(1, 2, tight_layout=True) 
axs[0].plot(np.arange(256), cdf_img_equ_normalized) 
axs[0].plot(np.arange(256), hist_img_equ) 
axs[1].plot(np.arange(256), cdf_img_equ) 
axs[0].legend(('cdf equ normalized', 'histogram equ'), loc = 'upper left') 
axs[1].legend(('cdf equ',), loc = 'upper left') 

# %% 
# 1.4 
plt.rcParams['figure.figsize'] = [15, 10] 
fig, axs = plt.subplots(2, 2, tight_layout=True) 
axs[0,0].imshow(img, cmap='gray') 
axs[0,1].hist(img.flatten(),256,[0,256], color = 'r') 
axs[0,1].plot(cdf_img_normalized, color = 'b') 
axs[0,1].legend(('cdf','histogram'), loc = 'upper left') 
axs[1,0].imshow(img_equ, cmap='gray') 
axs[1,1].hist(img_equ.flatten(),256,[0,256], color = 'r') 
axs[1,1].plot(cdf_img_equ_normalized, color = 'b') 
axs[1,1].legend(('cdf equ','histogram equ'), loc = 'upper left')

# %% 
# 2
wikipath = "D:\\envs\\215ICCzujR_env\\Images\\wiki.jpg"
planepath = "D:\\envs\\215ICCzujR_env\\Images\\plane.jpg"
def clashe(path):
    plt.clf()
    plt.rcParams['figure.figsize'] = [15, 10] 
    img = cv2.imread(path, 0) 
     
    plt.subplot(421) 
    plt.imshow(img, cmap='gray') 
     
    plt.subplot(422) 
    plt.hist(img.ravel(), bins=256, range=[0, 256]) 
     
    plt.subplot(423) 
    img_equ = cv2.equalizeHist(img) 
    plt.imshow(img_equ, cmap='gray') 
     
    plt.subplot(424) 
    plt.hist(img_equ.ravel(), bins=256, range=[0, 256]) 
     
    plt.subplot(425) 
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)) 
    img_equ_clahe = clahe.apply(img) 
    plt.imshow(img_equ_clahe, cmap='gray') 
     
    plt.subplot(426) 
    plt.hist(img_equ_clahe.ravel(), bins=256, range=[0, 256]) 
    
    plt.subplot(427) 
    img_clache2 = clahe.apply(img_equ_clahe) 
    plt.imshow(img_clache2, cmap='gray')
    
    plt.subplot(428) 
    plt.hist(img_clache2.ravel(), bins=256, range=[0, 256]) 
    plt.show()

clashe(wikipath)
#%%
clashe(planepath)

#%%
# 3.1
fruitpath = "D:\\envs\\215ICCzujR_env\\Images\\fruits.jpg"
plt.clf()
img_bgr = cv2.imread(fruitpath) 
 
img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB) 
img_lab2 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8, 8)) 
img_lab[..., 0] = clahe.apply(img_lab[..., 0]) 
img_lab2[..., 0] = clahe.apply(img_lab[..., 0]) 
img_equ = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR) 
img_hsv = cv2.cvtColor(img_lab2, cv2.COLOR_HSV2BGR)
 
plt.rcParams['figure.figsize'] = [8, 6] 
 
plt.subplot(321) 
plt.imshow(img_bgr[...,::-1]) 
plt.title("BGR") 
 
plt.subplot(322) 
plt.hist(img_bgr[..., 0].ravel(), bins=256, range=[0, 256], color='b') 
plt.hist(img_bgr[..., 1].ravel(), bins=256, range=[0, 256], color='g') 
plt.hist(img_bgr[..., 2].ravel(), bins=256, range=[0, 256], color='r') 
 
plt.subplot(323) 
plt.imshow(img_equ[...,::-1]) 
plt.title("LAB") 
 
plt.subplot(324) 
plt.hist(img_equ[..., 0].ravel(), bins=256, range=[0, 256], color='b') 
plt.hist(img_equ[..., 1].ravel(), bins=256, range=[0, 256], color='g') 
plt.hist(img_equ[..., 2].ravel(), bins=256, range=[0, 256], color='r') 

plt.subplot(325) 
plt.imshow(img_equ[...,::-1]) 
plt.title("HSV") 
 
plt.subplot(326) 
plt.hist(img_hsv[..., 0].ravel(), bins=256, range=[0, 256], color='b') 
plt.hist(img_hsv[..., 1].ravel(), bins=256, range=[0, 256], color='g') 
plt.hist(img_hsv[..., 2].ravel(), bins=256, range=[0, 256], color='r') 

 
plt.show() 

# %% 
# 4.1 

okretpath = "D:\\envs\\215ICCzujR_env\\Images\\okret.jpg"
 
def imshow(image): 
    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[-1] == 1): 
        plt.imshow(image, cmap='gray') 
    else: 
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
 
# import biblioteki scipy 
from scipy import ndimage 
 
# wczytujemy obraz w skali szarości 
src_gray = cv2.imread(okretpath, 0) 
print(src_gray.shape) 
 
# przygotowanie filtra 
kernel = np.array([[1,1,1],[1,1,0],[1,0,0]]) 
 
# zastosujemy filtr konwolucyjny z biblioteki scipy 
dst_conv  = ndimage.convolve(src_gray, kernel, mode='constant', cval=1.0) 
 
# zastosowanie oryginalnego filtra do celów korelacji 
dst_corr = cv2.filter2D(src_gray, -1, kernel) 
 
# złączenie obrazów 
result = np.concatenate((src_gray, dst_corr, dst_conv), axis=1) 
cv2.imshow('wynik', result) 

#%%
# 4.2
# Filtr uśredniający – obraz RGB 
# Funkcja blur() 
 
# wczytujemy obraz RGB 
src = cv2.imread(okretpath) 
print(src.shape) 
# zastosowanie funkcji blur jako filtra uśredniającego 
blur1 = cv2.blur(src, (5,5)) 
# wyświetlenie obrazów oryginalnego i po uśrednieniu za pomocą plt 
plt.figure() 
plt.subplot(121), imshow(src), plt.title('Oryginał') 
plt.xticks([]), plt.yticks([]) 
plt.subplot(122), imshow(blur1), plt.title('Blurred cv2.blur()') 
plt.xticks([]), plt.yticks([]) 
plt.show()  

# %% 
# 4 .3
# Filtr uśredniający – obraz RGB 
# Funkcja filter2D() 
# przygotowanie i wydruk maski 
size = 5 
kernel = 1 / size**2 * np.ones((size, size)) 
print(kernel) 
# zastosowanie funkcji filter2D jako filtra uśredniającego 
blur2 = cv2.filter2D(src, cv2.CV_8U, kernel) 
print(blur2.shape) 
# wyświetlenie obrazów oryginalnego i po uśrednieniu za pomocą plt 
plt.figure() 
plt.subplot(121), imshow(src), plt.title('Oryginał') 
plt.xticks([]), plt.yticks([]) 
plt.subplot(122), imshow(blur2), plt.title('Blurred cv2.filter2D') 
plt.xticks([]), plt.yticks([]) 
plt.show()

# %% 
# 4.4
# Filtr uśredniający - obraz w skali szarości 
# Funkcja filter2D() 
# przygotowanie i wydruk maski 
size = 5
kernel = 1 / size**2 * np.ones((size, size)) 
print(kernel) 
# zastosowanie funkcji filter2D jako filtra uśredniającego 
blur3 = cv2.filter2D(src_gray, -1, kernel) 
print(blur3.shape) 
# połączenie obrazów w poziomie i wyświetlenie za pomocą cv2 
result = np.concatenate((src_gray, blur3), axis=1) 
cv2.imshow('blured gray', result)

# %% 
# 4.5
# Filtr gaussowski - obraz RGB 
# Funkcja GaussianBlur() 
blur_4 = cv2.GaussianBlur(src,(5,5), 0) 
plt.subplot(121), imshow(src), plt.title('Oryginal') 
plt.xticks([]), plt.yticks([]) 
plt.subplot(122), imshow(blur_4), plt.title('Blurred GaussianBlur') 
plt.xticks([]), plt.yticks([]) 
plt.show() 
# %% 
# 4.6
# Filtr medianowy - obraz RGB 
# Funkcja medianBlur() 
# Generowanie szumu gaussowskiego 
gauss = np.random.normal(0,1,src.size) 
gauss = gauss.reshape(src.shape[0],src.shape[1],src.shape[2]).astype('uint8') 
# Dodanie szumu gaussowskiego do obrazu 
src_gauss = cv2.add(src,gauss) 
# zastosowanie filtra do obrazu pierwotnego, filtr staje się tu tablicą np 
blur5 = cv2.medianBlur(src_gauss, 5) 
# połączenie obrazów do wyświetlenia 
result = np.concatenate((src, src_gauss, blur5), axis=1) 
cv2.imshow('wynik', result) 

 
# %% 
# 4.7
# Filtr bilateralny - obraz RGB 
# Funkcja bilateralFilter() 
 
 
blur6 = cv2.bilateralFilter(src, 
                          d=11, 
                          sigmaColor=60, 
                          sigmaSpace=60) 
 
blur7 = cv2.bilateralFilter(src_gauss, 
                          d=11, 
                          sigmaColor=60, 
                          sigmaSpace=60) 
 
plt.figure() 
plt.subplot(221), imshow(src), plt.title('Original') 
plt.xticks([]), plt.yticks([]) 
plt.subplot(222), imshow(blur6), plt.title('Blur6 bilateralFilter') 
plt.xticks([]), plt.yticks([]) 
plt.subplot(223), imshow(src_gauss), plt.title('Gaussian noise') 
plt.xticks([]), plt.yticks([]) 
plt.subplot(224), imshow(blur7), plt.title('Blur7 bilateralFilter') 
plt.xticks([]), plt.yticks([]) 
plt.show() 
 
result = np.concatenate((src_gauss, blur7), axis=1) 
cv2.imshow('wynik', result)

# %% 
# 4.8
# Wyostrzenie obrazu 
kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) 
print(kernel) 
sharp = cv2.filter2D(src, cv2.CV_8U, kernel) 
plt.subplot(121), imshow(src), plt.title('Original') 
plt.xticks([]), plt.yticks([]) 
plt.subplot(122), imshow(sharp), plt.title('Sharpened') 
plt.xticks([]), plt.yticks([])
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

#%%

cv2.imshow('Obraz z HTP', img2)# Obraz wejściowy 

input_image = np.array(( 
[0, 0, 0, 0, 0, 0, 0, 0], 
[0, 255, 255, 255, 0, 0, 0, 255], 
[0, 0, 255, 255, 0, 0, 0, 0], 
[0, 0, 255, 255, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0], 
[0, 255, 255, 0, 0, 255, 255, 255], 
[0,255, 0, 0, 0, 0, 255, 0], 
[0, 255, 0, 0, 0, 0, 255, 0]), dtype="uint8") 
# Element strukturalny 
kernel = np.array(( 
[1, 1, 1], 
[0, 1, -1], 
[0, 1, -1]), dtype="int") 
# Zastosuj transformację hit-or-miss 
output_image = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel) 
print(input_image) 
print("\n\n") 
print(output_image) 
