# import cv2
# import numpy as np
#
# # Let's load a simple image with 3 black squares
# image = cv2.imread('test21.jpg')
# cv2.waitKey(0)
#
# # Grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # Find Canny edges
# edged = cv2.Canny(gray, 30, 200)
# cv2.waitKey(0)
#
# # Finding Contours
# # Use a copy of the image e.g. edged.copy()
# # since findContours alters the image
# contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#
# cv2.imshow('Canny Edges After Contouring', edged)
# cv2.waitKey(0)
#
# print("Number of Contours found = " + str(len(contours)))
#
# # Draw all contours
# # -1 signifies drawing all contours
# cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
#
# cv2.imshow('Contours', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import tensorflow as tf
# from PIL import Image
# import numpy as np
#
#
# pixels =[226, 137, 125, 226, 137, 125, 223, 137, 133, 223, 136, 128, 226, 138, 120, 226, 129, 116, 228, 138, 123, 227, 134, 124, 227, 140, 127, 225, 136, 119, 228, 135, 126, 225, 134, 121, 223, 130, 108, 226, 139, 119, 223, 135, 120, 221, 129, 114, 221, 134, 108, 221, 131, 113, 222, 138, 121, 222, 139, 114, 223, 127, 109, 223, 132, 105, 224, 129, 102, 221, 134, 109, 218, 131, 110, 221, 133, 113, 223, 130, 108, 225, 125, 98, 221, 130, 121, 221, 129, 111, 220, 127, 121, 223, 131, 109, 225, 127, 103, 223]
#
# # Convert the pixels into an array using numpy
# array = np.array(pixels, dtype=np.uint8)
#
# # Use PIL to create an image from the new array of pixels
# new_image = Image.fromarray(array)
# new_image.save('new.png')

# # Random initialization of a (2D array)
# a = np.array([[0, 0, 1, 0, 0, 0],
#            [0, 1, 0, 0, 2, 1],
#            [0, 0, 1, 1, 2, 2]])
# # print(a)
#
# # b will be all elements of a whenever the condition holds true (i.e only positive elements)
# # Otherwise, set it as 0
# b, c = np.where(a == 2)
# print(b)
# print(np.max(b))
#
# print(c)
# print(np.max(c))
#
# # data_tr = np.loadtxt('C:\\Users\\dpati\\OneDrive\\Desktop\\datapart\\data\\savedata\\all_tr.csv', delimiter=',',
# #                      dtype=int)
# data_cv = np.loadtxt('C:\\Users\\dpati\\OneDrive\\Desktop\\datapart\\data\\savedata\\all_cv.csv', delimiter=',',
#                      dtype=int)
# eval_data = data_cv[:, :-1].astype(np.float32)
# eval_labels = data_cv[:, -1].astype(np.int32)
#
#
# print(eval_data.shape)
# eval_data = tf.reshape(eval_data, [-1, 45, 45, 1])
# print("new shape", eval_data.shape)
# print(eval_labels.shape)
# print(eval_data[0])
# print(eval_data[0].shape)
# print(eval_labels)
# print(np.unique(eval_labels))
# print(data_cv.shape)

import sys
from os import listdir

#
# img_path = 'C:\\Users\\dpati\\OneDrive\\Desktop\\datapart\\data\\extracted_images\\'
# img_names = listdir(img_path)
# print(img_names)

from PIL import Image
from io import BytesIO
import cv2
import base64
import numpy as np
from matplotlib import pyplot as plt
data='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'
# fh = open('filetestwd11.jpg', 'wb')
# fh.write(base64.b64decode((data)))
# fh.close()
# with open("generatedImage.jpg", 'wb+') as image:
#     image.write(base64.b64decode(data))
# imgdata = base64.b64decode(data)
image = Image.open(BytesIO(base64.b64decode(data)))
print(type(image))
# imageLabel = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
X_gray_smooth = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
X_gray_smooth1 = cv2.cvtColor(cv2.imread('filetestwd11.jpg').astype(np.uint8), cv2.COLOR_BGR2GRAY)
plt.imshow(X_gray_smooth)
plt.axis('off')
plt.show()
plt.imshow(X_gray_smooth1)
plt.axis('off')
plt.show()