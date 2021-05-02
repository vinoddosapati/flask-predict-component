from os import listdir
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

from dataprocess.normilize import process_image

# img_path = 'C:\\Users\\dpati\\OneDrive\\Desktop\\datapart\\data\\extracted_images\\'
img_path = 'C:\\Users\\dpati\\OneDrive\\Desktop\\datafinal\\extracted_images\\'
img_names = listdir(img_path)

fig, axes = plt.subplots(3, 3, figsize=(10, 10))
for ax, name in zip(axes.ravel(), img_names[:9]):
    img_file = listdir(img_path + name + '\\')[0]
    img = mpimg.imread(img_path + name + '\\' + img_file)
    ax.imshow(img, cmap='gray')
    ax.axis('off')
    ax.set_title(name)
plt.show()

fig, axes = plt.subplots(3, 3, figsize=(10, 10))
for ax, name in zip(axes.ravel(), img_names[:9]):
    img_file = listdir(img_path + name + '\\')[0]
    img = mpimg.imread(img_path + name + '\\' + img_file)
    new_img = process_image(img)
    ax.imshow(new_img, cmap='gray')
    ax.axis('off')
    ax.set_title(name)
plt.show()