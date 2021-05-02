import numpy as np
import sys
from os import listdir
from matplotlib import pyplot as plt
from random import shuffle
import matplotlib.image as mpimg
import cv2
import csv


def process_image(img, close_kernel=None, dilate_kernel=None):
    if close_kernel is None: close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    if dilate_kernel is None: dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    new_img = img < 127
    new_img = new_img.astype(np.uint8)
    new_img = cv2.morphologyEx(new_img, cv2.MORPH_CLOSE, close_kernel)
    new_img = cv2.dilate(new_img, dilate_kernel)
    return new_img


def process_data(names, latex_key, img_path, save_path, close_kernel=None, dilate_kernel=None, tr_cv_split=80):
    if close_kernel is None: close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    if dilate_kernel is None: dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    key = {}
    for k, name in enumerate(names):
        sys.stdout.write(name + "\n")
        # create key
        if name in latex_key:
            key[k] = latex_key[name]
        else:
            key[k] = name.lower()

        # get files of images
        imgs = []
        img_files = listdir(img_path + name + '\\')
        shuffle(img_files)

        for i, img_file in enumerate(img_files):
            img = mpimg.imread(img_path + name + '\\' + img_file)

            # convert to binary and close + dilate
            img = process_image(img, close_kernel, dilate_kernel)

            # add label
            imgs.append(np.append(img.ravel(), k))

            # print progress
            sys.stdout.write('\r')
            sys.stdout.write('{:.2%}'.format(i / len(img_files)))
            sys.stdout.flush()
        sys.stdout.write('\r100.00%\n')

        # split into training and validation sets and save csv
        arr = np.asarray(imgs)
        ind = len(img_files) * tr_cv_split // 100
        np.savetxt(save_path + 'tr\\' + name + '_tr' + '.csv', arr[:ind], delimiter=',', fmt='%i')
        np.savetxt(save_path + 'cv\\' + name + '_cv' + '.csv', arr[ind:], delimiter=',', fmt='%i')

    # save key
    if save_path:
        with open(save_path + 'dict.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            for k, v in key.items():
                writer.writerow([k, v])
    return key


latex_key = {
    '(': '\\left(',
    ')': '\\right)',
    'alpha': '\\alpha',
    'ascii_124': '|',
    'beta': '\\beta',
    'cos': '\\cos',
    'Delta': '\\Delta',
    'div': '\\div',
    'exists': '\\exists',
    'forall': '\\forall',
    'forward_slash': '\\',
    'gama': '\\gamma',
    'geq': '\\geq',
    'gt': '>',
    'infty': '\\infty',
    'int': '\\int',
    'in': '\\in',
    'lambda': '\\lambda',
    'ldots': '\\ldots',
    'leq': '\\leq',
    'lim': '\\lim',
    'log': '\\log',
    'lt': '<',
    'mu': '\\mu',
    'neq': '\\neq',
    'phi': '\\phi',
    'pi': '\\pi',
    'pm': '\\pm',
    'prime': '\'',
    'rightarrow': '\\rightarrow',
    'sigma': '\\sigma',
    'sin': '\\sin',
    'sqrt': '\\sqrt',
    'sum': '\\sum',
    'tan': '\\tan',
    'theta': '\\theta',
    'times': '\\times',
    '[': '\\left[',
    ']': '\\right]',
    '{': '\\left{',
    '}': '\\right}'
}

# convert images to csv
# img_path = 'C:\\Users\\dpati\\OneDrive\\Desktop\\datapart\\data\\extracted_images\\'
img_path = 'C:\\Users\\dpati\\OneDrive\\Desktop\\datafinal\\extracted_images\\'
img_names = listdir(img_path)
# save_path = 'C:\\Users\\dpati\\OneDrive\\Desktop\\datapart\\data\\savedata\\'
save_path = 'C:\\Users\\dpati\\OneDrive\\Desktop\\datafinal\\savedata\\'
key = process_data(img_names, latex_key, img_path, save_path)


