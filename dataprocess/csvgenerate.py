import sys
from os import listdir
import numpy as np

def combine(names, tr_path, cv_path, save_path, shuffle=True):
    for path, ending in [(tr_path, '_tr.csv'), (cv_path, '_cv.csv')]:
        X = None
        for i, name in enumerate(names):
            # print progress
            sys.stdout.write('\r')
            sys.stdout.write('{:.2%}'.format(i/len(names)))
            sys.stdout.flush()

            curr = np.loadtxt(path + name + ending, delimiter=',')
            if X is None:
                X = curr.copy()
                continue
            X = np.vstack([X, curr])
        sys.stdout.write('\r100.00%\n')

        # shuffle
        if shuffle: np.random.shuffle(X)
        np.savetxt(save_path + 'all' + ending, X, delimiter=',', fmt='%i')

# img_path = 'C:\\Users\\dpati\\OneDrive\\Desktop\\datapart\\data\\extracted_images\\'
img_path = 'C:\\Users\\dpati\\OneDrive\\Desktop\\datafinal\\extracted_images\\'
img_names = listdir(img_path)

# save_path = 'C:\\Users\\dpati\\OneDrive\\Desktop\\datapart\\data\\savedata\\'
# tr_path = 'C:\\Users\\dpati\\OneDrive\\Desktop\\datapart\\data\\savedata\\tr\\'
# cv_path = 'C:\\Users\\dpati\\OneDrive\\Desktop\\datapart\\data\\savedata\\cv\\'

save_path = 'C:\\Users\\dpati\\OneDrive\\Desktop\\datafinal\\savedata\\'
tr_path = 'C:\\Users\\dpati\\OneDrive\\Desktop\\datafinal\\savedata\\tr\\'
cv_path = 'C:\\Users\\dpati\\OneDrive\\Desktop\\datafinal\\savedata\\cv\\'

combine(img_names, tr_path, cv_path, save_path)