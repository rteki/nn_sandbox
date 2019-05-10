import numpy as np
import glob
import itertools
import cv2




def getFilenames(exts):
    fnames = [glob.glob(ext) for ext in exts]
    fnames = list(itertools.chain.from_iterable(fnames))
    return fnames

dataset_x = np.array([]).reshape(57600, 0)
dataset_y = np.array([]).reshape(1, 0)

counter = 0

for image in getFilenames(["*.png"]):
    # img = cv2.imread(image)
    # img = cv2.resize(img, (128, 150))

    # img = np.array(img).reshape(img.shape[0]*img.shape[1]*img.shape[2], 1)
    
    # dataset_x = np.concatenate((dataset_x, img), axis=1)

    if "jd" in image:
        dataset_y = np.append(dataset_y, 1)
    else:
        dataset_y = np.append(dataset_y, 0)

    counter += 1
    print(counter)


print(dataset_x.shape)
print(dataset_y.shape)


# np.save("dataset_x", dataset_x)
np.save("dataset_y", dataset_y)