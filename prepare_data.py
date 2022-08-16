import os
from os.path import exists, sep, join
import shutil
from imutils.paths import list_images
from random import shuffle
from variables import val_split, trainPath, testPath

for label in next(os.walk('./own'))[1]:
    imagePaths = list(list_images(f'./own/{label}'))
    if len(imagePaths) == 0: continue
    shuffle(imagePaths)
    testPathsLen = int(len(imagePaths) * val_split)
    trainPathsLen = len(imagePaths) - testPathsLen
    trainPaths = imagePaths[:trainPathsLen]
    testPaths = imagePaths[trainPathsLen:]

    dstTrain = join(trainPath, label)
    if not exists(dstTrain): os.makedirs(dstTrain)
    for path in trainPaths:
        imageName = path.split(sep)[-1]
        shutil.copy(path, join(dstTrain, imageName))

    dstTest = join(testPath, label)
    if not exists(dstTest): os.makedirs(dstTest)
    for path in testPaths:
        imageName = path.split(sep)[-1]
        shutil.copy(path, join(dstTest, imageName))