import os
import csv
import cv2
import numpy as np

#Loading CSV File
lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

name = './data/IMG/'+lines[0].split('/')[-1]
image = cv2.imread(name)
image = cv2.flip(image,1)
cv2.imshow()
