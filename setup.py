import pickle
import numpy as np
from words import setImage
import cv2

path = r"./test/2.jpg"
# print "start"
image = cv2.imread(path)
# print "image"
data, res = setImage(image)
# print "load nn"
nn = pickle.load(open('nn.pkl', 'rb'))
# print "load res"
resultList = pickle.load(open('result.pkl', 'rb'))
# print "predict"
r = nn.predict(data)
# print r
print resultList[np.argmax(r)]
