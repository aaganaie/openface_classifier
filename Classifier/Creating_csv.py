import openface
import numpy as np 
import cv2
import os 
import csv
a='/home/arshad/openface/models/dlib/shape_predictor_68_face_landmarks.dat'
b='/home/arshad/openface/models/openface/nn4.small2.v1.t7'


Features=[]


def findrep(img):
	Img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	
	align = openface.AlignDlib(a)

	net = openface.TorchNeuralNet(b, imgDim=96)

	bb = align.getLargestFaceBoundingBox(Img)

	alignedFace = align.align(96, Img, bb,landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

	rep = net.forward(alignedFace)	
	
	return rep

def train(path,targ):
	
	for root,dirn,filen in os.walk(path):
		
		for name in filen:
			Path=os.path.join(root,name)
			Image=cv2.imread(Path)
			rep=findrep(Image)
			rept=np.append(rep,targ)
			Features.append(rept)
			


path1='/home/arshad/Classifier/Data/Train/mark'
path2='/home/arshad/Classifier/Data/Train/matt'

train(path1,0)
train(path2,1)


with open ("Features.csv",'w')as file:
    writer=csv.writer(file,delimiter=',')
    for data in Features :
        writer.writerow(data)



			


	
		
	
