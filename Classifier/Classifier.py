import openface
import numpy as np 
import cv2
import pandas as pd 
from sklearn.svm import SVC

a='/home/arshad/openface/models/dlib/shape_predictor_68_face_landmarks.dat'
b='/home/arshad/openface/models/openface/nn4.small2.v1.t7'


Features=[]
Target=[]

def FnT(Dataset):
	
	for i in range(0,13):
    		temp=[]
    		for j in Dataset.iloc[i]:
        
        		temp.append(j)
    		Features.append(temp[:-1])
    		Target.append(temp[-1])

def findrep(img):
	Img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	
	align = openface.AlignDlib(a)

	net = openface.TorchNeuralNet(b, imgDim=96)

	bb = align.getLargestFaceBoundingBox(Img)

	alignedFace = align.align(96, Img, bb,landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

	rep = net.forward(alignedFace)	
	
	return rep

D0=pd.read_csv(r'/home/arshad/Classifier/Features0.csv')
D1=pd.read_csv(r'/home/arshad/Classifier/Features1.csv')

FnT(D0)
FnT(D1)

clf = SVC(C=1, kernel='linear', probability=True)
clf.fit(Features,Target)

Image=cv2.imread(r'/home/arshad/Classifier/Data/Test_2.jpg')
REP=findrep(Image)
REP=np.reshape(REP,(1,-1))
predictions = clf.predict_proba(REP).ravel()
maxI = np.argmax(predictions)
confidence = predictions[maxI]

if maxI==0:
	print('Test Image is predicted to be Mark Wahlberg with %f confidence.'%confidence)

if maxI==1:
	print('Test Image is predicted to be Matt Damon with %f confidence.'%confidence)






