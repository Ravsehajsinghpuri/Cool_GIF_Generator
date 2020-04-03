# import necessary packages
from imutils import face_utils
from imutils import paths
import numpy as np 
import argparse
import imutils
import shutil
import json
import dlib
import cv2
import sys
import os

def overlay_image(bg,fg,fgMask,coords):
	(sH,sW)=fg.shape[:2]
	(x,y)=coords

	#size of overlay should be exactly same as that of the input image

	overlay=np.zeros(bg.shape,dtype="uint8")
	overlay[y:y+sH,x:x+sW]=fg # using array-slicing
	alpha =np.zeros(bg.shape[:2],dtype="uint8")
	alpha[y:y+sH,x:x+sW] =fgMask
	alpha=np.dstack([alpha]*3)
	output=alpha_blend(overlay,bg,alpha)
	return output

def alpha_blend(fg,bg,alpha):
	# alpha blending is applied on images with type "float" rather than int so we convert
	# each image into the required format
	fg=fg.astype("float")
	bg=bg.astype("float")
	alpha=alpha.astype("float")/255 # scaling alpha layer to the range [0,1]

	#performing alpha-blending
	fg=cv2.multiply(alpha,fg)
	bg=cv2.multiply(1-alpha,bg)

	output=cv2.add(fg,bg)
	return output.astype("uint8")

def create_gif(inputPath,outputPath,delay,finalDelay,loop):
	imagePaths=sorted(list(paths.list_images(inputPath)))
	lastPath=imagePaths[-1]
	imagePaths=imagePaths[:-1]
	cmd="convert -delay {} {} -delay {} {} -loop {} {}".format(delay, " ".join(imagePaths), finalDelay,lastPath,loop,outputPath)
	os.system(cmd)

ap=argparse.ArgumentParser()
ap.add_argument("-c", "--config",required=True, help="path to configuration file")
ap.add_argument("-i","--image",required=True, help="Path to input image")
ap.add_argument("-o","--output",required=True, help="Path to output GIF")
args=vars(ap.parse_args())


config=json.loads(open(args["config"]).read())  # Loading the configuration file and making a python directory for further use
                                                #in the program

sg=cv2.imread(config["sunglasses"])
sgMask=cv2.imread(config["sunglasses_mask"])

shutil.rmtree(config["temp_dir"],ignore_errors=True)
os.makedirs(config["temp_dir"])

print("[INFO] loading models.....")
# Detector model for face detection in an image
detector= cv2.dnn.readNetFromCaffe(config["face_detector_prototxt"],
            config["face_detector_weights"])

# Detector model for detecting landmarks in an input image

predictor= dlib.shape_predictor(config["landmark_predictor"]) # This predictor will help us predict the facial landmarks such as eyes,nose,mouth,jawline etc

# But we are interested in extracting just the eyes

# Detecting face in the input image

image=cv2.imread(args["image"]) # Reading input image from argument
(H,W) =image.shape[:2]
#  Creating a blob to send through the neural network for face detection
blob=cv2.dnn.blobFromImage(cv2.resize(image,(300,300)),1.0,(300,300),(104.0,177.0,123.0))

print("[INFO] computing object detections....")
detector.setInput(blob) # Sending the constructed blob to the neural network
detections= detector.forward()
i=np.argmax(detections[0,0,:,2])
confidence=detections[0,0,i,2]

# filtering out weak detections
if confidence<config["min_confidence"]:
	print("[INFO] no reliable faces found")
	sys.exit(0)

box=detections[0,0,i,3:7]*np.array([W,H,W,H])
(startX,startY,endX,endY)=box.astype("int")

rect=dlib.rectangle(int(startX),int(startY),int(endX),int(endY))
shape=predictor(image,rect)
shape=face_utils.shape_to_np(shape)

(lStart,lEnd)=face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart,rEnd)=face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
leftEyePts=shape[lStart:lEnd]
rightEyePts=shape[rStart:rEnd]

#computing center of mass for each eye

leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

#Computing the angle between the centroids of the eyes

dY=rightEyeCenter[1]-leftEyeCenter[1]
dX=rightEyeCenter[0]-leftEyeCenter[0]

angle=np.degrees(np.arctan2(dY,dX))-180

# Now we have the angle between the two eyes, So we align our sunglasses with the eyes by rotating the sunglasses by an angle
# which is equal to the angle between the eyes

# The sunglasses should not cover the whole face but rather 90% of the face so we do
# the following steps to obtain that
sg=imutils.rotate_bound(sg,angle)
sgW=int((endX-startX)*0.9)
sg=imutils.resize(sg,width=sgW)

sgMask = cv2.cvtColor(sgMask, cv2.COLOR_BGR2GRAY)
sgMask = cv2.threshold(sgMask,0,255,cv2.THRESH_BINARY)[1]
sgMask = imutils.rotate_bound(sgMask,angle)
sgMask = imutils.resize(sgMask,width=sgW, inter=cv2.INTER_NEAREST) 
# Here we're using nearest neighbour interpolation on our mask while resizing it


steps=np.linspace(0,rightEyeCenter[1],config["steps"], dtype="int")
for(i,y) in enumerate(steps):
	shiftX=int(sg.shape[1]*0.25)
	shiftY=int(sg.shape[0]*0.5)
	y=max(0,y-shiftY)
	output=overlay_image(image,sg,sgMask,(rightEyeCenter[0]-shiftX,y))

	if i==len(steps)-1:
		dwi=cv2.imread(config["deal_with_it"])
		dwiMask=cv2.imread(config["deal_with_it_mask"])
		dwiMask=cv2.cvtColor(dwiMask,cv2.COLOR_BGR2GRAY)
		dwiMask=cv2.threshold(dwiMask,0,255,cv2.THRESH_BINARY)[1]

		oW = int(W*0.8)
		dwi=imutils.resize(dwi,width=oW)
		dwiMask=imutils.resize(dwiMask,width=oW,inter=cv2.INTER_NEAREST)
		oX=int(W*0.1)
		oY=int(H*0.8)
		output=overlay_image(output,dwi,dwiMask,(oX,oY))
	p=os.path.sep.join([config["temp_dir"], "{}.jpg".format(str(i).zfill(8))])
	cv2.imwrite(p,output)

# Now, all of the frames are written to the disk and now we can finally create our output GIF image

print("[INFO] creating GIF...")
create_gif(config["temp_dir"],args["output"],config["delay"],config["final_delay"],config["loop"])

print("[INFO] cleaning up....")
shutil.rmtree(config["temp_dir"],ignore_errors=True)



























