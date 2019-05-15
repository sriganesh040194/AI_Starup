# USAGE
# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FileVideoStream
from imutils.video import FPS
from imutils.video import WebcamVideoStream
from skimage.measure import compare_ssim
import numpy as np
import argparse
import imutils
import time
import cv2
import os
prototxt ="MobileNetSSD_deploy.prototxt.txt"
model="MobileNetSSD_deploy.caffemodel"
videoFile="Sample01.mp4"
yolodir="yolo"
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", default=prototxt,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", default=model,
	help="path to Caffe pre-trained model")
ap.add_argument("-y", "--yolo", default=yolodir,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.7,
	help="minimum probability to filter weak detections")
ap.add_argument("-s", "--sample", default=videoFile,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())


# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
CLASSES = open(labelsPath).read().strip().split("\n")
 
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])


# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])
 
# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
#net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")

#abc= cv2.VideoCapture(args["sample"])
#abc= cv2.cvtColor(abc, cv2.COLOR_BGR2GRAY)
#vs = VideoStream(args["sample"]).start()
#vs= WebcamVideoStream(args["sample"]).start()
vs = VideoStream(src=0).start()
#vs = FileVideoStream(args["sample"]).start()
#vs.start()
#time.sleep(2.0)
fps = FPS().start()
old_frame = None
score=0

res=None
# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame = imutils.resize(frame, width=416, height=416)
	
	print("Frame Updated")

	#frame = type(FileVideoStream)(frame)
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
	#frame = imutils.rotate(frame,)
	#frame = np.transpose(frame)
	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	#blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
	#	0.007843, (300, 300), 127.5)
	#blob = cv2.dnn.blobFromImage(frame, 0.007843, (300,300), 127.5)
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),swapRB=True, crop=False)

	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward(ln)

	# loop over the detections
	#for i in np.arange(0, detections.shape[2]):
	#for i in detections:
		#print(i)
	for detection in detections:
		#print(detection)
		# extract the class ID and confidence (i.e., probability)
		# of the current object detection
		scores = detection[5:]
		#print(scores)

		classID = np.argmax(scores)
		#print(classID)
		confidence = scores[classID]
		#print(confidence)
		# extract the confidence (i.e., probability) associated with
		# the prediction
		#confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			print(confidence)
			# extract the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object
			#idx = int(i[0, 0, detection, 1])
			#box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			#print(detection)
			box = detection[0:4] * np.array([w, h, w, h])
			#(centerX, centerY, width, height) = box.astype("int")
			(startX, startY, endX, endY) = box.astype("int")
			# use the center (x, y)-coordinates to derive the top
			# and and left corner of the bounding box
			startX = int(startX - (endX / 2))
			startY = int(startY - (endY / 2))

			# draw the prediction on the frame
			label = "{}: {:.2f}%".format(CLASSES[classID], confidence * 100)
			print(label)
			color = [int(c) for c in COLORS[classID]]
			#cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
			cv2.rectangle(frame, (startX, startY), (endX+startX, endY+startY), color, 2)
			#cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

	#show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	#key='k'
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
	# update the FPS counter
	fps.update()
	print(fps._numFrames)
	#fps._numFrames+=20
    

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()