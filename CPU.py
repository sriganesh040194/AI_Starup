# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FileVideoStream
from imutils.video import FPS
from imutils.video import WebcamVideoStream
from skimage.measure import compare_ssim
import datetime
import numpy as np
import argparse
import imutils
import time
import cv2
import os

videoFile="HighlineGate.mp4"
yolodir="yolo"

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-y", "--yolo", default=yolodir,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.7,
	help="minimum probability to filter weak detections")
ap.add_argument("-s", "--sample", default=videoFile,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())


# load the COCO class labels
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
CLASSES = open(labelsPath).read().strip().split("\n")
 
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("[INFO] loading model...")

# YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])
 
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
ln1 = net.getLayerNames()

print("[INFO] starting video stream...")

#Input streams
#vs = VideoStream(args["sample"]).start()
#vs= WebcamVideoStream(args["sample"]).start()
#vs = VideoStream(src=0).start()
vs = FileVideoStream(args["sample"]).start()

#time.sleep(2.0)
fps = FPS().start()
old_frame = None
score=0

res=None
# loop over the frames from the video stream
while True:
	# grab the frame
	startFrame = datetime.datetime.now()
	frame = vs.read()
	if frame is None:
		break
	frame = imutils.resize(frame, width=416, height=416)
	ln = [ln1[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	# convert it to a blob
	channel = frame.shape[2:]
	(h, w) = frame.shape[:2]

	#To make sure that the frame is of 3 channels
	if channel[0] < 3:
		dim = np.zeros((234,416))
		frame = np.stack((frame,dim, dim), axis=2)
	#frame_cm = cv2.applyColorMap(frame,cv2.COLORMAP_WINTER)
	#frame = frame_cm.astype('uint8')
	#frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB)
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),swapRB=True, crop=False)

	# predictions
	net.setInput(blob)
	detections = net.forward(ln)

	# loop over the detections
	for i in detections:
		for detection in i:
			scores = detection[5:]
			classID = np.argmax(scores)
			# extract the confidence
			confidence = scores[classID]

			# check confidence is greater than the minimum confidence
			if confidence > args["confidence"]:
				# the bounding box for the object
				box = detection[0:4] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				startX = int(startX - (endX / 2))
				startY = int(startY - (endY / 2))

				# draw the prediction on the frame
				label = "{}: {:.2f}%".format(CLASSES[classID], confidence * 100)
				print(label)
				color = [int(c) for c in COLORS[classID]]
				cv2.rectangle(frame, (startX, startY), (endX+startX, endY+startY), color, 2)
				y = startY - 15 if startY - 15 > 15 else startY + 15
				cv2.putText(frame, label, (startX, y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

				endFrame = datetime.datetime.now()
				frameTime = (endFrame - startFrame).total_seconds()
				print("Time per frame : ",frameTime)

		#show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
