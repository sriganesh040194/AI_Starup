from imutils.video import VideoStream
from imutils.video import FileVideoStream
from imutils.video import FPS
from skimage.measure import compare_ssim
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import datetime
from pydarknet import Detector, Image
import Threshold

videoFile="HighlineGate.mp4"
yolodir="yolo"
threshold =  Threshold.Threshold(videoFile).get()

# parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-y", "--yolo", default=yolodir,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-s", "--sample", default=videoFile,
	help="minimum probability to filter weak detections")
ap.add_argument("-st", "--thresh", default=threshold,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())


# COCO class labels
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
CLASSES = open(labelsPath).read().strip().split("\n")
 
COLORS = np.random.uniform(20, 255, size=(len(CLASSES), 3))

print("[INFO] loading model...")

# YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])
cocodata = os.path.sep.join([args["yolo"], "coco.data"])
 
print("[INFO] loading YOLO from disk...")
net = Detector(bytes(configPath, encoding="utf-8"), bytes(weightsPath, encoding="utf-8"),0, bytes(cocodata, encoding="utf-8"))

print("[INFO] starting video stream...")

#vs = VideoStream(args["sample"]).start()
#vs = VideoStream(src=0).start()
vs = FileVideoStream(args["sample"]).start()

time.sleep(2.0)
fps = FPS().start()
old_frame = None
score=0
res=None

# loop frames
while True:
	#read the frame
	frame = vs.read()
	if frame is None:
		break
	
	#start timer
	startFrame = datetime.datetime.now()
	try:
		frame = imutils.resize(frame, width=416, height=416)
	except:
		fps.update()
		pass
	
	#To make sure that the frame is of 3 channels
	channel = frame.shape[2:]
	if channel[0] < 3:
		dim = np.zeros((234,416))
		frame = np.stack((frame,dim, dim), axis=2)
	#Frame differencing
	if old_frame is not None:
		try:
			(score, diff) = compare_ssim(old_frame, frame, multichannel=True, full=True)
		except:
			fps.update
			pass
		
	if score<args["thresh"]:
		old_frame =frame
		print("Frame Updated")
	
		img_darknet = Image(frame)
		result = net.detect(img_darknet)
		for cat, score, bounds in result:
			(startX, startY, endX, endY) = bounds
			startX = int(startX - (endX / 2))
			startY = int(startY - (endY / 2))

			#get catogory name
			cat = cat.decode("utf-8")
			
			#get label
			label = "{}: {:.2f}%".format(cat, score * 100)
			print(label)

			#get catogory ID to generate colors
			catID = CLASSES.index(cat)
			color = [int(c) for c in COLORS[catID]]
			
			# draw the prediction on the frame
			cv2.rectangle(frame, (startX, startY), (int (endX+startX), int(endY+startY)), color, 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
			endFrame = datetime.datetime.now()
			frameTime = (endFrame - startFrame).total_seconds()
			print("Time per frame : ",frameTime)
	
		# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

	else:
		endFrame = datetime.datetime.now()
		frameTime = (endFrame - startFrame).total_seconds()
		print("Time for frame skip : ",frameTime)
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