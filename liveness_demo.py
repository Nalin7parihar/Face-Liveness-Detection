from imutils.video import VideoStream
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os

# xây dựng phân tích tham số đầu vào từ dòng lệnh
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True, help="path to trained model")
ap.add_argument("-l", "--le", type=str, required=True, help="path to label encoder")
ap.add_argument("-d", "--detector", type=str, required=True, help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())


# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load the liveness detector model and label encoder from disk
print("[INFO] loading liveness detector...")
model = load_model(args["model"])
le = pickle.loads(open(args["le"], "rb").read())
# initialize the video stream and allow the camera laptop to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)


# loop over the frames from the video stream
while True:
	# lấy frame từ video stream resize kích thước của nó sao cho chiều rộng tối đa là 600 pixel
	frame = vs.read()
	frame = imutils.resize(frame, width=600)
	# lấy kích thước của frame sử dụng thuật toán trừ trung bình
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))
	# chuyển blob qua mạng và nhận được các phát hiện và dự đoán
	net.setInput(blob) # đặt blob làm đầu vào cho mạng neural.
	detections = net.forward()
    
    # loop over the detections
	for i in range(0, detections.shape[2]):
		# extract độ tin cậy (i.e., probability) liên quan đến dự đoán
		confidence = detections[0, 0, i, 2]
		# loại bỏ các confidence yếu
		if confidence > args["confidence"]:
			# tính toán tọa độ (x, y) của bouding box cho face và extract the face ROI
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
   
			# đảm bảo rằng bouding box được phát hiện không nằm ngoài kích thước của frame
			startX = max(0, startX)
			startY = max(0, startY)
			endX = min(w, endX)
			endY = min(h, endY)
   
			# trích xuất vùng quan tâm của khuôn mặt (ROI) và sau đó 
            # tiền xử lý nó theo cách chính xác như dữ liệu training"
			face = frame[startY:endY, startX:endX]
			face = cv2.resize(face, (32, 32))
			face = face.astype("float") / 255.0
			face = img_to_array(face)
			face = np.expand_dims(face, axis=0)
   
            # truyền vùng quan tâm của khuôn mặt (ROI) qua mô hình liveness 
            # đã được huấn luyện để xác định xem khuôn mặt là 'thật' hay 'giả
			preds = model.predict(face)[0]
			j = np.argmax(preds)
			label = le.classes_[j]
   
			# draw the label and bounding box on the frame
			label = "{}: {:.4f}".format(label, preds[j])
			cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
			cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
   
    # show the output frame and wait for a key press
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()