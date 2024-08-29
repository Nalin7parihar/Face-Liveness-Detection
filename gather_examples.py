import numpy as np
import argparse
import cv2
import os

# xây dựng phân tích tham số đầu vào từ dòng lệnh
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True, help="path to input video")
ap.add_argument("-o", "--output", type=str, required=True, help="path to output directory of cropped faces")
ap.add_argument("-d", "--detector", type=str, required=True, help="path to OpenCV's deep learning face detections")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-s", "--skip", type=int, default=16, help="# of frames to skip before applying face detection")
args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# open a pointer to the video file stream and initialize the total
# number of frames read and saved thus far
vs = cv2.VideoCapture(args["input"])
read = 0
saved = 0

# lặp qua các frames từ video file stream
while True:
    # Lấy frame từ file
    (grabbed, frame) = vs.read()
    
    # if the frame was not grabbed, then we have reached the end of the stream
    if not grabbed:
        break
    
    # tăng tổng số frame đã read cho đến hiện tại
    read += 1
    
    # kiểm tra xem liệu có cần sử lý frame này không
    if read % args["skip"] != 0:
        continue
    
    # lấy kích thước khung và tạo một đốm màu từ khung
    (h, w) = frame.shape[:2] # lấy chiều cao (h) và chiều rộng (w) của khung hình hiện tại
    # sử dụng phép trừ trung bình
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    # chuyển blob qua mạng và nhận được các phát hiện và dự đoán 
    net.setInput(blob) # đặt blob làm đầu vào cho mạng neural.
    detections = net.forward()
    # đảm bảo tìm thấy ít nhất một khuôn mặt
    if len(detections) > 0:
        # giả định rằng mỗi hình ảnh chỉ có một khuôn mặt, tìm bouding box có xác suất lớn nhất
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]
        # là thử nghiệm xác suất tối thiểu(do đó giúp lọc ra phát hiện yếu)
        if confidence > args["confidence"]:
            # tính toán tọa độ (x, y) của bouding box cho face và extract the face ROI
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = frame[startY:endY, startX:endX]
            # ghi frame vào disk
            p = os.path.sep.join([args["output"], "{}.png".format(saved)])
            cv2.imwrite(p, face)
            saved += 1
            print("[INFO] saved {} to disk".format(p))
            
vs.release()
cv2.destroyAllWindows()           