# USAGE
# python yolo_video.py --input videos/airport.mp4 --output output/airport_output.avi --yolo yolo-coco

# Khai báo thư viện
#numpy
import numpy as np
import argparse
import imutils
import time
#opencv
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", default="videos/airport.mp4",
	help="path to input video")
ap.add_argument("-o", "--output", default="output/airport_output.avi",
	help="path to output video")
ap.add_argument("-y", "--yolo", default="yolo-coco",
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO model đã được training
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# Khởi tạo list màu sắc của đường bao sẽ sử dụng
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

#Load weight của model
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# Load coco model đã được training (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Khởi tạo luồng đọc video
# frame dimensions
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

#Tính tổng số frame trong video 
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

i = 0
# Lặp từng frame của video
while True:
	# Đọc từng ảnh vào 
	(grabbed, frame) = vs.read()
	

	#nếu khung không được gỡ bỏ, thì chúng đã kết thúc kết thúc
	# of the stream
	if not grabbed:
		break

	# kích cỡ khung này bị trống, lấy chúng
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# xây dựng một blob từ khung đầu vào và sau đó thực hiện chuyển tiếp
	# vượt qua trình phát hiện đối tượng YOLO, cung cấp cho chúng tôi các hộp giới hạn của chúng tôi
	# và xác suất liên quan
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	# khởi tạo danh sách các hộp giới hạn được phát hiện, tâm sự,
	# và ID lớp, tương ứng
	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# trích xuất ID lớp và độ tin cậy (nghĩa là xác suất)
			# của phát hiện đối tượng hiện tại
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# lọc ra các dự đoán yếu bằng cách đảm bảo phát hiện
			# xác suất lớn hơn xác suất tối thiểu
			if confidence > args["confidence"]:
				# quy mô tọa độ hộp giới hạn trở lại so với
				# kích thước của hình ảnh, hãy nhớ rằng YOLO
				# thực sự trả về trung tâm (x, y) -cordord của
				# hộp giới hạn theo sau là chiều rộng của hộp và
				# Chiều cao
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# sử dụng trung tâm (x, y) để phối hợp hàng đầu
				# và góc trái của hộp giới hạn
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# cập nhật danh sách tọa độ hộp giới hạn của chúng tôi,
				# tâm sự và ID lớp
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	# áp dụng triệt tiêu không cực đại để triệt tiêu yếu, chồng chéo
	# hộp giới hạn
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
		args["threshold"])

	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# giải nén tọa độ hộp giới hạn
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			# vẽ một hình chữ nhật hộp giới hạn và nhãn trên khung
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]],
				confidences[i])
			cv2.putText(frame, text, (x, y - 5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
	
	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

		# một số thông tin về xử lý khung đơn
		if total > 0:
			elap = (end - start)
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time to finish: {:.4f}".format(
				elap * total))

	# write the output frame to disk
	writer.write(frame)
	print(i)
	i = i +  1

#cv2.waitKey(0)
# release the file pointers
print("[INFO] cleaning up...")
#writer.release()
vs.release()
cv2.destroyAllWindows()
