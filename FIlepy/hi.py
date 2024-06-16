import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import cv2
import numpy as np
import time
import json
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import matplotlib.pyplot as plt  # matplotlib 임포트

# COCO 클래스 레이블을 coco.names 파일에서 읽어옵니다.
with open('coco.names', 'r') as f:
    COCO_LABELS = [line.strip() for line in f.readlines()]

# 물체의 가격 정보를 저장하는 딕셔너리 파일의 경로를 지정합니다.
PRICE_INFO_FILE = 'price_info.json'

# 가격 정보를 로드하는 함수를 정의합니다.
def load_price_info(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# 가격 정보를 저장하는 함수를 정의합니다.
def save_price_info(file_path, price_info):
    with open(file_path, 'w') as f:
        json.dump(price_info, f, indent=4)

# 가격 정보를 로드합니다.
PRICE_INFO = load_price_info(PRICE_INFO_FILE)

# YOLOv4 모델과 구성 파일을 로드합니다.
net = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg')
layer_names = net.getLayerNames()
output_layer_indexes = net.getUnconnectedOutLayers()
output_layer_indexes = [[327], [353], [379]]
output_layers = [layer_names[i[0] - 1] for i in output_layer_indexes]

# 이메일 알림 설정
EMAIL_USER = 'USER.ID@gmail.com'  # 실제 이메일 주소
EMAIL_PASSWORD = 'PassWord'  # 실제 이메일 비밀번호
EMAIL_RECEIVER = 'USER.ID@gmail.com'  # 수신자 이메일 주소

# 알림 빈도를 제한하기 위한 딕셔너리
last_alert_time = defaultdict(lambda: datetime.min)
alert_interval = timedelta(minutes=5)  # 5분 간격으로 알림 제한

# 실시간 객체 감지 통계를 위한 카운터
detected_objects_counter = Counter()

# 이메일 알림을 보내는 함수 정의
def send_email_alert(label, confidence, frame):
    subject = f"Alert: {label} Detected"
    body = f"A {label} was detected with a confidence of {confidence:.2f}."
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = EMAIL_USER
    msg['To'] = EMAIL_RECEIVER

    # 텍스트 부분 추가
    msg.attach(MIMEText(body, 'plain'))

    # 이미지 부분 추가
    _, buffer = cv2.imencode('.jpg', frame)
    image_attachment = MIMEImage(buffer.tobytes())
    image_attachment.add_header('Content-Disposition', 'attachment', filename="detection.jpg")
    msg.attach(image_attachment)

    try:
        # 이메일 서버에 연결하고 이메일 전송
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:  # Gmail의 경우
            server.login(EMAIL_USER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_USER, EMAIL_RECEIVER, msg.as_string())
        print(f"Alert sent for {label}")
    except Exception as e:
        print(f"Failed to send email: {e}")

# 객체 감지를 수행하는 함수 정의
def detect_objects(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    # 감지된 객체의 정보를 저장
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    detected_objects = []
    if len(indices) > 0:
        for i in indices.flatten():
            detected_objects.append((boxes[i], class_ids[i], confidences[i]))

    # 감지된 객체 정보 처리 및 이메일 알림 전송
    for (box, class_id, confidence) in detected_objects:
        current_time = datetime.now()
        label = COCO_LABELS[class_id]
        detected_objects_counter[label] += 1  # 카운터 업데이트

        if label == 'cell phone':  # 예를 들어, 스마트폰이 감지되었을 때 알림
            if current_time - last_alert_time[label] > alert_interval:
                send_email_alert(label, confidence, frame)
                last_alert_time[label] = current_time

    return detected_objects

# 객체 정보를 화면에 표시하는 함수 정의
def display_objects(frame, detections):
    for (box, class_id, confidence) in detections:
        x, y, w, h = box
        label = str(COCO_LABELS[class_id])
        price = PRICE_INFO.get(label, 'Unknown')
        if price == 'Unknown':
            price = input(f"Enter price for {label}: ")
            PRICE_INFO[label] = price
            save_price_info(PRICE_INFO_FILE, PRICE_INFO)
        label += f' - ${price}'
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

# 실시간 통계 정보를 그래프로 시각화하는 함수 정의
def plot_realtime_stats(counter):
    labels = list(counter.keys())
    values = list(counter.values())

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color='blue')
    plt.xlabel('Object Labels')
    plt.ylabel('Counts')
    plt.title('Real-time Object Detection Statistics')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(True)

    plt.pause(0.1)  # Pause to allow the plot to update

    # Clear plot for the next update
    plt.clf()

# 메인 함수 정의
def main():
    cap = cv2.VideoCapture(0)
    total_objects = 0
    correctly_detected = 0

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from camera. Exiting...")
            break

        detections = detect_objects(frame)
        total_objects += len(detections)
        frame_with_detections = display_objects(frame, detections)

        cv2.imshow('Object Detection with Price', frame_with_detections)

        # 실시간 객체 통계 시각화
        plot_realtime_stats(detected_objects_counter)

        end_time = time.time()
        fps = 1 / (end_time - start_time)
        print(f"FPS: {fps}")

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if total_objects > 0:
        accuracy = (correctly_detected / total_objects) * 100
        print(f"Total objects detected: {total_objects}")

if __name__ == '__main__':
    main()
