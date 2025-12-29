import cv2
from datetime import datetime
from ultralytics import YOLO

# Завдання 1
YOLO_CLASSES_UK = {
    'person': 'людина',
    'bicycle': 'велосипед',
    'car': 'автомобіль',
    'motorcycle': 'мотоцикл',
    'bus': 'автобус',
    'truck': 'вантажівка',
    'train': 'поїзд',
    'dog': 'собака',
    'cat': 'кіт',
    'bird': 'птах',
    'horse': 'кінь',
    'sheep': 'вівця',
    'cow': 'корова',
    'chair': 'стілець',
    'sofa': 'диван',
    'tv': 'телевізор',
    'laptop': 'ноутбук',
    'cell phone': 'телефон',
    'cup': 'чашка',
    'bottle': 'пляшка',
    'book': 'книга'
}

# Завдання 2
# Завантаження моделі
model = YOLO("yolov5s.pt")  # або yolov8s.pt для останньої версії
model.conf = 0.4  # поріг впевненості

cap = cv2.VideoCapture(0)
log_file = open("detections.txt", "a", encoding="utf-8")

print("Натисніть Q для виходу")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    detections = results.pandas().xyxy

    # Завдання 3
    object_counts = {}

    for _, row in detections.iterrows():
        class_name = row['name']
        confidence = row['confidence']
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])

        label_uk = YOLO_CLASSES_UK.get(class_name, class_name)

        object_counts[label_uk] = object_counts.get(label_uk, 0) + 1

        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"[{time_str}] Виявлено: {label_uk} ({confidence*100:.1f}%)\n")

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{label_uk} {confidence*100:.1f}%",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_COMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    y = 30
    for name, count in sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
        cv2.putText(
            frame,
            f"{name}: {count}",
            (10, y),
            cv2.FONT_HERSHEY_COMPLEX,
            0.7,
            (255, 255, 0),
            2
        )
        y += 30

    cv2.imshow("YOLOv5 Розпізнавання об'єктів", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

log_file.close()
cap.release()
cv2.destroyAllWindows()

