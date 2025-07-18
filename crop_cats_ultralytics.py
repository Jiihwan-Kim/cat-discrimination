import os
import cv2
from ultralytics import YOLO
from pathlib import Path

# 모델 경로 (YOLOv11l 예시, 실제 모델 파일명에 맞게 수정)
MODEL_PATH = 'yolo11l.pt'  # 또는 yolov8l.pt 등
model = YOLO(MODEL_PATH)

DATASET_DIR = './datasets'
OUTPUT_DIR = './output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_EXTS = ['.jpg', '.jpeg', '.png', '.bmp']
VID_EXTS = ['.mp4', '.avi', '.mov', '.mkv']

def crop_and_save(img, box, save_path):
    x1, y1, x2, y2 = map(int, box)
    crop = img[y1:y2, x1:x2]
    cv2.imwrite(save_path, crop)

def process_image(img_path, out_dir, base_name):
    img = cv2.imread(img_path)
    results = model(img)
    for i, r in enumerate(results):
        for j, box in enumerate(r.boxes.xyxy.cpu().numpy()):
            cls = int(r.boxes.cls[j].cpu().numpy())
            # COCO 기준 고양이 class는 15번, 커스텀 모델이면 label 확인 필요
            if cls == 15 or r.names[cls].lower() == 'cat':
                save_path = os.path.join(out_dir, f"{base_name}_cat_{j}.jpg")
                crop_and_save(img, box, save_path)

def compute_iou(box1, box2):
    # box: [x1, y1, x2, y2]
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = interArea / float(box1Area + box2Area - interArea + 1e-6)
    return iou

def process_video(vid_path, out_dir, base_name):
    cap = cv2.VideoCapture(vid_path)
    frame_idx = 0
    prev_boxes = []
    IOU_THRESHOLD = 0.5
    FRAME_INTERVAL = 5

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_INTERVAL == 0:
            results = model(frame)
            for i, r in enumerate(results):
                for j, box in enumerate(r.boxes.xyxy.cpu().numpy()):
                    cls = int(r.boxes.cls[j].cpu().numpy())
                    if cls == 15 or r.names[cls].lower() == 'cat':
                        # 이전 박스와 IoU 비교
                        save = True
                        for prev_box in prev_boxes:
                            iou = compute_iou(box, prev_box)
                            if iou > IOU_THRESHOLD:
                                save = False
                                break
                        if save:
                            save_path = os.path.join(out_dir, f"{base_name}_frame{frame_idx}_cat_{j}.jpg")
                            crop_and_save(frame, box, save_path)
            # 현재 프레임의 박스들을 prev_boxes에 저장
            prev_boxes = [box for i, box in enumerate(r.boxes.xyxy.cpu().numpy()) if int(r.boxes.cls[i].cpu().numpy()) == 15 or r.names[int(r.boxes.cls[i].cpu().numpy())].lower() == 'cat']
        frame_idx += 1
    cap.release()

def main():
    for root, dirs, files in os.walk(DATASET_DIR):
        for file in files:
            ext = Path(file).suffix.lower()
            file_path = os.path.join(root, file)
            base_name = Path(file).stem
            if ext in IMG_EXTS:
                process_image(file_path, OUTPUT_DIR, base_name)
            elif ext in VID_EXTS:
                process_video(file_path, OUTPUT_DIR, base_name)

if __name__ == "__main__":
    main()
