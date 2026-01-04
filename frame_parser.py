import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

EMOTION_CLASSES = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']
DEVICE = torch.device("cuda") # 没有cuda就改成cpu
MODEL_PATH = './resource/best_emotion_model.pth' 

### 算法类 ###

class StabilityFilter:
    """
    时间防抖：只有连续笑了一段时间，才认为是真笑。
    """
    def __init__(self, max_history=15, on_thresh=10, off_thresh=4):
        self.counter = 0
        self.max_history = max_history
        self.on_thresh = on_thresh
        self.off_thresh = off_thresh
        self.is_active = False

    def update(self, raw_detection_bool):
        if raw_detection_bool:
            self.counter += 1
        else:
            self.counter -= 1
        
        self.counter = max(0, min(self.max_history, self.counter))

        if self.counter >= self.on_thresh:
            self.is_active = True
        elif self.counter <= self.off_thresh:
            self.is_active = False
        
        return self.is_active, self.counter

class SpatialStabilizer:
    """
    空间防抖：过滤位置的剧烈跳变。
    """
    def __init__(self, distance_threshold=70, patience_frames=8, smooth_factor=0.7):
        self.dist_thresh = distance_threshold
        self.patience = patience_frames
        self.smooth_factor = smooth_factor
        
        self.last_confirmed_pos = None
        self.tentative_pos = None
        self.stable_counter = 0

    def update(self, current_centers):
        if not current_centers:
            self.tentative_pos = None
            self.stable_counter = 0
            return None
        # 求所有笑脸的总重心
        avg_x = sum(c[0] for c in current_centers) / len(current_centers)
        avg_y = sum(c[1] for c in current_centers) / len(current_centers)
        current_pos = np.array([avg_x, avg_y])

        if self.last_confirmed_pos is None:
            self.last_confirmed_pos = current_pos
            return tuple(self.last_confirmed_pos.astype(int))

        dist = np.linalg.norm(current_pos - self.last_confirmed_pos)

        # 如果上一帧的检测值异常，跟上上帧的作比较
        if dist > self.dist_thresh:
            if self.tentative_pos is None:
                self.tentative_pos = current_pos
                self.stable_counter = 1
            else:
                dist_tentative = np.linalg.norm(current_pos - self.tentative_pos)
                if dist_tentative < 30: 
                    self.stable_counter += 1
                    self.tentative_pos = current_pos 
                else:
                    self.tentative_pos = current_pos 
                    self.stable_counter = 1
            
            if self.stable_counter >= self.patience:
                self.last_confirmed_pos = current_pos
                self.tentative_pos = None
                self.stable_counter = 0
        
        # 如果上一帧的移动不异常，就取两帧的某个加权均值作为新坐标
        else:
            self.tentative_pos = None
            self.stable_counter = 0
            self.last_confirmed_pos = (self.last_confirmed_pos * (1 - self.smooth_factor)) + (current_pos * self.smooth_factor)

        return tuple(self.last_confirmed_pos.astype(int))
    
# --- 3. 辅助函数 ---

def load_emotion_model(model_path=MODEL_PATH):
    print(f"Loading model from {model_path}...")
    model = models.mobilenet_v3_large(weights=None)
    num_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(num_features, 512),
        nn.Hardswish(),
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.3),
        nn.Linear(512, len(EMOTION_CLASSES))
    )
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE).eval()
        print("Model loaded successfully.")
        return model
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Model file '{model_path}' not found.")
        return None

def crop_face_emulate_training(img_cv, x, y, w, h):
    rows, cols, _ = img_cv.shape
    center_x = x + w // 2
    center_y = y + h // 2
    target_w = int(w * 1.2)
    target_h = int(target_w / 0.67) 
    
    new_x = max(0, int(center_x - target_w // 2))
    new_y = max(0, int(center_y - target_h // 2))
    new_x2 = min(cols, new_x + target_w)
    new_y2 = min(rows, new_y + target_h)
    
    face_crop = img_cv[new_y:new_y2, new_x:new_x2]
    fh, fw, _ = face_crop.shape
    if fh < 1 or fw < 1: return img_cv[y:y+h, x:x+w]

    if fh < target_h or fw < target_w:
        pad_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        start_x = max(0, (target_w - fw) // 2)
        start_y = max(0, (target_h - fh) // 2)
        pad_img[start_y:start_y+fh, start_x:start_x+fw] = face_crop
        return pad_img
    return face_crop

def calculate_rotate_speed(delta_x):
    if abs(delta_x) > 0.15:
        if delta_x > 0:
            speed = delta_x * 25 + 20
        else:
            speed = delta_x * 25 - 20
    else:
        speed = 0
    return speed

### 主类 ###

class frame_parser():
    def __init__(self, model_path=MODEL_PATH):
        self.model = load_emotion_model(model_path)

        self.face_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        self.temporal_filter = StabilityFilter(max_history=10, on_thresh=7, off_thresh=3)
        self.spatial_filter = SpatialStabilizer(distance_threshold=140, patience_frames=4, smooth_factor=0.8)
        print('\n视觉模块初始化成功')

    def parse(self, frame):
        if frame is None: return frame, None

        frame = cv2.resize(frame, (640, 480))
        fh, fw = frame.shape[:2]

        # 检测人脸，返回bounding boxes
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # 遍历人脸进行微笑打分
        max_smile_score = 0
        smiling_faces_rects = [] 
        for (x, y, w, h) in faces:
            # 切片出人脸图像
            processed_face = crop_face_emulate_training(frame, x, y, w, h)
            face_pil = Image.fromarray(cv2.cvtColor(processed_face, cv2.COLOR_BGR2RGB))
            input_tensor = self.face_transform(face_pil).unsqueeze(0).to(DEVICE)
            # 预测微笑概率
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                score = probs[0][3].item() 

            # 更新最大笑容分，用来时间防抖（当然当前算法是有问题的）
            max_smile_score = max(max_smile_score, score)
            
            # 阈值0.35，框就是绿的
            color = (0, 255, 0) if score > 0.35 else (255, 0, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            cv2.putText(frame, f"{score:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if score > 0.35:
                smiling_faces_rects.append((x, y, w, h))

        raw_smile_detected = (max_smile_score > 0.5)
        is_stable_smiling, energy = self.temporal_filter.update(raw_smile_detected)

        final_target_point = None
        
        if is_stable_smiling:
            centers = [(x + w//2, y + h//2) for (x, y, w, h) in smiling_faces_rects]
            final_target_point = self.spatial_filter.update(centers)
            
            if final_target_point:
                tx, ty = final_target_point
                cv2.drawMarker(frame, (fw//2, ty), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 10, 3)
                cv2.putText(frame, "CONFIRMED SMILING", (tx+10, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                # UI: 目标指示线
                cv2.line(frame, (tx, ty), (fw//2, ty), (255, 255, 0), 2)
                delta_x = tx/fw-0.5
                rotate = calculate_rotate_speed(delta_x)
                return frame, rotate
            else:
                return frame, None
        else:
            self.spatial_filter.update([])
            return frame, None
        