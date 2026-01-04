import time
import cv2
import numpy as np


# 设置尺寸（适应2560x1600屏幕）
VIDEO_WIDTH = 640   # 视频宽度
VIDEO_HEIGHT = 480  # 视频高度
SIDEBAR_WIDTH = 200  # 状态栏宽度
RIGHT_WIDTH = 640    # 截图区宽度


class UI():
    def __init__(self):
        self.screenshot = None
        self.last_screenshot_time = time.time()
        print('\n用户界面初始化成功')

    def take_screenshot(self, frame, x_span, y_span):
        t = time.time()
        if t-self.last_screenshot_time>3:
            h, w = frame.shape[:2]
            x0, x1 = x_span
            y0, y1 = y_span
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.screenshot = frame[int(x0*h):int(x1*h), int(y0*w):int(y1*w)]
            self.last_screenshot_time = t

    def show(self, frame, tello_status):
        if frame is not None:
            # 颜色转换
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 调整视频尺寸
            frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
            
            # 创建状态边栏
            sidebar = np.zeros((VIDEO_HEIGHT, SIDEBAR_WIDTH, 3), dtype=np.uint8)
            
            # 从字典中提取数据
            battery = tello_status.get('battery', 0)
            flight_time = tello_status.get('flight_time', 0)
            height_val = tello_status.get('height_val', 0)
            tof = tello_status.get('tof', 0)
            barometer = tello_status.get('barometer', 0)
            temp_low = tello_status.get('temp_low', 0)
            temp_high = tello_status.get('temp_high', 0)
            temp_avg = tello_status.get('temp_avg', 0)
            pitch = tello_status.get('pitch', 0)
            roll = tello_status.get('roll', 0)
            yaw = tello_status.get('yaw', 0)
            speed_x = tello_status.get('speed_x', 0)
            speed_y = tello_status.get('speed_y', 0)
            speed_z = tello_status.get('speed_z', 0)
            acc_x = tello_status.get('acc_x', 0.0)
            acc_y = tello_status.get('acc_y', 0.0)
            acc_z = tello_status.get('acc_z', 0.0)
            # 在边栏上绘制状态文本
            y_offset = 40
            line_height = 20
            
            def add_text(text, y, color=(0, 255, 0)):
                cv2.putText(sidebar, text, (10, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # 标题
            cv2.putText(sidebar, "TELLO STATUS", (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            # 基本状态（关键信息）
            add_text(f"Battery: {battery}%", y_offset, (0, 255, 0) if battery > 20 else (0, 0, 255))
            add_text(f"Flight: {flight_time}s", y_offset + line_height)
            add_text(f"Height: {height_val}cm", y_offset + line_height*2)
            add_text(f"TOF: {tof}cm", y_offset + line_height*3)
            
            # 温度
            add_text(f"Temp: {temp_avg:.1f}C", y_offset + line_height*4)
            add_text(f"Low: {temp_low}C", y_offset + line_height*5, (100, 100, 255))
            add_text(f"High: {temp_high}C", y_offset + line_height*6, (100, 100, 255))
            
            # 姿态
            add_text(f"Pitch: {pitch}deg", y_offset + line_height*7)
            add_text(f"Roll: {roll}deg", y_offset + line_height*8)
            add_text(f"Yaw: {yaw}deg", y_offset + line_height*9)
            
            # 速度（简洁显示）
            add_text(f"Speed X: {speed_x}", y_offset + line_height*10)
            add_text(f"Speed Y: {speed_y}", y_offset + line_height*11)
            add_text(f"Speed Z: {speed_z}", y_offset + line_height*12)
            
            # 加速度
            add_text(f"Acc X: {acc_x:.1f}", y_offset + line_height*13)
            add_text(f"Acc Y: {acc_y:.1f}", y_offset + line_height*14)
            add_text(f"Acc Z: {acc_z:.1f}", y_offset + line_height*15)
            
            # 右边截图窗格
            right_pane = np.zeros((VIDEO_HEIGHT, RIGHT_WIDTH, 3), dtype=np.uint8)
            if self.screenshot is not None:
                right_pane = cv2.resize(self.screenshot, (RIGHT_WIDTH, VIDEO_HEIGHT))
                # 添加"Screenshot"标签
                cv2.putText(right_pane, "Screenshot", (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 合并三个部分
            combined = np.hstack((sidebar, frame, right_pane))
            
            # 添加分隔线
            cv2.line(combined, (SIDEBAR_WIDTH, 0), (SIDEBAR_WIDTH, VIDEO_HEIGHT), (0, 255, 0), 1)
            cv2.line(combined, (SIDEBAR_WIDTH+VIDEO_WIDTH, 0), (SIDEBAR_WIDTH+VIDEO_WIDTH, VIDEO_HEIGHT), (0, 255, 0), 1)
            
            # 显示窗口标题
            cv2.putText(combined, "Status", (10, VIDEO_HEIGHT-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 255), 1)
            cv2.putText(combined, "Live Video", (SIDEBAR_WIDTH + 10, VIDEO_HEIGHT-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 255), 1)
            
            # 显示
            cv2.imshow('Tello Dashboard', combined)
        

# cv2.destroyAllWindows()