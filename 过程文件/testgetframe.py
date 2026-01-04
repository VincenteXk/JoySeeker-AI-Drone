from djitellopy import Tello
import cv2
import time
import socket
import numpy as np



# 套接字修复
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.close()

# 连接Tello
tello = Tello()
tello.connect()

# 开启视频流
tello.streamon()
time.sleep(2)

# 初始化
frame_reader = tello.get_frame_read()
screenshot = None
fps_start_time = time.time()
frame_counter = 0
current_fps = 0

# 设置尺寸（适应2560x1600屏幕）
VIDEO_WIDTH = 640   # 视频宽度
VIDEO_HEIGHT = 480  # 视频高度
SIDEBAR_WIDTH = 200  # 状态栏宽度
RIGHT_WIDTH = 640    # 截图区宽度


while True:
    frame = frame_reader.frame
    if frame is not None:
        # 颜色转换
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # 调整视频尺寸
        frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
        
        # FPS计算
        frame_counter += 1
        if frame_counter >= 10:
            fps_end_time = time.time()
            current_fps = frame_counter / (fps_end_time - fps_start_time)
            frame_counter = 0
            fps_start_time = time.time()
        
        # 创建状态边栏
        sidebar = np.zeros((VIDEO_HEIGHT, SIDEBAR_WIDTH, 3), dtype=np.uint8)
        
        # 获取状态数据
        try:
            battery = tello.get_battery()
            flight_time = tello.get_flight_time()
            height_val = tello.get_height()
            tof = tello.get_distance_tof()
            barometer = tello.get_barometer()
            temp_low = tello.get_lowest_temperature()
            temp_high = tello.get_highest_temperature()
            temp_avg = (temp_low + temp_high) / 2
            pitch = tello.get_pitch()
            roll = tello.get_roll()
            yaw = tello.get_yaw()
            speed_x = tello.get_speed_x()
            speed_y = tello.get_speed_y()
            speed_z = tello.get_speed_z()
            acc_x = tello.get_acceleration_x()
            acc_y = tello.get_acceleration_y()
            acc_z = tello.get_acceleration_z()
        except:
            battery = flight_time = height_val = tof = barometer = 0
            temp_low = temp_high = temp_avg = 0
            pitch = roll = yaw = speed_x = speed_y = speed_z = 0
            acc_x = acc_y = acc_z = 0.0
        
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
        if screenshot is not None:
            right_pane = cv2.resize(screenshot, (RIGHT_WIDTH, VIDEO_HEIGHT))
            # 添加"Screenshot"标签
            cv2.putText(right_pane, "Screenshot", (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 合并三个部分
        combined = np.hstack((sidebar, frame, right_pane))
        
        # 添加分隔线
        cv2.line(combined, (SIDEBAR_WIDTH, 0), (SIDEBAR_WIDTH, VIDEO_HEIGHT), (0, 255, 0), 1)
        cv2.line(combined, (SIDEBAR_WIDTH+VIDEO_WIDTH, 0), (SIDEBAR_WIDTH+VIDEO_WIDTH, VIDEO_HEIGHT), (0, 255, 0), 1)
        
        # 显示FPS
        fps_text = f'FPS: {current_fps:.1f}'
        cv2.putText(combined, fps_text, (SIDEBAR_WIDTH + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        # 显示窗口标题
        cv2.putText(combined, "Status", (10, VIDEO_HEIGHT-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 255), 1)
        cv2.putText(combined, "Live Video", (SIDEBAR_WIDTH + 10, VIDEO_HEIGHT-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 255), 1)
        
        # 显示
        cv2.imshow('Tello Dashboard', combined)
    
    # 按键处理
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        if frame is not None:
            screenshot = frame.copy()
            print("Screenshot saved")
    elif key == ord('c'):
        screenshot = None
        print("Screenshot cleared")

# 清理
tello.streamoff()
cv2.destroyAllWindows()