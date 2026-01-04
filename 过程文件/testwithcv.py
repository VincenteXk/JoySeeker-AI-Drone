import cv2
import numpy as np
import socket
import threading
import time
from djitellopy import Tello

class TelloCV:
    def __init__(self):
        self.tello = Tello()
        self.frame = None
        self.running = False
        
    def connect(self):
        self.tello.connect()
        print(f"已连接！电池: {self.tello.get_battery()}%")

    def video_capture_thread(self):
        # 尝试不同解码选项
        cap = cv2.VideoCapture('udp://@0.0.0.0:11111', cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print("视频流已打开")
        self.running = True
        
        frame_count = 0
        while self.running:  # 限制帧数用于测试
            ret, self.frame = cap.read()
            if ret:
                cv2.imshow('Tello Video', self.frame)
                frame_count += 1
                print(f"成功解码帧: {frame_count}")
            else:
                print("解码失败")
                break
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
    def run_test(self):
        self.connect()
        
        print("开启视频流...")
        self.tello.streamon()
        time.sleep(2)
        
        video_thread = threading.Thread(target=self.video_capture_thread)
        video_thread.start()
        
        # 等待视频线程自然结束（按q键或达到100帧）
        video_thread.join()
        
        # 关闭视频流
        self.tello.streamoff()
        self.tello.end()

if __name__ == "__main__":
    tello_cv = TelloCV()
    tello_cv.run_test()