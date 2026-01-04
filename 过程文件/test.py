import av
import numpy as np
import cv2
from djitellopy import Tello
import time

# 连接Tello
tello = Tello()
tello.connect()
print(f"已连接！电池: {tello.get_battery()}%")

# 开启视频流
tello.streamon()
time.sleep(2)

# 设置套接字重用
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.close()

# 使用av解码
try:
    container = av.open('udp://0.0.0.0:11111')
    print("AV视频流已打开")
    
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format='bgr24')
        cv2.imshow('Tello AV', img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
except Exception as e:
    print(f"AV解码错误: {e}")
finally:
    container.close()
    tello.streamoff()
    tello.end()
    cv2.destroyAllWindows()