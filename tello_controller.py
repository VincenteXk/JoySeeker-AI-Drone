import time
import socket
from djitellopy import Tello
Tello.LOGGER.setLevel(30) #只显示WARNING及以上的日志信息
import cv2

COMMAND_SUSTAIN = 0.5
SPEED = 25
command_set = {"上", "下", "左", "右", "前", "后", "顺时针", "逆时针","悬停"}

class tello_controller():
    def __init__(self):
        # 套接字修复
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.close()

        # 初始化
        self.tello = Tello()
        self.tello.connect()
        print('无人机初始化...')
        print(f"电池电量: {self.tello.get_battery()}%")
        print(f"温度: {self.tello.get_temperature()}℃")
                
        # 开启视频流
        self.tello.streamon()
        print('摄像头初始化...')
        time.sleep(2)
        self.frame_reader = self.tello.get_frame_read()

        # 计时器初始化
        self.rc_command_time = [time.time(),]*4

        print('\n无人机控制模块初始化成功')

        
    def get_status(self):
        '''
        回传当前状态
        '''
        try:
            tello_status = {
                'battery': self.tello.get_battery(),
                'flight_time': self.tello.get_flight_time(),
                'height_val': self.tello.get_height(),
                'tof': self.tello.get_distance_tof(),
                'barometer': self.tello.get_barometer(),
                'temp_low': self.tello.get_lowest_temperature(),
                'temp_high': self.tello.get_highest_temperature(),
                'pitch': self.tello.get_pitch(),
                'roll': self.tello.get_roll(),
                'yaw': self.tello.get_yaw(),
                'speed_x': self.tello.get_speed_x(),
                'speed_y': self.tello.get_speed_y(),
                'speed_z': self.tello.get_speed_z(),
                'acc_x': self.tello.get_acceleration_x(),
                'acc_y': self.tello.get_acceleration_y(),
                'acc_z': self.tello.get_acceleration_z()
            }
            # 计算平均温度
            tello_status['temp_avg'] = (tello_status['temp_low'] + tello_status['temp_high']) / 2
        except:
            # 如果出现异常，设置所有值为0或默认值
            tello_status = {
                'battery': 0,
                'flight_time': 0,
                'height_val': 0,
                'tof': 0,
                'barometer': 0,
                'temp_low': 0,
                'temp_high': 0,
                'temp_avg': 0,
                'pitch': 0,
                'roll': 0,
                'yaw': 0,
                'speed_x': 0,
                'speed_y': 0,
                'speed_z': 0,
                'acc_x': 0.0,
                'acc_y': 0.0,
                'acc_z': 0.0
            }
        return tello_status        

    def get_frame(self):
        '''
        回传当前图像
        '''
        return self.frame_reader.frame

    def send_control(self, audio_command, rotate):
        self.t = time.time()
        control = [0,0,0,rotate]

        # 起降指令
        if not self.tello.is_flying:
            if audio_command == '起飞':
                # Something it takes a looooot of time to take off and return a succesful takeoff.
                # So we better wait. Otherwise, it would give us an error on the following calls.
                self.tello.takeoff()
                self.rc_command_time = [time.time(),]*4
        elif audio_command == '降落':
            self.tello.land()
            self.rc_command_time = [time.time(),]*4

        # 平移指令
        if audio_command == '悬停':
            control = [0, 0, 0, 0]
        elif audio_command == '左':
            control[0] = -SPEED
        elif audio_command == '右':
            control[0] = SPEED
        elif audio_command == '前':
            control[1] = SPEED
        elif audio_command == '后':
            control[1] = -SPEED
        elif audio_command == '上':
            control[2] = SPEED
        elif audio_command == '下':
            control[2] = -SPEED
        
        check = 0
        for c in control:
            if c != 0:
                self.rc_command_time = self.t
                check = 1
        if self.tello.is_flying and (check or self.t-max(self.rc_command_time)>COMMAND_SUSTAIN):
            self.tello.send_rc_control(*control)
            print(control)
            self.rc_command_time = [time.time(),]*4
    
    def end(self):
        self.tello.end()
        
    def __del__(self):
        self.end()


if __name__ == '__main__':
    import cv2
    Drone_Controller = tello_controller()
    while True:
        frame = Drone_Controller.get_frame()
        if frame is not None:
            # 颜色转换
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # 显示
            cv2.imshow('Tello Dashboard', frame)
        
        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # 清理
    Drone_Controller.end()
    cv2.destroyAllWindows()