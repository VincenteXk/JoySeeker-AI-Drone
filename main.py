from asr_module import asr_module
from tello_controller import tello_controller
from frame_parser import frame_parser
from UI import UI
import cv2
import time

# 屏蔽视频处理过程的警告
import logging
logging.getLogger('libav').setLevel(50)

# 键盘-语音替代字典
key2audio = {ord('r'):"上", ord('f'):"下", ord('a'):"左", ord('d'):"右", ord('w'):"前", ord('s'):"后",
             ord('j'):"悬停",ord('k'):"起飞",ord('l'):"降落"}


ALLOW_AUDIO_CONTROL = 0
ALLOW_FACE_TRACKING = 1

def timer():
    global t
    t0 = time.time()
    print(t-t0)
    t = t0

if __name__ == '__main__':
    ### 各模块初始化 ###
    #语音识别：监听音频，输出固定的指令
    Audio_Recognizer = asr_module() #0.8s
    #无人机控制：读取视频和状态，发送控制信号
    Drone_Controller = tello_controller() #4s
    #帧解析：根据图像标记笑脸并计算旋转角速度
    Frame_Parser = frame_parser() #0.3s
    #用户界面
    User_Interface = UI() #0s

    ### 主循环 ###
    while True:
        # 按键监听
        key = cv2.waitKey(1) & 0xFF #同步阻塞等待，是循环中必要的，不监听按键也不能删
        # Q键退出
        if key == ord('q'):
            break
        # Y键控制语音开关
        elif key == ord('y'):
            ALLOW_AUDIO_CONTROL = not ALLOW_AUDIO_CONTROL
            print(f"ALLOW_AUDIO_CONTROL: {ALLOW_AUDIO_CONTROL}")
        # U键控制自动追踪开关
        elif key == ord('u'):
            ALLOW_FACE_TRACKING = not ALLOW_FACE_TRACKING
            print(f"ALLOW_FACE_TRACKING: {ALLOW_FACE_TRACKING}")

        # 音频输入监听
        audio_command = ''
        if ALLOW_AUDIO_CONTROL:
            audio_command = Audio_Recognizer.get_result() #0.1-0.3s，因此不能常开
            if audio_command != '':
                print(audio_command)
        # 否则允许键盘控制平移
        else:
            audio_command = key2audio.get(key,'')

        # 无人机信息回传
        frame = Drone_Controller.get_frame() #0s
        status = Drone_Controller.get_status()

        # 信息处理
        if frame is not None:
            # 解析回传图像
            parsed_frame, rotate = Frame_Parser.parse(frame) #0.01s
            # 发送无人机控制信号
            if ALLOW_FACE_TRACKING and rotate is not None:
                Drone_Controller.send_control(audio_command, rotate)
                if rotate == 0:
                    User_Interface.take_screenshot(frame, (0.15,0.85), (0.15,0.85))
            else:
                Drone_Controller.send_control(audio_command, 0)
            # 更新UI
            User_Interface.show(parsed_frame, status) #0s
    
    # 退出
    Drone_Controller.end()
    cv2.destroyAllWindows()
