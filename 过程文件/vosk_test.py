import vosk
import pyaudio
import json
import time
from djitellopy import Tello

command_language = 'cn'#['cn','en']

# 语音指令集
# 注意这会大幅提高召回率但降低准确率
# 也就是说你说的所有其他话都可能被误识别成命令，命令之间有一定概率被误识别，需要先熟悉模型的发音
if command_language == 'cn':
    COMMANDS = '["上", "下", "左", "右", "前", "后", "顺时针", "逆时针", "起飞", "悬停", "降落", "断电"]'#这个左念“做”才行
    model = vosk.Model("vosk-model-small-cn-0.22")
else:
    COMMANDS = '["up", "down", "left", "right", "forward", "backward", "clock", "counter", "take off", "hover", "land", "shut down"]'
    model = vosk.Model("vosk-model-small-en-us-0.15")

# 初始化Vosk模型（需先下载）
recognizer = vosk.KaldiRecognizer(model, 16000, COMMANDS)

# 音频输入
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000,
                input=True, frames_per_buffer=3200)

print("开始监听语音指令...")

class asr_module():
    def __init__(self, recognizer, stream):
        self.recognizer = recognizer
        self.stream = stream
        self.t = time.time()

    def get_result(self):
        data = self.stream.read(2000, exception_on_overflow=False)
        self.recognizer.AcceptWaveform(data)
        pr = json.loads(recognizer.PartialResult()).get('partial','')
        if pr != '':
            print(time.time()-self.t)
            self.t = time.time()
            # 靠获取FinalResult强制跳过静音延迟等待
            recognizer.FinalResult()

        return pr

# 初始化
tello = Tello()
tello.connect()

print(f"电池电量: {tello.get_battery()}%")
print(f"温度: {tello.get_temperature()}℃")



Audio_Recognizer = asr_module(recognizer, stream)

COMMAND_SUSTAIN = 0.5
SPEED = 15
rc_commands = {"上", "下", "左", "右", "前", "后", "顺时针", "逆时针","悬停"}

rc_command_time = time.time()

while True:
    try:

        t = time.time()

        command = Audio_Recognizer.get_result()
        if command != '':
            print(command)
        if not tello.is_flying:
            if command == '起飞':
                # Something it takes a looooot of time to take off and return a succesful takeoff.
                # So we better wait. Otherwise, it would give us an error on the following calls.
                tello.takeoff()
        # elif command == '断电':
        #     tello.emergency()
        #     tello.end()
        elif command == '降落':
            tello.land()
        elif command in rc_commands:
            if command == '悬停':
                tello.send_rc_control(0,0,0,0)
            elif command == '左':
                tello.send_rc_control(-SPEED,0,0,0)
            elif command == '右':
                tello.send_rc_control(SPEED,0,0,0)
            elif command == '前':
                tello.send_rc_control(0,SPEED,0,0)
            elif command == '后':
                tello.send_rc_control(0,-SPEED,0,0)
            elif command == '上':
                tello.send_rc_control(0,0,SPEED,0)
            elif command == '下':
                tello.send_rc_control(0,0,-SPEED,0)
            # elif command == '顺时针':
            #     tello.send_rc_control(0,0,0,SPEED)
            # elif command == '逆时针':
            #     tello.send_rc_control(0,0,0,-SPEED)
            rc_command_time = t
        
        if t-rc_command_time > COMMAND_SUSTAIN:
            tello.send_rc_control(0,0,0,0)
            rc_command_time = t
            print(t)
    except Exception as e:
        print(e)
        tello.end()
        break

tello.end()



