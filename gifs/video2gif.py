import cv2

# 读取视频
cap = cv2.VideoCapture('Exp2_test_R.mp4')
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 计算起止帧
start_frame = int(4 * fps)
end_frame = int(8 * fps)

# 设置起始帧
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

frames = []
for _ in range(start_frame, end_frame):
    ret, frame = cap.read()
    if not ret: break
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

cap.release()

# 导出GIF（需要安装imageio）
import imageio
imageio.mimsave('output.gif', frames[::3], fps=int(fps/1.5))