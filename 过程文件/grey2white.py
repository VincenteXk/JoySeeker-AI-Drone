if 1:
    import cv2
    import numpy as np

    # 读取图像
    img = cv2.imread('logo.png')

    # 设置亮度阈值 (0-255，值越大要求越亮)
    threshold = 200

    # 创建掩码：浅色像素为True
    mask = img.mean(axis=2) > threshold

    # 将浅色像素设为白色
    img[mask] = [255, 255, 255]

    # 保存结果
    cv2.imwrite('output.png', img)

# import imageio
# import numpy as np

# # 读取GIF
# gif = imageio.mimread('input.gif', memtest=False)

# # 处理每一帧
# for i, img in enumerate(gif):
#     img = img[..., :3]  # 如果有alpha通道，只取RGB
#     mask = img.mean(axis=2) > 200
#     img[mask] = 255
    

# # 保存GIF
# imageio.mimsave('output.gif', gif, fps=10)