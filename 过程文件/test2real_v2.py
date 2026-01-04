
def calculate_rotate_speed(delta_x):
    if abs(delta_x) > 0.15:
        raw_speed = delta_x * 120
        clamped_speed = max(-30, min(30, raw_speed))
    else:
        clamped_speed = 0
    return clamped_speed



# --- 4. 系统集成类 ---



# --- 5. 主程序入口 ---

if __name__ == "__main__":
    flying = False
    import time
    rc_command_time = time.time()
    system = SmileDroneSystem(MODEL_PATH)
    if system.model is None:
        exit()

    print("Connecting to Tello...")
    drone = Tello()
    try:
        drone.connect()
        drone.streamon()
        bat = drone.get_battery()
        print(f"Battery: {bat}%")
        if bat < 20:
            print("WARNING: Battery too low!")
    except Exception as e:
        print(f"Connection failed: {e}")
        exit()

    frame_read = drone.get_frame_read()
    time.sleep(2) 

    auto_mode = False 
    SAFE_SPEED = 15 # 定义安全速度常量
    
    print("\n=== READY ===")
    print(" [t] - Toggle Auto-Tracking (Warning: Propellers will spin)")
    print(" [l] - LAND immediately")
    print(" [h] - TAKEOFF")
    print(" [w/s/a/d] - Manual Move (Priority over Auto)")
    print(" [q] - Quit")
    import time
    try:
        while True:
            t = time.time()
            # 1. 图像处理
            frame = frame_read.frame

            processed_frame, command = system.process_frame(frame)

            # 2. 初始化本帧控制量 (Reset to 0)
            # lr: 左右, fb: 前后, ud: 上下, yv: 旋转
            lr, fb, ud, yv = 0, 0, 0, 0

            # 3. 叠加自动控制 (Layer 1)
            if auto_mode:
                # 将 0.3 的浮点速度转换为 Tello 的整数速度 (30)
                yv = int(command['yaw_speed'] * 100)
                
                cv2.putText(processed_frame, "AUTO MODE", (500, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(processed_frame, "MANUAL", (20, 450), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # 4. 叠加键盘控制 (Layer 2 - Priority)
            # 使用 waitKey 获取按键，如果按住不放，每一帧都会进入这里
            key = cv2.waitKey(1) & 0xFF
            
            # --- 功能键 ---
            if key == ord('q'):
                drone.land()
                break
            elif key == ord('l'):
                drone.land()
                auto_mode = False
            elif key == ord('h') and not flying:
                drone.takeoff()
                flying = True
            elif key == ord('t'):
                auto_mode = not auto_mode
                print(f"Auto Mode: {auto_mode}")

            # --- 飞行键 (覆盖或叠加) ---
            # 设置为 SAFE_SPEED (15) 保证缓慢安全
            if key == ord('w'):
                fb = SAFE_SPEED       
            elif key == ord('s'):
                fb = -SAFE_SPEED      
            
            if key == ord('a'):
                lr = -SAFE_SPEED      
            elif key == ord('d'):
                lr = SAFE_SPEED       
                
            if key == ord('u'): # 上升 Up
                ud = SAFE_SPEED
            elif key == ord('j'): # 下降 Jump/Down
                ud = -SAFE_SPEED
            if fb or lr or ud:
                rc_command_time = t
            
            if t-rc_command_time > 0.5:
                drone.send_rc_control(lr, fb, ud, yv)
                rc_command_time = t
                print(t)

            # 5. 发送最终指令
            # 这里的 send_rc_control 是本帧唯一的出口
            # 如果没按键且没自动模式，发送 (0,0,0,0) 悬停
            # 如果有按键，按键值会覆盖 0
            # 如果有自动模式+按键，会同时发送旋转和位移 (混合动作)
            
            
            
            # 显示画面
            cv2.imshow("Smart Drone Pilot", processed_frame)

    except KeyboardInterrupt:
        print("Emergency Stop!")
        drone.land()
    finally:
        drone.streamoff()
        drone.end()
        cv2.destroyAllWindows()