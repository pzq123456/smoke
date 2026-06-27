#!/usr/bin/env python3
import cv2

# 刚才你启动的服务地址
STREAM_URL = "http://127.0.0.1:8080/stream"

def main():
    print(f"正在尝试连接推流服务: {STREAM_URL} ...")
    print("提示: 成功打开后，按 'q' 键或 'Esc' 键可以退出窗口。")
    
    # OpenCV 的 VideoCapture 完美支持直接读取 MJPEG HTTP 流
    cap = cv2.VideoCapture(STREAM_URL)
    
    if not cap.isOpened():
        print("❌ 错误：无法连接到推流服务，请检查服务端是否已启动。")
        return

    # 只要连接成功，此时服务端的摄像头必然会亮起
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 错误：未能获取到视频帧。")
            break
            
        cv2.imshow("MJPEG Stream Test", frame)
        
        # 每 10 毫秒检测一次键盘，按 q 或 Esc 退出
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("测试结束。")

if __name__ == "__main__":
    main()