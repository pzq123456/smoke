import cv2
import threading
from ultralytics import YOLO
from pathlib import Path
import time

class RTSPStreamer:
    """多线程视频流读取类，确保永远获取最新的一帧"""
    def __init__(self, rtsp_url):
        # ------------------ 【修改 1/2】 ------------------
        # 如果是纯数字字符串（如 "0"），转换成 int 以支持本地摄像头；否则保持字符串用于远程流
        source = int(rtsp_url) if str(rtsp_url).isdigit() else rtsp_url
        
        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
            else:
                break
        self.cap.release()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.thread.join()

def main():
    # 获取脚本所在目录（项目根目录）
    script_dir = Path(__file__).parent
    
    # 模型路径：runs\detect\yolo26s_smoking_20260625_0033\weights\best.pt
    model_path = script_dir / "runs" / "detect" / "yolo26s_smoking_20260630_0404" / "weights" / "best.pt"

    # ------------------ 【修改 2/2】 ------------------
    # 【方案 A】使用远程 RTSP 流
    rtsp_url = "rtsp://118.140.234.166:8554/dahua1001722"
    
    # 【方案 B】使用本地摄像头（取消注释下一行即可切换）
    # rtsp_url = "0" 
    # --------------------------------------------------
    
    # 检查模型文件是否存在
    if not model_path.exists():
        print(f"错误：找不到模型文件 {model_path}")
        print("请确认模型路径是否正确")
        return
    
    print(f"加载模型: {model_path}")
    model = YOLO(str(model_path), task="detect")
    
    # 使用多线程读取视频流
    print(f"连接视频源: {rtsp_url}")
    streamer = RTSPStreamer(rtsp_url)
    time.sleep(1)  # 等待启动
    
    # 获取第一帧以确定视频尺寸
    frame = streamer.read()
    if frame is None:
        print("错误：无法读取视频流")
        streamer.stop()
        return
        
    height, width = frame.shape[:2]
    print(f"视频尺寸: {width}x{height}")
    print("--- 开始实时流播放 ---")
    print("按 'q' 键退出")
    
    frame_count = 0
    start_all = time.time()
    fps_display = 0
    fps_timer = time.time()
    
    try:
        while True:
            t1 = time.time()
            
            # 从缓存读取最新帧，没有阻塞
            frame = streamer.read()
            if frame is None:
                print("警告：读取到空帧，跳过")
                continue
            
            # 使用 predict 进行推理（比 track 更快）
            results = model.predict(
                frame, 
                conf=0.35,      # 置信度阈值
                verbose=False,  # 不打印详细信息
                half=True,      # 使用半精度加速
                device=0        # 使用 GPU 0
            )
            
            t2 = time.time()
            
            # 绘制检测结果
            annotated_frame = results[0].plot()
            
            # 计算并显示推理时间和 FPS
            inference_ms = (t2 - t1) * 1000
            frame_count += 1
            
            # 计算 FPS
            current_time = time.time()
            if current_time - fps_timer >= 1.0:
                fps_display = frame_count / (current_time - start_all)
                fps_timer = current_time
            
            # 在画面上显示信息
            info_text = [
                f"Inf: {inference_ms:.1f}ms",
                f"FPS: {fps_display:.1f}",
                f"Frame: {frame_count}"
            ]
            
            for i, text in enumerate(info_text):
                cv2.putText(annotated_frame, text, (20, 40 + i * 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 显示实时流
            cv2.imshow('Real-time Detection', annotated_frame)
            
            # 每30帧打印一次状态
            if frame_count % 30 == 0:
                print(f"帧数: {frame_count} | 推理延迟: {inference_ms:.1f}ms | FPS: {fps_display:.1f}")
            
            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n用户按 'q' 键退出")
                break
                
    except KeyboardInterrupt:
        print("\n用户中断处理")
    except Exception as e:
        print(f"处理出错: {e}")
    finally:
        # 清理资源
        streamer.stop()
        cv2.destroyAllWindows()
        
        # 计算并显示统计信息
        if frame_count > 0:
            elapsed_time = time.time() - start_all
            avg_time_per_frame = (elapsed_time / frame_count) * 1000
            print(f"\n--- 统计信息 ---")
            print(f"总处理帧数: {frame_count}")
            print(f"总运行时间: {elapsed_time:.2f}秒")
            print(f"平均推理时间: {avg_time_per_frame:.1f}ms/帧")
            print(f"平均 FPS: {frame_count / elapsed_time:.1f}")
        else:
            print("未处理任何帧")

if __name__ == "__main__":
    main()