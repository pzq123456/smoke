import cv2
import threading
from ultralytics import YOLO
from pathlib import Path
import time

class RTSPStreamer:
    """多线程视频流读取类，确保永远获取最新的一帧"""
    def __init__(self, rtsp_url):
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
            if ret: self.frame = frame
            else: break
        self.cap.release()

    def read(self): return self.frame
    def stop(self):
        self.stopped = True
        self.thread.join()

def main():
    script_dir = Path(__file__).parent
    model_path = script_dir / "runs" / "detect" / "yolo26m_smoking_20260626_0601" / "weights" / "best.pt"
    
    if not model_path.exists():
        print(f"错误：找不到模型文件 {model_path}"); return

    # 加载模型：base_model用于检测人(COCO类0)，smoking_model用于识别抽烟
    print("加载模型中...")
    base_model = YOLO("yolo26n.pt", task="detect")
    smoking_model = YOLO(str(model_path), task="detect")
    
    rtsp_url = "rtsp://118.140.234.166:8554/dahua1001722"
    # rtsp_url = "0"  # 使用摄像头测试
    streamer = RTSPStreamer(rtsp_url)
    time.sleep(1)
    
    frame_count = 0
    start_all = time.time()
    fps_display, fps_timer = 0, time.time()
    
    try:
        while True:
            t1 = time.time()
            frame = streamer.read()
            if frame is None: continue
            
            annotated_frame = frame.copy()
            h, w = frame.shape[:2]
            
            # 1. 基础模型首先检测人体 (classes=[0] 代表人)
            person_results = base_model.predict(frame, conf=0.4, classes=[0], verbose=False, half=True, device=0)
            boxes = person_results[0].boxes.data.cpu().numpy() if len(person_results) > 0 else []
            
            # 2. 遍历检测到的人体，裁剪并判断是否在抽烟
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                
                person_roi = frame[y1:y2, x1:x2]
                is_smoking = False
                
                if person_roi.size > 0:
                    # 局部区域检测抽烟动作
                    smoke_res = smoking_model.predict(person_roi, conf=0.35, verbose=False, half=True, device=0)
                    if len(smoke_res) > 0 and len(smoke_res[0].boxes) > 0:
                        is_smoking = True
                
                # 3. 绘制人体框及上方标签
                label = "SMOKING" if is_smoking else "Normal"
                color = (0, 0, 255) if is_smoking else (0, 255, 0)
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, label, (x1, max(y1 - 10, 20)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # 性能指标计算
            inference_ms = (time.time() - t1) * 1000
            frame_count += 1
            if time.time() - fps_timer >= 1.0:
                fps_display = frame_count / (time.time() - start_all)
                fps_timer = time.time()
            
            # 渲染流信息并展示
            cv2.putText(annotated_frame, f"Inf: {inference_ms:.1f}ms  FPS: {fps_display:.1f}", 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Real-time Detection', annotated_frame)
            
            if frame_count % 30 == 0:
                print(f"帧数: {frame_count} | 延迟: {inference_ms:.1f}ms | FPS: {fps_display:.1f}")
            if cv2.waitKey(1) & 0xFF == ord('q'): break
                
    except Exception as e: print(f"异常: {e}")
    finally:
        streamer.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()