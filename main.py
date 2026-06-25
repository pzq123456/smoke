import cv2
from ultralytics import YOLO

def main():
    model_path = r"runs\detect\yolo26s_smoking_20260625_0033\weights\best.pt"
    video_path = "gettyimages-752-31-640_adpp.mp4"
    output_path = "output_detected.mp4"
    
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    
    # 获取原视频参数
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 初始化 VideoWriter
    # 使用 'mp4v' 编码器生成 .mp4 文件
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"正在进行推理并保存到: {output_path} ...")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # 推理 (使用 device='cuda' 如果你有显卡)
        results = model.predict(source=frame, conf=0.5, verbose=False)
        
        # 渲染并写入
        annotated_frame = results[0].plot()
        out.write(annotated_frame)
        
        # 可选：如果你想一边写一边看进度
        cv2.imshow("Processing", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"处理完成！文件已保存为: {output_path}")

if __name__ == "__main__":
    main()