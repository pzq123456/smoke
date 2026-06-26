# yoloCons/train.py
from datetime import datetime
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

def main():
    SETTINGS["tensorboard"] = True
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    run_name = f"yolo26s_smoking_{current_time}"
    
    model = YOLO("yolo26s.pt")

    model.train(
        data="Smoking/data.yaml",
        epochs=300,
        patience=50,
        imgsz=640,
        batch=128,
        device=-1,
        name=run_name,
        workers=8,

        mixup=0.0,
        mosaic=0.5,
    )

if __name__ == "__main__":
    main()