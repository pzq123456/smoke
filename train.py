# yoloCons/train.py
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

def main():
    SETTINGS["tensorboard"] = True

    model = YOLO("yolo26n.pt")

    model.train(
        data="Smoking.v4i.yolov11/data.yaml",
        epochs=500,
        imgsz=640,
        patience=50,
        batch=32,
        name="yolo26n_smoking",
        device=-1,
    )

if __name__ == "__main__":
    main()