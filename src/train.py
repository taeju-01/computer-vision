import os
from ultralytics import YOLO

def main():
    data_yaml = os.getenv("DATA_YAML", "/content/Hard-Hat-Workers-1/data.yaml")
    model_name = os.getenv("MODEL", "yolov8n.pt")
    epochs = int(os.getenv("EPOCHS", "100"))
    imgsz = int(os.getenv("IMGSZ", "640"))

    model = YOLO(model_name)
    model.train(data=data_yaml, epochs=epochs, imgsz=imgsz)

if __name__ == "__main__":
    main()
