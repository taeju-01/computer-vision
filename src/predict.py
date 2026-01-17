import os
from ultralytics import YOLO

def main():
    weights = os.getenv("WEIGHTS", "/content/runs/detect/train/weights/best.pt")
    source = os.getenv("SOURCE", "/content/Hard-Hat-Workers-1/test/images")
    conf = float(os.getenv("CONF", "0.25"))
    save = os.getenv("SAVE", "True").lower() == "true"

    model = YOLO(weights)
    model.predict(source=source, conf=conf, save=save)

if __name__ == "__main__":
    main()
