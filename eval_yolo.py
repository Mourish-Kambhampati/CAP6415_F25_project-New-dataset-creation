import os
from ultralytics import YOLO

def main():
    # Paths
    weights = "runs/detect/train/weights/best.pt"
    data_yaml = "data.yaml"

    # Load model
    model = YOLO(weights)

    # Evaluate on validation set
    model.val(
        data=data_yaml,
        split="val",
        save=True,
        imgsz=640,
        device="cuda",  # or "cpu"
        project="yolo_eval",
        name="exp",
        exist_ok=True
    )

    print("Evaluation complete. Results saved in yolo_eval/exp")

if __name__ == "__main__":
    main()
