import os
from ultralytics import YOLO

def main():
    # Paths
    weights = "runs/detect/train2/weights/best.pt"
    source = "dataset/test/images"

    # Load model
    model = YOLO(weights)

    # Run inference
    model.predict(
        source=source,
        save=True,
        imgsz=640,
        device='cuda',  # or 'cpu'
        project="yolo_infer",
        name="exp",
        exist_ok=True
    )

    print("Inference complete. Results saved in yolo_infer/exp")

if __name__ == "__main__":
    main()
