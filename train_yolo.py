from ultralytics import YOLO

def main():
    DATASET = "data.yaml"
    MODEL_WEIGHTS = "yolov8n.pt"

    model = YOLO(MODEL_WEIGHTS)

    results = model.train(
        data=DATASET,
        imgsz=640,
        epochs=50,
        batch=16,
        workers=8,

        optimizer="AdamW",
        lr0=0.001,
        lrf=0.005,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,

        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        scale=0.3,
        flipud=0.4,
        fliplr=0.5,

        mosaic=0.8,
        mixup=0.25,
        copy_paste=0.25,

        device=0,
        amp=True,
        val=True,

        plots=True,
        patience=50,
        save_period=10,
        pretrained=True,
    )

if __name__ == "__main__":
    main()
