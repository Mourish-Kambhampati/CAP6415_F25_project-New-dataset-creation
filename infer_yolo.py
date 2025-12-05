import os
from ultralytics import YOLO

def main():
    # default YOLO training output path
    weights = os.path.join('runs', 'train', 'exp', 'weights', 'best.pt')
    
    # folder with test images
    source = os.path.join('data', 'images', 'test')

    model = YOLO(weights)

    results = model.predict(
        source=source,
        save=True,
        imgsz=640,
        device='cuda',
        project='runs', 
        name='infer',
        exist_ok=True
    )

    print("Inference complete. Results saved in runs/infer")

if __name__ == '__main__':
    main()
