import os
from ultralytics import YOLO

def main():
    weights = os.path.join('runs', 'train', 'exp', 'weights', 'best.pt')
    data_yaml = os.path.join('data', 'data.yaml')

    model = YOLO(weights)

    results = model.val(
        data=data_yaml,
        split='val',
        save=True,
        imgsz=640,
        device='cuda',
        project='runs',
        name='eval',
        exist_ok=True
    )

    print('Evaluation complete. Results saved in runs/eval')

if __name__ == '__main__':
    main()
