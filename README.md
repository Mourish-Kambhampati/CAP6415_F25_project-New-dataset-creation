# CAP6415_F25_project-New-dataset-creation
This project provides a complete pipeline for preparing a custom lion–tiger dataset and training a YOLOv8 object detector.

### Features
- Rename and organize images and labels
- Convert raw images and YOLO labels into a unified dataset structure
- Split dataset into train/val/test (70/15/15)
- Train a YOLOv8 model on the prepared dataset
- Evaluate and visualize detection results

## Getting Started
#### 1. Clone the Repository
!git clone https://github.com/Mourish-Kambhampati/CAP6415_F25_project-New-dataset-creation.git
cd CAP6415_F25_project-New-dataset-creation

#### 2. Install Requirements

##### Prerequisites
- Python 3.8+
- Optional: NVIDIA GPU with updated drivers (for faster YOLOv8 training)

##### Using pip

- pip install ultralytics
- pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
- pip install opencv-python pillow

##### Verify Installation
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

#### 3. Prepare Your Data

Your images and labels are already organized inside:

dataset_images/
    lion/images/
    tiger/images/

dataset_labels/
    lion/labels/
    tiger/labels/

So preprocessing will simply verify them and build the YOLO train/val/test structure.

#### 4. Run Dataset Preparation

python scripts/preprocessing.py


This will:

- Verify existing YOLO labels

- Split the dataset into train / val / test (70/15/15)

- Create folders:

dataset/{train,val,test}/images
dataset/{train,val,test}/labels


- Generate data.yaml

Then rename images and labels:

python scripts/rename_yolo_dataset.py


This will:

- Rename all image–label pairs to clean zero-padded names (something.jpg → 0001.jpg)

- Keep YOLO structure intact


#### 5. Train YOLOv8
python train_yolo.py


Training results and best weights will be saved in:

 runs/detect/train/


#### 6. Evaluate Model Performance

python infer_yolo.py


This will evaluate the model on the validation set and save:

- mAP scores

- PR curve

- F1 curve

- Confusion matrix

Results stored in:

yolo_eval/exp/


#### 7. Test the Model on New Images
python eval_yolo.py


This will run inference on the test set and save visualized predictions in:

yolo_infer/exp/


### Project Structure
CAP6415_F25_project-New-dataset-creation/
│
├── dataset_images/          # Raw lion and tiger images
│   ├── lion/images/
│   └── tiger/images/
│
├── dataset_labels/          # YOLO label files for each image
│   ├── lion/labels/
│   └── tiger/labels/
│
├── scripts/                 # Data preparation utilities
│   ├── preprocessing.py         # Builds YOLO train/val/test dataset + data.yaml
│   └── rename_yolo_dataset.py   # Renames images & labels to clean zero-padded names
│
├── train_yolo.py            # YOLOv8 training script
│
├── yolo_eval/               # Evaluation outputs (confusion matrix, curves)
├── yolo_infer/              # Inference outputs (predicted images)
│
└── data.yaml                # YOLO configuration file (auto-generated)
