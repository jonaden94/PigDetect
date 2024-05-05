# PigDetect: a diverse and challenging benchmark dataset for the detection of pigs in images

This is the codebase associated with the ECPLF2024 paper titled 'PigDetect: A Diverse and Challenging Benchmark Dataset for the Detection of Pigs in Images'. It provides functionality for training and inference of pig detection models. All commands in this readme must be run while being in the root directory of this repository.

## Setup

To make use of the functionality provided in this repository, you first have to set up the environment.\
For this, we recommend Conda. If Conda is set up and activated, run the following: 

```
source setup/setup.sh
```

The setup has been tested on a linux machine and we cannot provide any information for other operating systems.


## Inference

To perform inference using the trained pig detection models presented in the paper, you first have to download the trained model weights.

| Files        | Download                                             | Size |
| ----------- | :----------------------------------------------------------- | ------------ |
| YOLOX weights   | ```python detection_utils/download.py --name yolox_weights --root data/pretrained_weights/yolox_pigs``` | 718 MB          |
| YOLOv8 weights   | ```python detection_utils/download.py --name yolov8_weights --root data/pretrained_weights/yolov8_pigs``` | 573 MB          |
| Co-DINO weights   | ```python detection_utils/download.py --name codino_weights --root data/pretrained_weights/codino_pigs``` | 900 MB          |

Then, you can use mmdetection functionality to load the models and use them for inference. For this, we prepared a [demo notebook](tools/inference/inference_demo.ipynb).


## Training

To use mmdetection/mmyolo tools for the training of pig detection models, you need to download the PigDetect benchmark dataset and the train/val/test annotation files. Furthermore, to reproduce the results presented in the related paper, you need to download the model checkpoints pre-trained on COCO. Feel free to add further training data or use other train/val splits for your application at hand.

| Files        | Download                                             | Size |
| ----------- | :----------------------------------------------------------- | ------------ |
| PigDetect benchmark dataset   | ```python detection_utils/download.py --name benchmark_dataset --root data/PigDetect``` | 1200 MB          |
| COCO annotation files   | ```python detection_utils/download.py --name coco_annotation_files --root data/PigDetect``` | 14 MB          |
| YOLOX COCO pre-trained weights   | ```python detection_utils/download.py --name yolox_pretrained_weights --root data/pretrained_weights/yolox_coco``` | 718 MB          |
| YOLOv8 COCO pre-trained weights   | ```python detection_utils/download.py --name yolov8_pretrained_weights --root data/pretrained_weights/yolov8_coco``` | 570 MB          |
| Co-DINO COCO pre-trained weights   | ```python detection_utils/download.py --name codino_pretrained_weights --root data/pretrained_weights/codino_coco``` | 900 MB          |

Once you downloaded the dataset, you need to restructure it so that it can be used for training:

```
python detection_utils/restructure_dataset.py
```

With the restructured dataset, the annotation files and the pre-trained weights of the model of your choice, you can train the model using mmdetection/mmyolo functionality. The training of any of the YOLOX or YOLOv8 models only requires a single GPU. For example, the following command is used to train the YOLOv8-X model:

```
python mmyolo/tools/train.py configs/yolov8/yolov8_x.py --work-dir ./work_dirs/pig_models/yolov8_x
```

The training commands for all further models can be found under ``tools/train``. You might need to adjust the batch size in case your GPU does not have sufficient memory. This and other adaptations can be made in the respective config files located in ``configs``. We refer to the [mmdetection documentation](https://mmdetection.readthedocs.io/en/dev-3.x/index.html) for further information on configs.

To train Co-DINO, multiple GPUs are required. If four GPUs are available on your system you can run the following command:
```
bash tools/train/dist_train.sh configs/co-detr/co_dino_swin.py 4 ./work_dirs/pig_models/co_dino_swin
```

## Evaluation

For evaluation, you first need to ensure that the dataset and the coco annotation files are downloaded and the dataset is restructured as described in **Training**.
To obtain mAP and AP50 evaluation metrics with a certain model, you can use the test functionality provided by mmdetection. For example, to obtain test performance of our Co-DINO model trained for pig detection, run the following command:

```
python mmdetection/tools/test.py configs/co-detr/co_dino_swin.py data/pretrained_weights/codino_pigs/codino_swin.pth --work-dir data/evaluation/codino_swin
```

The evaluation commands for all further models trained by us can be found under ``tools/eval``
