# PARTIALLY TAKEN FROM https://github.com/pdebench/PDEBench (MIT LICENSE)

import argparse
from torchvision.datasets.utils import download_url
import os	

BASE_PATH = "https://data.goettingen-research-online.de/api/access/datafile/:persistentId?persistentId=doi:10.25625/I6UYE9/"

class benchmark_dataset:
    files = [
        ["YWBQZN", "PigDetect.zip"]
    ]

class coco_annotation_files:
    files = [
        ["580KFH", "train.json"],
        ["TFNJBS", "val.json"],
        ["J3WSW9", "test.json"]
    ]

class yolov8_weights:
    files = [
        ["PQWVVN", "yolov8_s.pth"],
        ["IZMHTS", "yolov8_m.pth"],
        ["TBGZ7O", "yolov8_l.pth"],
        ["HHFT3E", "yolov8_x.pth"]
    ]

class yolox_weights:
    files = [
        ["VEHNCW", "yolox_s.pth"],
        ["MXBWTW", "yolox_m.pth"],
        ["DFSULD", "yolox_l.pth"],
        ["L2YDBW", "yolox_x.pth"]
    ]

class codino_weights:
    files = [
        ["6YEIHC", "codino_swin.pth"]
    ]

class yolov8_pretrained_weights:
    files = [
        ["https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth", 
        "yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth"],
        ["https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_m_syncbn_fast_8xb16-500e_coco/yolov8_m_syncbn_fast_8xb16-500e_coco_20230115_192200-c22e560a.pth",
        "yolov8_m_syncbn_fast_8xb16-500e_coco_20230115_192200-c22e560a.pth"],
        ["https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_l_syncbn_fast_8xb16-500e_coco/yolov8_l_syncbn_fast_8xb16-500e_coco_20230217_182526-189611b6.pth",
        "yolov8_l_syncbn_fast_8xb16-500e_coco_20230217_182526-189611b6.pth"],
        ["https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_x_syncbn_fast_8xb16-500e_coco/yolov8_x_syncbn_fast_8xb16-500e_coco_20230218_023338-5674673c.pth",
        "yolov8_x_syncbn_fast_8xb16-500e_coco_20230218_023338-5674673c.pth"]
    ]

class yolox_pretrained_weights:
    files = [
        ["https://download.openmmlab.com/mmyolo/v0/yolox/yolox_s_fast_8xb32-300e-rtmdet-hyp_coco/yolox_s_fast_8xb32-300e-rtmdet-hyp_coco_20230210_134645-3a8dfbd7.pth",
        "yolox_s_fast_8xb32-300e-rtmdet-hyp_coco_20230210_134645-3a8dfbd7.pth"],
        ["https://download.openmmlab.com/mmyolo/v0/yolox/yolox_m_fast_8xb32-300e-rtmdet-hyp_coco/yolox_m_fast_8xb32-300e-rtmdet-hyp_coco_20230210_144328-e657e182.pth",
        "yolox_m_fast_8xb32-300e-rtmdet-hyp_coco_20230210_144328-e657e182.pth"],
        ["https://download.openmmlab.com/mmyolo/v0/yolox/yolox_l_fast_8xb8-300e_coco/yolox_l_fast_8xb8-300e_coco_20230213_160715-c731eb1c.pth",
        "yolox_l_fast_8xb8-300e_coco_20230213_160715-c731eb1c.pth"],
        ["https://download.openmmlab.com/mmyolo/v0/yolox/yolox_x_fast_8xb8-300e_coco/yolox_x_fast_8xb8-300e_coco_20230215_133950-1d509fab.pth",
        "yolox_x_fast_8xb8-300e_coco_20230215_133950-1d509fab.pth"]
    ]

class codino_pretrained_weights:
    files = [
        ["https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth",
        "co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth"]
    ]


def get_ids(name):
    datasets = {
        "benchmark_dataset": benchmark_dataset,
        "coco_annotation_files": coco_annotation_files,
        "yolox_weights": yolox_weights,
        "yolov8_weights": yolov8_weights,
        "codino_weights": codino_weights,
        "yolox_pretrained_weights": yolox_pretrained_weights,
        "yolov8_pretrained_weights": yolov8_pretrained_weights,
        "codino_pretrained_weights": codino_pretrained_weights
    }
    
    dataset = datasets.get(name)
    if dataset is not None:
        return dataset.files
    else:
        raise NotImplementedError (f"Dataset {name} does not exist.")


def download_data(root, name):
    """ "
    Download data splits specific to a given setting.

    Args:
    root: The root folder where the data will be downloaded
    name: The name of the dataset to download, must be defined in this python file.  """

    print(f"Downloading data for {name} ...")

    # Load and parse metadata csv file
    files = get_ids(name)
    os.makedirs(root, exist_ok=True)

    # Iterate ids and download the files
    if name in ['yolox_pretrained_weights', 'yolov8_pretrained_weights', 'codino_pretrained_weights']:
        for url, save_name in files:
            download_url(url, root, save_name)
        return
    else:
        for id, save_name in files:
            url = BASE_PATH + id
            download_url(url, root, save_name)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        prog="Download Script",
        description="Helper script to download the TreeLearn data",
        epilog="",
    )

    arg_parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root folder where the data will be downloaded",
    )
    arg_parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name of the dataset setting to download",
    )

    args = arg_parser.parse_args()

    download_data(args.root, args.name)