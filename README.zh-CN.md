# YOLO Magic🚀 - Enhancing the YOLOv5 Visual Task Framework

![image](https://github.com/WangQvQ/Yolov5_Magic/assets/58406737/621ae72b-1856-4e49-b5af-1d28b8c3d384)
<div align="center">
  
[English](README.en-EN.md)|[简体中文](README.md)<br>

</div>

  
YOLO Magic🚀 is an extension built on top of [Ultralytics](https://ultralytics.com) YOLOv5, designed to provide more powerful capabilities and simpler operations for visual tasks. It introduces a variety of network modules on top of YOLOv5 and offers an intuitive web-based interface aimed at providing greater convenience and flexibility for both beginners and professionals.

## Key Features

### 1. Powerful Network Module Extensions

YOLO Magic🚀 introduces a range of powerful network modules designed to expand the functionality of YOLOv5 and provide users with more choices and possibilities:

- **Spatial Pyramid Modules**: Includes SPP, SPPF, ASPP, SPPCSPC, SPPFCSPC, etc. These modules aim to capture targets at different spatial scales and enhance the model's visual perception.

- **Feature Fusion Structures**: We provide diverse feature fusion structures such as FPN, PAN, BIFPN, etc., designed to effectively fuse feature information from different levels, improving the model's object detection and localization performance.

- **New Backbone Networks**: YOLO Magic🚀 supports various pre-trained backbone networks, including EfficientNet, ShuffleNet, etc. These backbone networks offer additional choices to enhance the model's performance and efficiency.

- **Rich Attention Mechanisms**: We offer various attention mechanisms that can be easily embedded into your model to enhance focus on targets and improve detection performance.

### 2. Simple and User-Friendly Web Interface

YOLO Magic🚀 greatly simplifies the model inference process with an intuitive web-based interface. No more cumbersome command-line operations. You can easily accomplish the following tasks:

- **Image Inference**: Perform image inference and object detection with simple drag-and-drop and configuration. You can freely adjust confidence levels, thresholds, upload images, and crop areas of interest.

- **Video Inference**: TODO

![image](https://github.com/WangQvQ/Yolov5_Magic/assets/58406737/3402e5c3-d4f2-4182-b805-160ff319aa58)

## Why Choose YOLO Magic🚀

- **Enhanced Performance**: Introduces advanced network modules to improve model performance and accuracy.

- **Simplified Operations**: The web interface makes operations more intuitive and user-friendly, even for beginners.

- **Customizability**: Supports various custom configurations to meet the needs of different scenarios and tasks.

- **Community Support**: YOLO Magic🚀 has an active community that provides rich tutorials and resources to help users make the most of this powerful tool.

## Getting Started

You can quickly get started with YOLO Magic🚀 by following these steps:

**Installation**

```bash
git clone https://github.com/ultralytics/yolov5  # Clone the repository
cd yolov5
pip install -r requirements.txt  # Install the environment
```

**Inference with detect.py**

`detect.py` runs inference on various data sources. It automatically downloads the latest YOLOv5 [model](https://github.com/ultralytics/yolov5/tree/master/models) from the [repository](https://github.com/ultralytics/yolov5/releases) and saves detection results to the `runs/detect` directory.

```bash
python detect.py --source 0  # Camera
                          img.jpg  # Image
                          vid.mp4  # Video
                          path/  # Folder
                          'path/*.jpg'  # Glob
                          'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                          'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP
```

**Web Page Inference**

Launch a web page quickly using the `Gradio`-based interface.

```bash
python detect_web.py
```

**Training**

The following command reproduces YOLOv5 results on the [COCO](https://github.com/ultralytics/yolov5/blob/master/data/scripts/get_coco.sh) dataset. [Models](https://github.com/ultralytics/yolov5/tree/master/models) and [datasets](https://github.com/ultralytics/yolov5/tree/master/data) are automatically downloaded from the latest YOLOv5 [release](https://github.com/ultralytics/yolov5/releases). Training times for YOLOv5n/s/m/l/x are 1/2/4/6/8 days on a V100 GPU (multi-GPU scales linearly). Use the largest `--batch-size` possible or enable YOLOv5 [auto-batching](https://github.com/ultralytics/yolov5/pull/5092) with `--batch-size -1`. Batch sizes shown for V100-16GB.

```bash
python train.py --data coco.yaml --epochs 300 --weights '' --cfg yolov5n.yaml  --batch-size 128
                                                                 yolov5s                    64
                                                                 yolov5m                    40
                                                                 yolov5l                    24
                                                                 yolov5x                    16
```

![img](https://user-images.githubusercontent.com/26833433/90222759-949d8800-ddc1-11ea-9fa1-1c97eed2b963.png)

**Validation**

Use `val.py` to validate your model.

```bash
python val.py --weights yolov5s.pt --task test
```

## Contribution

We welcome developers and researchers to contribute code to improve YOLO Magic🚀 together.

If you have any questions or suggestions, feel free to raise an issue. Our community members will be happy to provide assistance and support.

## License

The code and documentation for this project are now licensed under the GNU Affero General Public License 3.0 (AGPL-3.0). Please refer to the accompanying [LICENSE](LICENSE) file for detailed license terms.

This means that any user who uses, modifies, and redistributes this project must publicly release the source code when providing network services based on this project. Please read the license for more information.

---

Whether you are a beginner or an experienced researcher in visual tasks, YOLO Magic🚀 provides you with a powerful and user-friendly tool to succeed in the field of computer vision.

*Explore new frontiers in visual tasks with YOLO Magic🚀.* 🌟👁️
