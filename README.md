# YOLO Magic🚀 - 强化YOLOv5的视觉任务框架
<div align="center">

![image](https://github.com/WangQvQ/Yolov5_Magic/assets/58406737/24a7718c-2403-46b7-a81e-205ebeb9869e)
[English](README.en-EN.md)|[简体中文](README.md)<br>
 </div>
 
YOLO Magic🚀是一个基于[Ultralytics](https://ultralytics.com) YOLOv5的扩展，旨在为视觉任务提供更强大的功能和更简单的操作。它在YOLOv5的基础上引入了丰富的网络模块，并提供了直观易用的Web操作界面，旨在为新手和专业用户提供更大的便利和灵活性。

## 主要特性

### 1. 强大的网络模块扩展

YOLO Magic🚀引入了一系列强大的网络模块，旨在扩展YOLOv5的功能，并为用户提供更多的选择和可能性：

- **空间金字塔模块**：包括SPP、SPPF、ASPP、SPPCSPC、SPPFCSPC等，这些模块旨在在不同的空间尺度上捕获目标，并增强模型的视觉感知能力。

- **特征融合结构**：我们提供了多样化的特征融合结构，如FPN、PAN、BIFPN等，这些结构旨在有效地融合来自不同层级的特征信息，从而提高模型的目标检测和定位性能。

- **新型骨干网络**：YOLO Magic🚀支持多种预训练的骨干网络，包括EfficientNet、ShuffleNet等，这些骨干网络提供了额外的选择，以提高模型的性能和效率。

- **丰富的注意力机制**：我们提供多种注意力机制，这些机制可以轻松嵌入到您的模型中，以增强对目标的关注度，并提升模型的检测性能。

### 2. 简单易用的Web操作页面

YOLO Magic🚀通过直观的Web操作页面，大大简化了模型推理过程，无需繁琐的命令行操作，您可以轻松完成以下任务：

- **图片推理**：只需进行简单的拖放和配置，即可执行图片推理和目标检测。您可以自由调整置信度、阈值，上传图像并截取感兴趣的区域。
- **视频推理**：TODO

![image](https://github.com/WangQvQ/Yolov5_Magic/assets/58406737/97a2432a-386b-4d7c-b941-f745b4b38db3)


## 为什么选择YOLO Magic🚀

- **更强大的性能**：引入了先进的网络模块，提升了模型的性能和准确性。

- **更简单的操作**：Web界面使操作更加直观和友好，即使是初学者也能快速上手。

- **可定制性**：支持各种自定义配置，满足不同场景和任务的需求。

- **社区支持**：YOLO Magic🚀拥有一个活跃的社区，提供丰富的教程和资源，帮助用户充分利用这一强大的工具。

## 快速开始

你可以通过以下步骤快速开始使用YOLO Magic🚀：

**安装**

```bash
git clone https://github.com/ultralytics/yolov5  # 克隆仓库
cd yolov5
pip install -r requirements.txt  # 安装环境
```

**detect.py 推理**

`detect.py` 在各种数据源上运行推理, 其会从最新的 YOLOv5 [版本](https://github.com/ultralytics/yolov5/releases) 中自动下载 [模型](https://github.com/ultralytics/yolov5/tree/master/models) 并将检测结果保存到 `runs/detect` 目录。

```bash
python detect.py --source 0  # 摄像头
                          img.jpg  # 图像
                          vid.mp4  # 视频
                          path/  # 文件夹
                          'path/*.jpg'  # glob
                          'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                          'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP
```

**Web 页面推理**

使用 `Gradio` 搭建的页面启动一个 `Web` 页面快速启动

```bash
python detect_web.py
```

**训练**

以下指令再现了 YOLOv5 [COCO](https://github.com/ultralytics/yolov5/blob/master/data/scripts/get_coco.sh) 数据集结果. [模型](https://github.com/ultralytics/yolov5/tree/master/models) 和 [数据集](https://github.com/ultralytics/yolov5/tree/master/data) 自动从最新的YOLOv5 [版本](https://github.com/ultralytics/yolov5/releases) 中下载。YOLOv5n/s/m/l/x的训练时间在V100 GPU上是 1/2/4/6/8天（多GPU倍速）. 尽可能使用最大的 `--batch-size`, 或通过 `--batch-size -1` 来实现 YOLOv5 [自动批处理](https://github.com/ultralytics/yolov5/pull/5092). 批量大小显示为 V100-16GB。

```bash
python train.py --data coco.yaml --epochs 300 --weights '' --cfg yolov5n.yaml  --batch-size 128
                                                                 yolov5s                    64
                                                                 yolov5m                    40
                                                                 yolov5l                    24
                                                                 yolov5x                    16
```

![img](https://user-images.githubusercontent.com/26833433/90222759-949d8800-ddc1-11ea-9fa1-1c97eed2b963.png)

**验证**

使用 `val.py` 对你的模型实现验证。

```bash
python val.py --weights yolov5s.pt --task test
					  val
```

## 贡献

我们欢迎开发者和研究者一起贡献代码，共同改进YOLO Magic🚀。

如果你有任何问题或建议，欢迎你提出issue。我们的社区成员将很高兴地为你提供帮助和支持。

## 许可证

本项目的代码和文档现在采用 GNU Affero General Public License 3.0（AGPL-3.0）许可证。详细的许可证内容请参阅附带的 [LICENSE](LICENSE) 文件。

这意味着，任何使用、修改和重新分发本项目的用户必须在提供该项目的网络服务时，公开源代码。请详细阅读许可证以了解更多信息。

---

无论你是一个新手还是一个经验丰富的视觉任务研究者，YOLO Magic🚀都将为你提供一个强大、易用的工具，助力你在计算机视觉领域取得成功。

*探索视觉任务的新境界，尽在YOLO Magic🚀。* 🌟👁️


-----
<div  align="center">
	
<img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Smilies/Face%20with%20Spiral%20Eyes.png" width="10%" alt="Broken system!"/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Smilies/Relieved%20Face.png" width="10%" alt="It's working!"/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Smilies/Astonished%20Face.png" width="10%" alt="It's working but you don't know how!"/><br>



</div>
