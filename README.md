# 分享一些关于改进Yolov5的tricks以及实验结果
# Share some tricks of improving Yolov5  and experiment results
![赛博朋克](https://user-images.githubusercontent.com/58406737/185147502-cf221312-db48-4635-ab95-fb45e443bed3.png)

 <center>  <div xss=removed> <img src="https://img.shields.io/badge/QQ%E4%BA%A4%E6%B5%81%E7%BE%A4-750560739-blue"
                        alt="QQ交流群">
                    <b><a href="https://github.com/WangQvQ/Yolov5_Magic">
                            <img src="https://img.shields.io/badge/%E8%BF%AA%E8%8F%B2%E8%B5%AB%E5%B0%94%E6%9B%BC-YOLO%20Magic-critical"
                                alt="迪菲赫尔曼">
                        </a>
                        <a href="https://github.com/iscyy/yoloair">
                            <img src="https://img.shields.io/badge/%E8%8A%92%E6%9E%9C%E6%B1%81%E6%B2%A1%E6%9C%89%E8%8A%92%E6%9E%9C-YOLO%20Air-red"
                                alt="芒果汁没有芒果"></a>
                        <a href="https://www.captainai.net/diffie/">
                             <img
                                    src="https://img.shields.io/badge/%E7%A6%8F%E5%88%A9-%E5%85%8D%E8%B4%B9AI%E6%95%99%E7%A8%8B-success"
                                    alt="AI教程"> </center>

## 《Yolov5实验数据全部开源》

分享一些改进YOLOv5的技巧，不同的数据集效果肯定是不同的。有算力的话还是要多尝试

-----

有关代码怎么使用我就不过多介绍了，大家可以去看我的博文，或者官方的文档，我在这统一做一个汇总

1.[手把手带你调参Yolo v5 (v6.2)（一）](https://blog.csdn.net/weixin_43694096/article/details/124378167)🌟强烈推荐

2.[手把手带你调参Yolo v5 (v6.2)（二）](https://blog.csdn.net/weixin_43694096/article/details/124411509?spm=1001.2014.3001.5502)🚀

3.[如何快速使用自己的数据集训练Yolov5模型](https://blog.csdn.net/weixin_43694096/article/details/124457787)

4.[手把手带你Yolov5 (v6.2)添加注意力机制(一)（并附上30多种顶会Attention原理图）](https://blog.csdn.net/weixin_43694096/article/details/124443059?spm=1001.2014.3001.5502)🌟

5.[手把手带你Yolov5 (v6.2)添加注意力机制(二)（在C3模块中加入注意力机制）](https://blog.csdn.net/weixin_43694096/article/details/124695537)

6.[Yolov5如何更换激活函数？](https://blog.csdn.net/weixin_43694096/article/details/124413941?spm=1001.2014.3001.5502)

7.[Yolov5 (v6.2)数据增强方式解析](https://blog.csdn.net/weixin_43694096/article/details/124741952?spm=1001.2014.3001.5502)

8.[Yolov5更换上采样方式( 最近邻 / 双线性 / 双立方 / 三线性 / 转置卷积)](https://blog.csdn.net/weixin_43694096/article/details/125416120)

9.[Yolov5如何更换EIOU / alpha IOU / SIoU？](https://blog.csdn.net/weixin_43694096/article/details/124902685)

10.[Yolov5更换主干网络之《旷视轻量化卷积神经网络ShuffleNetv2》](https://blog.csdn.net/weixin_43694096/article/details/126109839?spm=1001.2014.3001.5501)🍀

11.[YOLOv5应用轻量级通用上采样算子CARAFE](https://blog.csdn.net/weixin_43694096/article/details/126148795)🍀

12.[空间金字塔池化改进 SPP / SPPF / ASPP / RFB / SPPCSPC](https://blog.csdn.net/weixin_43694096/article/details/126354660?spm=1001.2014.3001.5502)🍀

13.[用于低分辨率图像和小物体的模块SPD-Conv](https://blog.csdn.net/weixin_43694096/article/details/126398068)🍀

14.持续更新中

------
 **参数量与计算量（以yolov5s为baseline）**
 
**注意力**：
| Model         | 参数量(parameters) | 计算量(GFLOPs) |
| ------------- | ------------------ | -------------- |
| 主干加单层**SE**| 7268157            | 16.6           |
| 主干加单层**CBAM**  | 7268255            | 16.6           |
| 主干加单层**ECA**| 7235392        |   16.5            |
| 主干加单层**CA**|  7261037        |  \          |
| 主干所有**C3**的**BottleNeck**中加（第一版本）|  \        |  \          |
| 主干所有**C3**中加单层（第二版本）|  \       |  \          |
| 。。。| 。。。        |  。。。          |



**SPP结构**：
| Model         | 参数量(parameters) | 计算量(GFLOPs) |
| ------------- | ------------------ | -------------- |
| SPP           | 7225885            | 16.5           |
| SPPF          | 7235389            | 16.5           |
| ASPP          | 15485725           | 23.1           |
| BasicRFB      | 7895421            | 17.1           |
| SPPCSPC       | 13663549           | 21.7           |
| SPPCSPC_group | 8355133            | 17.4           |



**Others**：


| Model         | 参数量(parameters) | 计算量(GFLOPs) |
| ------------- | ------------------ | -------------- |
| TransposeConv upsampling| 7241917            | 16.6           |
| InceptionConv | 7233597            | 16.2           |
| BiFPN         | 7384006            | 17.2           |
| ShuffleNetv2  | 3844193            | 8.1            |
| CARAFE        | 7369445            | 17.0           |

------

实验结果（仅供参考）

| Model             | epoch | freeze | multi_scale | mAP 0.5   | Parameters(M) | GFLOPs |
| ----------------- | ----- | ------ | ----------- | --------- | ------------- | ------ |
| Yolov5s           | 300   | 0      | false       | **0.953** | Nan           | Nan    |
| Yolov5s           | 120   | 8      | false       | 0.936     | Nan           | Nan    |
| Yolov5s_SE        | 120   | 7      | false       | 0.874     | Nan           | Nan    |
| Yolov5s_ECA       | 200   | 7      | false       | 0.937     | Nan           | Nan    |
| Yolov5s_CBAM      | 200   | 7      | **true**    | 0.882     | Nan           | Nan    |
| Yolov5s_BiFPN     | 200   | 7      | false       | 0.935     | Nan           | Nan    |
| Yolov5s_BiFPN_ECA | 200   | 0      | false       | 0.951     | Nan           | Nan    |

------


------
2022.8.24 新加了Pyqt页面的demo，目前只能实现检测🍀

2022.8.22 yolo.py文件新增了中文注释🍀

---
![image](https://user-images.githubusercontent.com/58406737/186384916-db71770b-22a9-4bba-8739-89cb3d6f81dc.png)

