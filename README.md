# 分享一些关于改进Yolov5的tricks以及实验结果
# Share some tricks of improving Yolov5  and experiment results
![赛博朋克](https://user-images.githubusercontent.com/58406737/185147502-cf221312-db48-4635-ab95-fb45e443bed3.png)

## 《Yolov5实验数据全部开源》

分享一些改进YOLOv5的技巧，不同的数据集效果肯定是不同的。有算力的话还是要多尝试

-----

有关代码怎么使用我就不过多介绍了，大家可以去看我的博文，或者官方的文档，我在这统一做一个汇总

1.[手把手带你调参Yolo v5 (v6.1)（一）](https://blog.csdn.net/weixin_43694096/article/details/124378167)🌟强烈推荐

2.[手把手带你调参Yolo v5 (v6.1)（二）](https://blog.csdn.net/weixin_43694096/article/details/124411509?spm=1001.2014.3001.5502)🚀

3.[如何快速使用自己的数据集训练Yolov5模型](https://blog.csdn.net/weixin_43694096/article/details/124457787)

4.[手把手带你Yolov5 (v6.1)添加注意力机制(一)（并附上30多种顶会Attention原理图）](https://blog.csdn.net/weixin_43694096/article/details/124443059?spm=1001.2014.3001.5502)🌟

5.[手把手带你Yolov5 (v6.1)添加注意力机制(二)（在C3模块中加入注意力机制）](https://blog.csdn.net/weixin_43694096/article/details/124695537)

6.[Yolov5如何更换激活函数？](https://blog.csdn.net/weixin_43694096/article/details/124413941?spm=1001.2014.3001.5502)

7.[Yolov5 (v6.1)数据增强方式解析](https://blog.csdn.net/weixin_43694096/article/details/124741952?spm=1001.2014.3001.5502)

8.[Yolov5更换上采样方式( 最近邻 / 双线性 / 双立方 / 三线性 / 转置卷积)](https://blog.csdn.net/weixin_43694096/article/details/125416120)

9.[Yolov5如何更换EIOU / alpha IOU / SIoU？](https://blog.csdn.net/weixin_43694096/article/details/124902685)

10.[Yolov5更换主干网络之《旷视轻量化卷积神经网络ShuffleNetv2》](https://blog.csdn.net/weixin_43694096/article/details/126109839?spm=1001.2014.3001.5501)🍀

11.[YOLOv5应用轻量级通用上采样算子CARAFE](https://blog.csdn.net/weixin_43694096/article/details/126148795)🍀

12.[空间金字塔池化改进 SPP / SPPF / ASPP / RFB / SPPCSPC](https://blog.csdn.net/weixin_43694096/article/details/126354660?spm=1001.2014.3001.5502)🍀

13.持续更新中

------
 参数量与计算量（以yolov5s为baseline）

| Model         | 参数量(parameters) | 计算量(GFLOPs) |
| ------------- | ------------------ | -------------- |
| C3CBAM        | 6631243            | 14.7           |
| CA            | 7271069            |   /             |
| More_ECA      | 7235401            | 16.5           |
| SE            | 7268157            | 16.6           |
| TransposeConv | 7241917            | 16.6           |
| InceptionConv | 7233597            | 16.2           |
| BiFPN         | 7384006            | 17.2           |
| ShuffleNetv2  | 3844193            | 8.1            |
| CARAFE        | 7369445            | 17.0           |
| SPP           | 7225885            | 16.5           |
| SPPF          | 7235389            | 16.5           |
| ASPP          | 15485725           | 23.1           |
| BasicRFB      | 7895421            | 17.1           |
| SPPCSPC       | 13663549           | 21.7           |
| SPPCSPC_group | 8355133            | 17.4           |


------

还有一些其他tircks的实验结果我正在整理中，后续我会更新在Github的


