---

layout: post
title: "论文整理-03.08"
subtitle: ''
author: "Sun"
header-style: text
tags:
  - Paper
  - Summary
---

2020年的对比方法 -- 6-7个 3个传统方法 2d-cnn svm等 3-4个 19-20年方法 越新含金量越高的越好



## HSI可做方向

#### **变化检测**

==变化检测, 考虑用深度学习怎么做==



### 解决维度问题

#### 波段选择



## ==解决样本少==

#### 迁移学习

用学好的一个网络去训练其他未标记的高光谱图像

1. 很有现实意义

### ==Gan网络生成虚拟样本==

### ==做一些特征工程, 提取更多有用信息, 让信息增多==



#### 各种无监督学习



#### 对比学习,使用数据增强,



#### 度量学习



#### 自监督学习, 

多种图像数据进行预测同一个任务



#### 预训练

标注数据较少的话 pretrain 还是非常重要的，没有 pretrain 很容易在小数据集上无法收敛或者过拟合

## ==解决过拟合==

#### ==attention== 

==可以有不同的attention, 比如说在分支上用 师兄用的特征映射==







### /基于FCN  的图像分割

[/blog/_posts/2020-04-14-神经网络-全连接网络FCN.md](/blog/_posts/2020-04-14-神经网络-全连接网络FCN.md)

现用的是基于cnn的光谱图像分割, 可考虑用基于fcn的HSI 分类

**传统的基于CNN的分割方法：为了对一个像素分类，使用该像素周围的一个图像块作为CNN的输入用于训练和预测。**这种方法有几个**缺点**：

一是**存储开销很大**。例如对每个像素使用的图像块的大小为15x15，然后不断滑动窗口，每次滑动的窗口给CNN进行判别分类，因此则所需的**存储空间根据滑动窗口的次数和大小急剧上升**。
二是**计算效率低下**。相邻的像素块基本上是重复的，针对每个像素块逐个计算卷积，这种计算也有很大程度上的重复。
三是**像素块大小的限制了感知区域的大小**。通常像素块的大小比整幅图像的大小小很多，只能提取一些局部的特征，从而导致分类的性能受到限制。
而全卷积网络**(FCN)则是从抽象的特征中恢复出每个像素所属的类别。**即从图像级别的分类进一步延伸到像素级别的分类。=



# 按文章总结收获和创新点

## MS-CNN+Diversified Metric(DPP+DML)+MS+

笔记:  [19_TGRS_A CNN With Multiscale Convolution and Diversified Metric   for Hyperspectral Image Classification](/blog/_posts/2020-03-02-A CNN With Multiscale Convolution and Diversified Metric   for Hyperspectral Image Classification.md)

IEEE Transactions on Geoscience and Remote Sensing, 2019

### ==收获==和点子

==**神经网络做度量学习**, 在人脸识别中应用的比较多了,== 

1. ==有资料可供研究, 利于迁移到自然图像学习==
2. ==加上其他方法==
3. ==使用DPP先验等==

==解决类内特征相似: 使用度量学习==

==将度量参数$B^*$也加入了训练== -- 只要是可微函数都可用来做损失函数训练

![image-20200302113240820](/img/in-post/20_03/image-20200302113240820.png)

![image-20200302122648104](/img/in-post/20_03/image-20200302122648104.png)

![image-20200302123010188](/img/in-post/20_03/image-20200302123010188.png)

![image-20200302123016706](/img/in-post/20_03/image-20200302123016706.png)

### 创新点

1. MS-CNN 更有利于提取多尺度特征

2. 使用度量学习, 用基于DPP prior的结构损失来讲使**Metric factors多样化**, 可以表示更多的features, 因此该度量模型便获得了更强的表示能力

   ![image-20200306085006435](/img/in-post/20_03/image-20200306085006435.png)

3. 一种加入DPP先验的特殊的结构损失:

   ![image-20200302122652956](/img/in-post/20_03/image-20200302122652956-3661492.png)

### 整体流程

使用**后一个模型微调了参数, 其中将B 和 W 一起训练**

![image-20200302123035175](/img/in-post/20_03/image-20200302123035175.png)

![image-20200302112517053](/img/in-post/20_03/image-20200302112517053.png)

![image-20200302123432385](/img/in-post/20_03/image-20200302123432385.png)

## 3D-CNN

 [17_rs_3DCNN Spectral–Spatial Classification of Hyperspectral Imagery with 3D Convolutional Neural Network.pdf](../../Users/king/Library/Mobile Documents/iCloud~QReader~MarginStudy/Documents/论文/经典网络等/17_rs_3DCNN Spectral–Spatial Classification of Hyperspectral Imagery with 3D Convolutional Neural Network.pdf) 

笔记: [HSI Classification基于深度学习.md](/blog/_posts/2020-03-09-HSI Classification基于深度学习.md)

需要50%的训练样本, 可以达到99的accuracy, 需要的训练集太多

### 优势

1. 无需预处理
2. 更少的参数
   1. lighter
   2. less likely to over-fit
   3. easiler to train

### 收获和点子

**解决分辨率低: 不使用池化操作的原因**: Reducing the spatial resolution in HSI

**Pixel-level**: 在HSI的卷积操作相对于其他的3D-CNN的应用

==**解决样本少的问题:  研究基于3D-CNN的无监督和半监督**分类方法的集成. 研究可以利用未标记样本的3D-CNN的HSI分类技术, 可以解决高光谱图像标记样本难以获取的问题.== (与度量学习目的同, 度量学习算是啥? )



## Code+Contextual CNN+MS+ResNet+FCN

17_TIP_Going Deeper with Contextual CNN for Hyperspectral Image Classification

**IEEE Transactions on Image Processing(Impact factor 6.79)**

**实现:**

[paperwithcode-pytorch](https://paperswithcode.com/paper/going-deeper-with-contextual-cnn-for#code)

### 收获和点子

1. ==考虑利于FCN进行HSI图像分割==
   1. 笔记: [FCN](/blog/_posts/2020-03-07-CNN知识点总结.md)
   2. 能不能通过FCN使用少数标记点(结合语义分割)来标记其他像素点进行训练
   3. ==看语义分割能不能转化为像素点分类==

2. 为了避免过拟合，**作者对patch做了水平和竖直方向及对角线方向的镜像**，从而实现了4倍率的augmentation。

### 优势

1. **multi-scale: 实现了deeper and wider** than other existing deep networks for hyperspectral image classification
2. ==FCN: 使用1*1卷积替代全连接网络, 其实就是对**将卷积核上同一点的所有channel进行FC操作**。==
   1. 能不能通过FCN使用少数标记点(结合语义分割)来标记其他像素点进行训练
   2. ==看语义分割能不能转化为像素点分类==
   3. 声称是第一次用比较**深的网络**来进行高光谱分类

### 整体流程

<img src="https://img-blog.csdnimg.cn/20190320194547550.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9zaGVybG9jay5ibG9nLmNzZG4ubmV0,size_16,color_FFFFFF,t_70" alt="在这里插入图片描述"  />

较深的网络:

![img](https://img-blog.csdnimg.cn/20190320202016861.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9zaGVybG9jay5ibG9nLmNzZG4ubmV0,size_16,color_FFFFFF,t_70)
训练方法:

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190320211214652.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9zaGVybG9jay5ibG9nLmNzZG4ubmV0,size_16,color_FFFFFF,t_70)



200 training samples

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190320211517900.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9zaGVybG9jay5ibG9nLmNzZG4ubmV0,size_16,color_FFFFFF,t_70)

## CONTEXTUAL CNN+FCN

16_IGRSS_CONTEXTUAL DEEP CNN BASED HYPERSPECTRAL CLASSIFICATION

International Geoscience and Remote Sensing Symposium**(0.5??)**

### 整体流程

### 整体流程

![image-20200311100427872](/img/in-post/20_03/image-20200311100427872.png)

相当于只在**第一个layer利用了3*3提取了空间信息**, 之后便只使用1d convolution

<img src="/img/in-post/20_03/image-20200311100950799.png" alt="image-20200311100950799" style="zoom:50%;" />

### 创新点

**提出**?????了一种可以共同利用高光谱图像局部时空光谱特征的**全卷积神经网络**。提出的CNN架构总共使用**9个卷积层**，使用相对较少数量的训练样本即可对其进行有效训练而不会过度拟合。

## 3D-CNN+L2 Regularization+Dropout

[_tgrs_(3dcnn)Deep Feature Extraction and Classification of Hyperspectral Images Based on Convolutional Neural Networks.md](/blog/_posts/2020-03-18-16_tgrs_(3dcnn)Deep Feature Extraction and Classification of Hyperspectral Images Based on Convolutional Neural Networks.md)

16_tgrs_(3dcnn)Deep Feature Extraction and Classification of Hyperspectral Images Based on Convolutional Neural Networks

### 收获

### 3D-CNN公式

在第$i$层第$j$个特征图的位置$(x,y,z)$处神经元的值$v_{i,j}^{xyz}$

<img src="/img/in-post/20_03/image-20200312104532915.png" alt="image-20200312104532915" style="zoom: 50%;" />

#### L2 Regularization of CNN + Dropout + ReLu-- 防止过拟合

<img src="/img/in-post/20_03/image-20200312102837430.png" alt="image-20200312102837430" style="zoom:67%;" />

#### VIRTUAL SAMPLE ENHANCED CNN--扩充样本方法

1. Changing Radiation-Based Virtual Samples
   ![image-20200312112056776](/img/in-post/20_03/image-20200312112056776.png)

   > New virtual sample $y_n$ is obtained by multiplying a random factor and adding random noise to a training sample $x_m$
   >
   > The training sample xmis a cube extracted from the hyper- spectral cube
   >
   > αm indicates the disturbance of light intensity
   >
   > β controls the weight of the random Gaussian noise n

2. Mixture-Based Virtual Samples
   ==Because of the long distance between the object and the sensor, mixture is very common in remote sensing.???== Inspired by the phenomenon, it is possible to generate a virtual sample yk from two given samples of the same class with proper ratios
   
3. <img src="/img/in-post/20_03/image-20200312153238697.png" alt="image-20200312153238697" style="zoom:50%;" />

   > $x_i$and $x_j$are two training samples from the same class

### Feature analyse 特征表示/特征分析 

#### 1D-CNN convolution kernal in HSI

![image-20200312155930922](/img/in-post/20_03/image-20200312155930922.png)



<img src="/img/in-post/20_03/image-20200312155215988.png" alt="image-20200312155215988" style="zoom:33%;" />

> 第一层6个1*8卷积核, (a)随机初始化的权重, (b)学习到的权重

#### 2D-CNN convolution kernal in HSI

![image-20200312182226368](/img/in-post/20_03/image-20200312182226368.png)

#### 特征提取

<img src="/img/in-post/20_03/image-20200312190736709.png" alt="image-20200312190736709" style="zoom:33%;" />

### Result

<img src="/img/in-post/20_03/image-20200312231340782.png" alt="image-20200312231340782" style="zoom: 50%;" />

## HSI-CNN+2D-CNN+code+XgBoost+CapsNet

17_icpr_HSI-CNN: A Novel Convolution Neural Network for Hyperspectral Image

<img src="/img/in-post/20_03/image-20200318092714752.png" alt="image-20200318092714752" style="zoom:67%;" />

训练集需求多: 80%

![image-20200318092913587](/img/in-post/20_03/image-20200318092913587-4494955.png)

## M3D-DCNN+MS-CNN+3D-CNN

MULTI-SCALE 3D DEEP CONVOLUTIONAL NEURAL NETWORK FOR HYPERSPECTRAL IMAGE CLASSIFICATION

MS-CNN标准结构:

<img src="/img/in-post/20_03/image-20200318093520333.png" alt="image-20200318093520333" style="zoom:50%;" />

提出的结构M3D-DCNN:

<img src="/img/in-post/20_03/image-20200318093535883.png" alt="image-20200318093535883" style="zoom:50%;" />

## band-selection(只看这点)

Hyperspectral CNN for Image Classification & Band Selection, with Application to Face Recognition



## ResNet-plus

Identity Mappings in Deep Residual Networks

### 创新点

提出新的残差单元

分析了resnet结构原理

<img src="/img/in-post/20_03/image-20200318133124770.png" alt="image-20200318133124770" style="zoom:67%;" />



## Self-supervised + Hyperspectral Image Restoration + structual prior

19_arxiv_(0)Self-supervised Hyperspectral Image Restoration using Separable Image Prior(利用可分图像先验进行自监督高光谱图像复原)

### Abstract

用于图像恢复的监督学习深度学习方法取得了很好的效果, 但是其在像HSI这种非灰度/彩色图像上效果不佳. 

我们提出了一个新型的自监督学习策略, 其通过一个被破坏图像自动产生一个训练集, 并使用干净图片生成一个去噪网络, 

另一个值得注意的地方是我们方法中使用了一个可分离的卷积层, 实验证明其可以获得更多的HSI先验用以进行图像恢复



19_IS_(77)Hyperspectral image unsupervised classification by robust manifold matrix factorization(基于鲁棒流形矩阵分解的高光谱图像无监督分类)

Information Science: 5.5



19_TGRS_(37)Learning Compact and Discriminative Stacked Autoencoder for Hyperspectral Image Classification(学习紧凑而有区别的堆叠式自动编码器，用于高光谱图像分类)

TGRS: 5.7



19_TGRS_(55)Deep Learning for Hyperspectral Image Classification- An Overview

