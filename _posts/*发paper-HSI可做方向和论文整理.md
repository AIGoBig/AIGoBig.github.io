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

# 写论文准备, 如何提高论文阅读效率

## 规定

用绿色标记出好的表达, 好的观点, 常复习, 写论文可用

把单词放入欧陆, 休息时间背诵 

2020年的对比方法 -- 6-7个 3个传统方法 2d-cnn svm等 3-4个 19-20年方法 越新含金量越高的越好

<img src="/img/in-post/20_07/image-20200817222728605.png" alt="image-20200817222728605" style="zoom:50%;" />

## 步骤

## 学习队列

### 三元组的方法, 相似的放在一起, 不相似的远离

输入为3D块好还是1D好?

### 整合后, 用数据蒸馏减少模型, 提高效率



### 加grid masks 加大学习难度,可以学习到更加细粒度的特征

脑pet

加遮挡

和bert联系

### `mix up`: 提升很大,可行性很大

mixup
mixup是基于邻域风险最小化（VRM）原则的数据增强方法，使用线性插值得到新样本数据。

![image-20201022165408522](/img/in-post/20_07/image-20201022165408522.png)

### 声纹特征

抓的不是主体特征—内容,而是次要特征—声纹,识别是谁说的话, 看下对HSI分类效果提升



### `ResNeSt网络`

最新模型

Resnest18是其改进

![image-20201022145333464](/img/in-post/20_07/image-20201022145333464.png)



### 自动调参工具 NNI

### 结果后处理 结果修正方案

![image-20201022145503754](/img/in-post/20_07/image-20201022145503754.png)



### SNE降维后分类 — 找代码实现

降维后精度比较高, 但时间复杂度比较高, 

`可创新` — 

​	`1.可以用来提升精度,然后尽量在其他方法上创新`

​	`2.能不能使复杂度降低 — 大创新`

t-sne cnn

![IMG_96C99F638D56-1](/img/in-post/20_07/IMG_96C99F638D56-1.jpeg)

![image-20201007192744520](/img/in-post/20_07/image-20201007192744520.png)

> 组合性方法要把每一个小方法都单独拎出来然后对比,才能分析出到底是那部分在起作用

![IMG_2158](/img/in-post/20_07/IMG_2158.PNG)

### 光谱聚类

### 表示学习

representative learning [37] has become a new branch of remote sensing image classification, which is highly effective and promising

### 对比学习

SimCLR

### 自动网络结构设计 — 自适应于任何的HSI都可获得最佳参数

Automatic Design of Convolutional Neural Network for Hyperspectral Image Classification

深度学习架构的自动设计为未来的研究开辟了新窗口，显示了使用神经架构的优化功能进行准确的HSI分类的巨大潜力

使用基于梯度下降的搜索算法来有效地找到在验证数据集上评估的最佳深度结构。

![image-20200818023310727](/img/in-post/20_07/image-20200818023310727.png)



## 论文笔记方式

1. 论文用marginnote, 在上面做好**重点和大纲的标注**, 加上**标签,** 方便后期可以按类别阅读
2. 笔记在markdown上写好, 尤其要写**创新点和自己能做的地方**, 随时可以回来看

## HSI可做方向

### 表示学习非常有前景

### **变化检测**

`变化检测, 考虑用深度学习怎么做`

### 解决维度问题

### 波段选择

### `零样本学习 — 新添加的类别的学习`  — 避免分类精度过高无法提升

![image-20200827132928253](/img/in-post/20_07/image-20200827132928253.png)

![image-20200827133027050](/img/in-post/20_07/image-20200827133027050.png)

法1: 见过, 直接预测 ; 没见过, 用欠拟合的模型进行预测

> 见过没见过: feature map距离和测试集里样本的feature map 距离远近判断
>
> 

法2:  用已有字体生成没见过的字体





### /基于FCN  的图像分割

[/blog/_posts/2020-04-14-神经网络-全连接网络FCN.md](/blog/_posts/2020-04-14-神经网络-全连接网络FCN.md)

现用的是基于cnn的光谱图像分割, 可考虑用基于fcn的HSI 分类

**传统的基于CNN的分割方法：为了对一个像素分类，使用该像素周围的一个图像块作为CNN的输入用于训练和预测。**这种方法有几个**缺点**：

一是**存储开销很大**。例如对每个像素使用的图像块的大小为15x15，然后不断滑动窗口，每次滑动的窗口给CNN进行判别分类，因此则所需的**存储空间根据滑动窗口的次数和大小急剧上升**。
二是**计算效率低下**。相邻的像素块基本上是重复的，针对每个像素块逐个计算卷积，这种计算也有很大程度上的重复。
三是**像素块大小的限制了感知区域的大小**。通常像素块的大小比整幅图像的大小小很多，只能提取一些局部的特征，从而导致分类的性能受到限制。
而全卷积网络**(FCN)则是从抽象的特征中恢复出每个像素所属的类别。**即从图像级别的分类进一步延伸到像素级别的分类。=

### `噪声标签问题` — 避免分类精度过高无法提升

19_TGRS_Spatial Density Peak Clustering for Hyperspectral Image Classification With Noisy Labels

“噪声标签”问题是高光谱图像（HSI）分类的主要挑战之一。为了解决这个问题，提出了一种基于空间密度峰值（SDP）聚类的方法来检测训练集中标记错误的样本。

现有的大多数监督分类器都是基于一种假设，即用于训练分类模型的样本已正确标记[38]。 然而，在实际应用中，该假设并不总是成立，因为在标记过程中需要全面掌握捕获场景的先验知识[39]。 嘈杂的标签（即标签错误的训练样本）始终存在，通常可分为以下三种类型

1）由于全球定位系统（GPS）的精度有限而产生的嘈杂标签。 GPS的野外探索是标记训练样本的最可靠方法之一。 但是，由于GPS的精度有限，因此很难确定所捕获场景中较小区域和不规则区域的位置。 一旦**定位不正确，土地覆被可能贴错标签**，因此可能产生嘈杂的标签。 

2）手动贴标签过程中产生的噪音标签。 视觉解释是标记训练样本的另一种有效方法。 但是，**手动标记过程实际上是非常耗时的**，并且如果不充分了解捕获的场景，则很难实施。 此外，在较大的规则区域中，可能存在一些非同类的小物体.为了减少工作量, 那些**非同类区域可以标记为与周围区域相同的类别**

3) 由于环境因素而产生的噪音标签。 对于某些场景，例如海洋和湿地，不可能进行地面调查，因为这些场景对于人类来说可能是无法到达的。 **此外，由于其他环境因素，例如恶劣的天气，也可能产生标签错误。** 在这些情况下，基于人类标签的视觉解释不可避免地会产生嘈杂的标签。 

因此，可以得出结论，“噪声标签”问题确实是HSI分类中的主要挑战之一。

为了提高深度学习分类器的鲁棒性，Mnih和Hinton提出了两个鲁棒的损失函数来训练深度神经网络[41]。宋等。文献[42]提供了一种标签细化算法来为这种嘈杂和丢失的标签数据集调整标签，这已被证明是一种有效的标签细化算法，并且对于生成更好的标签很有用。在[43]中，提出了一种新颖的判别方法，用于从自然语言中提取位置关系，从而可以减少在地理空间界面中呈现给用户的噪声。维森特等。 [44]提出了一种“懒惰注解”，通过共同学习阴影区域分类器并恢复训练集中的标签来解决阴影检测中噪声标签的挑战，从而标记重要的阴影区域和一些非阴影区域。

尽管许多研究已经解决了计算机视觉领域中噪声标签的问题，但是由于HSI的高维和非线性结构，这些方法无法直接扩展到HSI分类中。近年来，还对带有噪声标签的HSI分类进行了研究。 Kang等。 [45]在该领域首次提出了一种基于频谱检测和边缘保留滤波（EPF）的噪声标签检测和校正方法。 Tu等。 [46]通过融合光谱角度和局部离群因子（SALOF）来检测HSI中的噪声标签。江等。 [47]开发了一种随机标签传播算法（RLPA）来清除HSI中的标签噪声。在我们之前的工作中，首次应用密度峰值（DP）聚类来检测训练集中的嘈杂标签[48]。结果表明，训练样本的局部密度是区分HSI中噪声标签的重要指标。然而，**基于DP的方法的局限性在于它没有在检测过程中考虑相邻像素之间的空间相关性。为了解决这个问题，提出了一种新的基于空间DP聚类的噪声标签检测算法，该算法可以结合中心样本的相邻样本，进一步测量中心样本的异常程度。**同时，在测量中心样本时不可避免地会引入一些来自不同类别样本的干扰。最近，Tu等。 [49]证明，K选择规则可以在局部区域找到最具代表性的样本，以从置信区域过滤掉这些不同类别的样本。因此，本文引入了K选择规则，以改善空间DP聚类并增强中心样本的表示，它包括以下主要步骤。首先，通过考虑训练样本周围的相邻样本来计算每个类别的训练样本之间的相关系数。然后，基于该像素与其他类别的训练样本的相关系数，计算每个训练样本的局部密度。最后，将具有显着较低的局部密度的训练样本检测为有噪声的标签，并将其从训练集中删除。

具体来说，这项工作的**主要贡献**可以总结如下。 1）通过**考虑相邻像素的空间相关性，**改进了DP聚类算法。此外，发现代替K个相邻像素，选择K个代表性的相邻样本可能是定义不同样本之间的距离的更有效方式。 2）提出的**空间DP聚类算法首次应用于噪声标签检测**。发现空间信息在定义训练样本的局部密度方面也起着重要作用。通过对几个真实的高光谱数据集进行实验，证明了该方法的有效性和优势。 3）提出的方法能够提高不同光谱或光谱空间分类器的分类性能。此外，所提出方法的计算时间仅为几秒钟，这使其在实际应用中非常有用。

本文的其余部分安排如下。第二节介绍了DP聚类算法。第IV-A节详细介绍了建议的噪声标签检测方法。在第四节中，分析了实验结果。最后，结论在第五节中给出。



### `自动网络结构设计`—自适应数据集, 找到最佳结构

Automatic Design of Convolutional Neural Network for Hyperspectral Image Classification

使用基于梯度下降的搜索算法来有效地找到在验证数据集上评估的最佳深度结构。

此外，在3-D Auto-CNN中使用了**改进的cutout方法**，它可以**减轻过拟合现象并提高分类性能**。在训练样本有限的四个流行的高光谱数据集上进行实验（例如，每个数据集总计100个训练样本）。自动CNN（即1维自动CNN和3维自动CNN）始终可以获得更好的分类结果。 **CNN体系结构的优化需要时间（例如数十分钟），但是不需要人工。**此外，设计的CNN`会自动适应特定的数据集`，更重要的是，它为将来的研究开辟了新的研究路径，表明深度学习模型的自动设计对于准确的HSI分类具有巨大潜力

# 可用方法

## 网络相关

### 预训练模型 , 如Bert(NLP方法), 作为初始化, 可以有效避免过拟合  

`使用多的数据进行预训练, 迁移学习到其他数据上..`

`测试集和训练集一起加入预训练`

![image-20200827001524190](/img/in-post/20_07/image-20200827001524190.png)

### 加入word2vec , 转化为向量的形式, 进行分类

### 伪标签的方法,提升成绩 

用大量测试集当做训练集输入..

### 能不能用TFIDF , 统计特征的重要性

TF-IDF: 

1. N-gram , 是一种语言模型, 利用滑窗思想统计频率
2. Countvectorizer, 将文本编码并统计, 统计词的数量(对应CV中特征的数量)
3. 模型和公式:
   ![image-20200826173729123](/img/in-post/20_07/image-20200826173729123.png)
   ![image-20200826173751785](/img/in-post/20_07/image-20200826173751785.png)

### 利用fasttext等, 输出前面最相近类别的类别

对测试集加标签, 找前十个最相近的类别, 当做标签

test_pred = [train_df.iloc[np.argsort(x)[::-1][:10]]['label'].value_counts().index[0] for x in ids[:]]

主要是用相似度判断时要结合多个样本的标签

### `三元组的方法, 相似的放在一起, 不相似的远离`

![image-20200827125655093](/img/in-post/20_07/image-20200827125655093.png)

### 修改Dropout丢弃概率

#### 增加多样性方式:(数据增强)

embading多样性和样本多样性, 如dropout![image-20200819205814238](/img/in-post/20_07/image-20200819205814238.png)

![image-20200819210218654](/img/in-post/20_07/image-20200819210218654.png)

> `使用了3层的dropout, 可以对向量预测多次, 对多次结果求平均. 最后优化整个loss, 可以加速模型收敛, 提高泛化能力`









## 解决样本少

### 迁移学习

用学好的一个网络去训练其他未标记的高光谱图像

1. 很有现实意义

### `Gan网络生成虚拟样本`

### `做一些特征工程, 提取更多有用信息, 让信息增多`



### 各种无监督学习



### 对比学习,使用数据增强,

![image-20200827002631781](/img/in-post/20_07/image-20200827002631781.png)



### 度量学习



### 自监督学习, 

多种图像数据进行预测同一个任务



### 预训练

标注数据较少的话 pretrain 还是非常重要的，没有 pretrain 很容易在小数据集上无法收敛或者过拟合

### `表示学习是非常有前景的`

## 样本扩充的方法



## 解决过拟合

### `attention` 

`可以有不同的attention, 比如说在分支上用 师兄用的特征映射`

## 加上提高效率的方法

### Torch1.6 的混合精度运算

## 好的实验方法

### `不同参数对实验影响的参数分析`, 可以提现较大的工作量及增加说服力

会分析实验, 会把好处对比出来讲出来很重要

### `层内的可视化`

### 训练时间和效率的对比

<img src="/img/in-post/20_07/image-20200807104231519.png" alt="image-20200807104231519" style="zoom: 33%;" />

# 按文章总结收获和创新点

在保证准确率的条件下, 尝试对网络进行修改创新

1. 找到精度/小样本下效果都比较好的网络



## MS-CNN+Diversified Metric(DPP+DML)+MS+

笔记:  [19_TGRS_A CNN With Multiscale Convolution and Diversified Metric   for Hyperspectral Image Classification](/blog/_posts/2020-03-02-A CNN With Multiscale Convolution and Diversified Metric   for Hyperspectral Image Classification.md)

IEEE Transactions on Geoscience and Remote Sensing, 2019

### `收获`和点子

`**神经网络做度量学习**, 在人脸识别中应用的比较多了,` 

1. `有资料可供研究, 利于迁移到自然图像学习`
2. `加上其他方法`
3. `使用DPP先验等`

`解决类内特征相似: 使用度量学习`

`将度量参数$B^*$也加入了训练` -- 只要是可微函数都可用来做损失函数训练

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
   `Because of the long distance between the object and the sensor, mixture is very common in remote sensing.???` Inspired by the phenomenon, it is possible to generate a virtual sample yk from two given samples of the same class with proper ratios
   
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

## 08-15 deep Residual Conv–Deconv Network for Hyperspectral Image Classificatio

### 第一次端到端的全卷积-反卷积网络实现无监督特征学习

<img src="/img/in-post/20_07/image-20200815102242016.png" alt="image-20200815102242016" style="zoom:50%;" />

A. Initial Conv–Deconv Network Architecture

B. Refined Network Architecture

C. Usage of Learned Features for Classification by Fine-Tuning the Network



### 对已学特征深入研究

### 结果

Indian

<img src="/img/in-post/20_07/image-20200815104843334.png" alt="image-20200815104843334" style="zoom:50%;" />

## 08-15 19_TGRS_Cascaded Recurrent Neural Networks for Hyperspectral Image Classification

通过将光谱特征视为一个序列，递归神经网络（RNN）最近已成功用于从高光谱图像（HSI）中学习判别特征。但是，大多数这些模型仅将整个频谱直接输入到RNN中，这可能无法完全探究HSI的特定属性。

在本文中，我们提出**使用门控循环单元的级联RNN模型**，以探索HSI的冗余和互补信息。

**它主要由两个RNN层组成。第一RNN层用于消除相邻光谱带之间的冗余信息，而第二RNN层旨在从不相邻的光谱带中学习补充信息。**

为了提高学习特征的判别能力，我们为提出的模型设计了两种策略。此外，考虑到HSI中包含的丰富的空间信息，我们**通过合并一些卷积层将提议的模型进一步扩展到其光谱空间对应物**。为了测试我们提出的模型的有效性，我们对两个广泛使用的HSI进行了实验。实验结果表明，我们提出的模型可以取得比比较模型更好的结果

<img src="/img/in-post/20_07/image-20200815130354007.png" alt="image-20200815130354007" style="zoom:50%;" />

> 如图1所示，提出的级联RNN模型主要包括四个步骤。 
>
> 对于给定的像素，我们首先将其分为不同的光谱组。 
>
> 然后，对于每个组，我们将其中的光谱带视为一个序列，将其馈入RNN层以**学习特征**。 
>
> 之后，将从每个组中学习到的特征再**次重新排序为序列**，并馈入另一个RNN层以**学习它们的补充信息**。 
>
> 最后，第二RNN层的输出连接到softmax层以得出分类结果

### RNN在HSI中使用

<img src="/img/in-post/20_07/image-20200815145559817.png" alt="image-20200815145559817" style="zoom: 33%;" />

> 序列数据 $x=(x_1,x_2,…,x_T)$, $x_t$通常代表第t个时间step, HSI中代表**第t个band**. 其中φ是非线性激活函数，例如对数S形或双曲线正切函数，$b_h$是**偏差矢量**，$h_{t-1}$是**上次隐藏层的输出**，$W_{hi}$和$W_{hh}$表示从当前输入层到隐藏层的权重矩阵和先前的隐藏层到当前的隐藏层的**权重矩阵**

对于分类任务，通常将hT输入到输出层，并且可以通过使用softmax函数来得出序列属于第i类的概率。 这些过程可以表述为

<img src="/img/in-post/20_07/image-20200815151502375.png" alt="image-20200815151502375" style="zoom: 33%;" />



> 其中bo是一个偏置矢量，$W_{oh}$是从隐藏层到输出层的**权重矩阵**，θ和b是softmax的参数C是要区分的类的数量。 可以使用以下损**失函数来**训练（1）和（2）中的所有这些权重参数

<img src="/img/in-post/20_07/image-20200815151621073.png" alt="image-20200815151621073" style="zoom: 33%;" />

###   级联RNNs,使用GRU

HSI通常包含数百个波段，这使得x的序列很长。 **这样的长期序列增加了训练难度，因为梯度趋于消失或爆炸[49]。** 

为了解决这个问题，一种常用的方法是通过**使用门控单元**（例如LSTM单元和GRU [50]）来设计更复杂的激活功能。 与LSTM单位相比，**GRU具有较少的参数[49]，这可能更适合于HSI分类，因为它通常只有有限数量的训练样本。** 因此，本文选择GRU作为RNN的基本单位。

**隐藏层的激活函数:**

<img src="/img/in-post/20_07/image-20200815154417094.png" alt="image-20200815154417094" style="zoom: 33%;" />

<img src="/img/in-post/20_07/image-20200815155144133.png" alt="image-20200815155144133" style="zoom:33%;" />

> $u_t$ 是更新门, $w_u$是权重值, $v_u$是权重向量, $r_t$是重置门

#### 如何分的光谱组?

由于高光谱传感器的密集频谱采样，**HSI中的相邻频带具有一定的冗余性，而非相邻频带具有一定的互补性。** 为了全面考虑此类信息，我们提出了一个级联的RNN模型。 具体地，**我们将频谱序列x划分为l个子序列z =（z1，z2，...，zl），每个子序列由相邻的谱带组成**。 除最后一个子序列zl外，其他子序列的长度为$d = floor（k / l）$，它表示小于或等于$k / l$的最近整数。 因此，对于第i个子序列$z_i，i∈\{1、2，…,l\}$，由以下频段组成

<img src="/img/in-post/20_07/image-20200815155622383.png" alt="image-20200815155622383" style="zoom:33%;" />

然后，我们将所有子序列分别馈入**第一层RNN**。**这些RNN具有相同的结构并共享参数，从而减少了要训练的参数数量。**在子序列zi中，每个波段都有来自GRU的输出。我们将最后一个波段的输出用作zi的最终特征表示，可以将其表示为$F_i^{(1)}$∈ RH1，其中H1是第一层RNN中隐藏层的大小。

该序列被馈**送到第二层RNN以学习它们的补充信息**。 类似于第一层RNN，我们还将最后一次l的GRU输出用作学习特征F（2）。 为了获得x的分类结果，我们需要将F（2）输入到输出层中，该输出层的大小等于候选类C的数量。这两层RNN都有很多权重参数。 我们选择（3）作为损失函数，并使用BPTT算法同时对其进行优化

### 级联RNN的改进

### `光谱空间级联RNN`

由于大气，仪器噪声和自然光谱变化的影响，同一类别的材料可能具有非常不同的光谱响应，而不同类别的材料可能具有相似的光谱响应。如果仅使用光谱信息，则得到的分类图将具有许多离群值，这被称为“盐和胡椒”现象。

`**我们通过添加一些卷积层将级联的RNN模型扩展到其频谱空间版本。**`

<img src="/img/in-post/20_07/image-20200815161029160.png" alt="image-20200815161029160" style="zoom:50%;" />

> 对于每个$x_i$，我们将其馈入**几个卷积层以学习空间特征**。与[38]相同，我们也使用了三个卷积层，**前两层是池化层。**输入尺寸ω×ω为27×27。三个卷积滤波器的大小分别为4×4×32、5×5×64和4×4×128。经过这些卷积运算符后，**每个$x_i$将生成一个128维空间特征$s_i$**。类似于级联的RNN模型，我们还可以将s =（s1，s2，...，sk）视为长度为k的序列。**将该序列分为1个子序列，然后将它们分别馈入第一层RNN，以减少每个子序列内部的冗余。来自第一层RNN的输出再次组合以生成另一个序列，该序列被馈送到第二层RNN中以学习补充信息。与级联RNN模型相比，频谱空间级联RNN模型更深且更难训练。因此，我们提出一种转移学习的方法来对其进行训练**
>
> 具体来说，我们首先使用所有xi，i∈{1，2，...，k}对卷积层进行预训练。 我们**将输出层替换为两层RNN**，**该输出层的大小为类C的数量**。此外，假定ˆ xi的标签等于其相应像素x的标签**。 然后，我们将得到N×k个样本。 这些样本用于训练卷积层。** 之后，**固定这些卷积层的权重，并再次使用N个训练样本来训练两层RNN。** 最后，**根据学习的参数对整个网络进行微调.**

### 结果

<img src="/img/in-post/20_07/image-20200815130817959.png" alt="image-20200815130817959" style="zoom:50%;" />

**提出的级联RNN模型**与传统RNN对比, 所提出的模型可以**充分探索高维光谱信号的冗余和互补信息**。

在此基础上，我们通过**构造第一层RNN与输出层之间的连接**来设计两种改进策略，从而**生成更具判别力的光谱特征。**

另外，考虑到**空间信息的重要性**，我们进一步**将提出的模型扩展到其光谱空间版本中，以同时学习光谱和空间特征**。

<img src="/img/in-post/20_07/image-20200815162253164.png" alt="image-20200815162253164" style="zoom:33%;" />

<img src="/img/in-post/20_07/image-20200815162334432.png" alt="image-20200815162334432" style="zoom:33%;" />

### 有什么缺点,还能加什么

#### 结果不好, 能否通过结合CNN进行改进? 使用RNN提取光谱特征, CNN提取空谱特征

#### 考虑能不能改进其中的CNN结构来提升精度, 试下程序, 比如加上残差, 密集连接, 对比学习

## 8-16 19_TGRS_Spatial Density Peak Clustering for Hyperspectral Image Classification With Noisy Labels

带噪声标签的高光谱图像分类的**空间密度峰值聚类**

### `噪声标签问题定义`

**“噪声标签”问题**是高光谱图像（HSI）分类的主要挑战之一。为了解决这个问题，提出了一种基于空间密度峰值（SDP）聚类的方法来**检测训练集中标记错误的样本**。

### 步骤

首先，估计每个类别中训练样本之间的**相关系数**。在此步骤中，不是通过考虑各个样本来测量相关系数，而是**考虑围绕每个训练样本的局部窗口中的所有邻居样本或K个代表性邻居样本**。通过这种方式，可以使用空间上下文信息，并且所提出的方法的两个版本（即使用所有相邻样本或K个代表性样本测量相关系数）分别称为SDP和K-SDP。

其次，利用上面计算的相关系数，可以**通过DP聚类算法**获得每个训练样本的局部密度。

最后，那些标签错误的样本通常在每个类别中具有**较低的局部密度**，可以通过定义的决策函数进行识别。在一系列实际的高光谱数据集上使用一系列光谱和光谱空间分类方法评估了所提出的检测方法的有效

### DP CLUSTERING

DP聚类算法已成功应用于噪声标签的检测[48]，并且发现了以下发现。 首先，对于每个类别中的训练样本，发现具有**较低局部密度的那些样本更有可能是嘈杂的标签**。 其次，相关系数被证明是处理HSI的更有效的距离度量标准，而不是使用欧几里得距离。 最后，引入了一种新的基于高斯的局部密度函数，以便更准确地对样本的局部密度建模。 

然而，我们先前工作的一个主要限制是在嘈杂标签检测过程中未考虑空间上下文信息。 为了解决这个问题，提出了两种新颖的基于空间DP（SDP）聚类的方法

![image-20200816103914071](/img/in-post/20_07/image-20200816103914071.png)

**第m类样本之间的相关系数:**

<img src="/img/in-post/20_07/image-20200816104442606.png" alt="image-20200816104442606" style="zoom: 33%;" />

> 其中a和b表示第m类像素的索引。 
>
> 

但是，在这项工作中，还**考虑了每个像素周围的相邻样本**。 这里，提出了两种解决方案，即SDP和K-SDP，如图2所示

<img src="/img/in-post/20_07/image-20200816104602570.png" alt="image-20200816104602570" style="zoom:33%;" />



### 结果

<img src="/img/in-post/20_07/image-20200816105701888.png" alt="image-20200816105701888" style="zoom:33%;" />

<img src="/img/in-post/20_07/image-20200816105730542.png" alt="image-20200816105730542" style="zoom:33%;" />

<img src="/img/in-post/20_07/image-20200816105813968.png" alt="image-20200816105813968" style="zoom:50%;" />

<img src="/img/in-post/20_07/image-20200816110003944.png" alt="image-20200816110003944" style="zoom:50%;" />







![image-20200816105143095](/img/in-post/20_07/image-20200816105143095.png)

![image-20200816105209652](/img/in-post/20_07/image-20200816105209652.png)

首次提出了一种基于空间DP聚类的新颖的噪声标签检测框架，以检测用于HSI分类的噪声标签。 这项工作利用每个训练样本的邻居样本可以保证在检测过程中可以利用更多信息。 

在四个真实的高光谱数据集上进行的实验结果表明，该方法在主观和客观评估方面都是有效的。 但是，该方法的局限性在于**无法自适应地调整空间区域**，这可能意味着要包含更多无用的样本。 因此，设计一种可以使**空间区域自适应调整**以进一步提高检测性能的算法将是我们未来研究的重点。 此外，由于训练样本有限，检测到的标签错误样本**是否可以正确校正**而不是去除样本将是另一个研究课题。

### 可做的地方, 能改进的地方

#### 将异常样本检测与HSI分类结合, 甚至与zero-shoot结合, 尽量节约时间, 实现真实场景下的HSI分类

#### 空间区域自适应调整

#### 是否可以矫正样本而不是消除

## 08-18 19_TGRS_Automatic Design of Convolutional Neural Network for Hyperspectral Image Classification

高光谱图像（HSI）的分类是遥感界的一项核心任务，近来，基于深度学习的方法已显示出它们对HSI进行准确分类的能力。在基于深度学习的方法中，深度卷积神经网络（CNN）已被广泛用于HSI分类。**为了获得良好的分类性能，需要付出巨大的努力来设计合适的深度学习架构。**此外，**手动设计的体系结构可能无法很好地适应特定的数据集。**本文首次提出了自动CNN用于HSI分类的思想。首先，选择许多操作，包括卷积，合并，标识和批处理规范化。然后，**使用基于梯度下降的搜索算法来有效地找到在验证数据集上评估的最佳深度结构。**之后，选择最佳的CNN架构作为HSI分类的模型。具体来说，分别将自动一维自动CNN和3-D自动CNN用作频谱和频谱空间HSI分类器。此外，该切口被引入作为HSI频谱空间分类的正则化技术，以进一步提高分类精度。对四个广泛使用的高光谱数据集（即萨利纳斯，帕维亚大学，肯尼迪航天中心和印第安纳州派恩斯）进行的实验表明，与当前状态相比，自动设计的依赖数据的CNN具有竞争性的分类精度。艺术方法。**此外，深度学习架构的自动设计为未来的研究开辟了新窗口，显示了使用神经架构的优化功能进行准确的HSI分类的巨大潜力**

( 泛读: 快速浏览, 把握概要. 标题,摘要,结论,小标题,图表)

(标题, 摘要, 结论, 图表)

### 概要

(写下小标题)

I. INTRODUCTION

II. AUTOMATIC DESIGN OF CNN AS SPECTRAL CLASSIFIER

​	A. CNN for HSI Classification

​	B. Principles of Automatic Architecture Design

III. AUTOMATIC DESIGN OF CNN AS SPECTRAL–SPATIAL CLASSIFIER

​	A. 3-D Auto-CNN as Spectral–Spatial Classifier

​	B. Cutout-Based Regularization

IV. EXPERIMENTAL RESULTS

​	A. Data Description

​	B. Experimental Setup

​	C. Classification Results for the 1-D Auto-CNNs

​	D. Classification Results for the 3-D Auto-CNNs

​	E. Optimal Architecture

V. CONCLUSION

​	

1. (什么问题)解决问题: **为了获得良好的分类性能，需要付出巨大的努力来设计合适的深度学习架构。**此外，**手动设计的体系结构可能无法很好地适应特定的数据集。**

2. (什么方法)所用方法: 首次提出了**自动CNN用于HSI分类**的思想。首先，选择许多操作，包括卷积，合并，标识和批处理规范化。然后，**使用基于梯度下降的搜索算法来有效地找到在验证数据集上评估的最佳深度结构。**之后，选择最佳的CNN架构作为HSI分类的模型。具体来说，分别将1-D自动CNN和3-D自动CNN用作频谱和频谱空间HSI分类器。此外，该切口被引入作为HSI频谱空间分类的正则化技术，以进一步提高分类精度。

3. (什么效果)达到效果: com- petitive classification accuracy compared with the state-of-the-art methods. 此外，深度学习架构的自动设计为未来的研究开辟了新窗口，显示了使用神经架构的优化功能进行准确的HSI分类的巨大潜力

### GOOD ideas
(精读: 找出关键内容仔细阅读. 注意精度的重点是要有选择性的读)
(所读段落是否详细掌握)

#### 结构自动设计

![image-20200820111654539](/img/in-post/20_07/image-20200820111654539.png)

> 设计合适的体系结构可以看作是一个**搜索问题**。 图1显示了自动设计体系结构的过程。 搜索系统分为三个部分：搜索空间，搜索策略和分类性能估计

#### concat

![image-20200820125726653](/img/in-post/20_07/image-20200820125726653.png)

<img src="/img/in-post/20_07/image-20200820135816813.png" alt="image-20200820135816813" style="zoom: 33%;" />

> 1-D Auto-CNN
>
> 由3个cell组成, 每个cell有2个序列节点
>
>  `**concat每个节点的输入是前面所以节点的输出,  — 类似不同level层特征的结合..**`
>
> 单元格h的输出是通过串联每个中间节点的输出获得的
>
> 令O为一组操作（例如，卷积和池化操作）。 为了加快搜索过程，考虑了在优化问题（例如神经网络）中有用的基于递归的梯度方法。

#### 3-D 自动化设计+1*1卷积

![image-20200820141544250](/img/in-post/20_07/image-20200820141544250.png)

在本文中，使用1×1卷积层作为瓶颈层，这减少了输入数据通道的数量。 由于使用了瓶颈层，因此大大减少了自动搜索和评估最佳网络体系结构的时间成本和复杂性。 此外，它还有助于3-D Auto-CNN训练过程的稳定性

#### cutout

抠图会随机删除输入层而不是要素层的区域，并使用连续的正方形区域而不是单个像素遮盖输入。 此外，抠图是一种很容易实现的方法

<img src="/img/in-post/20_07/image-20200820141506909.png" alt="image-20200820141506909" style="zoom:50%;" />

### 结果

![image-20200820141954154](/img/in-post/20_07/image-20200820141954154.png)

![image-20200820142013834](/img/in-post/20_07/image-20200820142013834.png)

![image-20200820142032826](/img/in-post/20_07/image-20200820142032826.png)

![image-20200820142055837](/img/in-post/20_07/image-20200820142055837.png)

![image-20200820142109296](/img/in-post/20_07/image-20200820142109296.png)

![image-20200820142125018](/img/in-post/20_07/image-20200820142125018.png)

![image-20200820142135401](/img/in-post/20_07/image-20200820142135401.png)

<img src="/img/in-post/20_07/image-20200820142149335.png" alt="image-20200820142149335" style="zoom:50%;" />

<img src="/img/in-post/20_07/image-20200820142201830.png" alt="image-20200820142201830" style="zoom:33%;" />

![image-20200820142334549](/img/in-post/20_07/image-20200820142334549.png)

> 1. 效果方面, 只需要比正常结构的好就行
>
> 2. 参数量也变少了, 说明网络更轻量化易训练了
> 3. 从设计的网络图可以看出, 比手动设计的更有优势

在本文中，**首次提出了用于HSI分类的CNN的自动设计**。与人类专家设计的最新深度学习模型（包括CNN，RNN，3-D CNN和SSRN）相比，**自动设计的CNN具有更好的分类精度**。

具体来说，提出了两个自动HSI分类框架，分别将1-D Auto-CNN和3-D Auto-CNN用作光谱和光谱空间HSI分类器。一种**基于梯度下降的优化技术**用于确定搜索空间中的最佳架构。所选的两种体系结构展示了HSI分类任务的出色能力。

此外，在3-D Auto-CNN中使用了**改进的cutout方法**，它可以**减轻过拟合现象并提高分类性能**。在训练样本有限的四个流行的高光谱数据集上进行实验（例如，每个数据集总计100个训练样本）。自动CNN（即1维自动CNN和3维自动CNN）始终可以获得更好的分类结果。 **CNN体系结构的优化需要时间（例如数十分钟），但是不需要人工。**此外，设计的CNN`会自动适应特定的数据集`，更重要的是，它为将来的研究开辟了新的研究路径，表明深度学习模型的自动设计对于准确的HSI分类具有巨大潜力







### 缺点, 可做的点
(总结: 总览全文, 归纳总结)

#### 实现一下, ,   可以在自己网络的调参上使用,一定可以获得最佳参数

代码实现

任务定义

数据, 参数
环境, 工具
运行结果
如何实现 — 整体架构, 实现细节

## 20_TGRS_Few-Shot Hyperspectral Image Classification With Unknown Classes Using Multit

该论文针对未知类别的小样本高光谱图像分类问题，首次提出一种新颖的**多任务深度卷积网络分类方法。**

`【代码链接】`https://sjliu.me/MDL4OW

### 提出原因

现有的高光谱图像分类方法都假设和预定义图像分类系统是封闭且完整的，并在不可见的数据中**没有未知的或新的图像类别**。然而，这个假设对于现实世界来说可能过于严格，在构建分类系统时，通常会忽略新的类别。分类系统的封闭性会迫使模型在给定新样本的情况下指定标签，并可能导致对已知样本的标签(如作物面积)的覆盖。为了解决这个问题，本文提出一种多任务深度学习方法，该方法**可以在未知类别可能存在的开放世界(MDL4OW)中同时进行分类和重构**。将重构数据与原始数据进行比较;且**未被重构的数据被认为是未知的**，并且由于缺少分类标签，这些重构数据在潜在特征中没有被很好地表示，故需要定义一个阈值来区分未知类别和已知类别；基于此，**本文提出了两种基于极值理论的策略，分别用于少样本和多样本学习场景**。该方法在真实高光谱图像上的测试结果获得了最优的结果，如在萨利纳斯数据的测试结果中使分类的整体精度提高了4.94%。通过考虑开放世界中未知类别的存在，该方法实现了在小样本背景下的更精确的高光谱图像分类。

### 本文主要贡献如下

1. 针对未知类别的高光谱图像分类问题，本文提出了一种新的**多任务深度卷积神经网络学习方法MDL4OW**。该方法能够识别未知小样本图像类别，**显著提高高光谱图像分类精度**； 
2. 本文提出的方法使用**统计模型极值理论来估计所有数据的未知分数**，而不是使用基于质心的方法以分类的方式来估计未知分数。因此该方法在小样本高光谱图像分类方面优于现有的开放集图像分类方法； 
3. 本文针对未知类别的高光谱图像分类精度问题，提出了一种新的评价指标—映射误差，这种度量指标对不平衡分类特别敏感，非常适用于高光谱图像分类问题。

<img src="/img/in-post/20_07/image-20201109214401498.png" alt="image-20201109214401498" style="zoom:50%;" />

![image-20201109215304855](/img/in-post/20_07/image-20201109215304855.png)



### 可以创新的地方

与对比学习结合, 度量学习结合, SimCLR

# 程序实现

## 08-06 18_TGRS_DFFN_Hyperspectral Image Classification With Deep Feature Fusion Network 

Deep Network + Feature Fusion  + 

keras + MATLAB

### Residual Learning

•一方面，引入**残差学习**以优化多个卷积层作为身份映射，这可以简化深度网络的训练并受益于深度的增加。 结果，我们可以建立一个非常深的网络来提取HSI的更多区别特征。 

•另一方面，提出的DFFN模型**融合了不同层次层的输出**，可以进一步提高分类精度。

<img src="/img/in-post/20_07/image-20200806154955776.png" alt="image-20200806154955776" style="zoom:50%;" />

> 由公式也可看出，输出F是经过2个卷积和一个短路连接的X构成的
>
> <img src="/img/in-post/20_07/image-20200806190329356.png" alt="image-20200806190329356" style="zoom:50%;" />

### Feature Fusion

![image-20200806154948646](/img/in-post/20_07/image-20200806154948646.png)

> 首先把**整个模型**分为底层中层和高层三个大模块，**每个模块**中又含有3个卷积块，利用**残差网络**首先对每个模块内部进行融合，保证深度和防止过拟合，
>
> 而后将低中高三层再次进行**特征融合**.
>
> 用**64个大小为1×1**的核来卷积它们都变成了64，进行维度匹配函数确保融合前有相同的维数
>
> z表示融合特征，g1，g2和g3是如上所述的尺寸匹配函数
>
> 这个网络主要就是利用resent把网络做深然后特征融合的结合,**这种机制还是比较适合小样本的**

实验结果:

DFFN 结合了残差学习和feature fusion，发现这两种方法能够比较好地改善网络性能  在OA、AA和Kappa中都取得了最高的精度。

使用了基于网络的方法 --> 优于传统方法(S-CNN 探索样本间的关系, 3d gan 利用对抗训练提高网络的性能)



### `好的实验方法 — 参数分析,可以提现较大的工作量及增加说服力`

1. 主要成分的数量（表示为N）和图像块的大小（表示为S），网络深度（表示为D）是分类性能的关键因素。 因此，本节将分析这三个参数的影响

<img src="/img/in-post/20_07/image-20200806202817096.png" alt="image-20200806202817096" style="zoom: 25%;" />



> 说明HSI中的大多数信息都存在于前几个主要组成部分中。  
>
> 因此，OA值一开始倾向于随n的增加而增加, 然后利用更多的主成分不会进一步提高性能. 

<img src="/img/in-post/20_07/image-20200806202839935.png" alt="image-20200806202839935" style="zoom: 25%;" />

> 主要原因是IP和Salinas具有较大的平滑区域，而PU图像具有更详细的区域
>
> 因此IP和Sa通常随着S变大而增加或变得相对稳定。 相比之下，PU对S比较敏感并且在23*23最大之后下降

<img src="/img/in-post/20_07/image-20200806202908194.png" alt="image-20200806202908194" style="zoom:25%;" />

> 增加D可以提高分类精度。 但是，太深的网络也会导致精度的细微变化。 
>
> 这种现象的主要原因是有限的训练样本（例如，每类仅将10％，2％和0.5％的标记像素用于印度松，帕维亚大学和萨利纳斯图像）是不够训练深度过大的网络的, 

2. 不同样本量时的性能分析

![image-20200806203043079](/img/in-post/20_07/image-20200806203043079.png)

> 精度会随着样本数增加而增加
>
> 也可以看出在所有不同的训练样本下，所提出的DFFN方法始终提供优于其他比较方法的性能。 具体来说，当使用较少的样本时，我们提出的方法比其他分类器具有更多优势

3. 不同的融和策略

<img src="/img/in-post/20_07/image-20200806155248797.png" alt="image-20200806155248797" style="zoom: 50%;" />

> 本文中融合了多层功能，从三个不同的角度（即低，中和高层）探索了深度网络中存在的强大的互补和相关信息。
>
> 在本节中，比较不同的融合方法，以证明所提出的融合策略的有效性。表V显示了通过三种HSI的不同融合方法获得的OA值。在表V中，DFFN2，DFFN6和DFFN9分别指的是融合两个，六个和九个分层层的方法。特别地，**DFFN2-LH表示低层和高层的融合**，而**DFFN2-MH是指中间层和高层的融合**。
>
> 从表V中可以看出，与DRN相比，**融合多层可以在一定程度上改善分类结果**，并且所提出的融合策略DFFN确实优于其他方法。但是，反过来**融合太多层可能会带来冗余信息**，从而大大降低性能（例如DFFN9

###  有什么缺点?还能加什么?

可以再进行堆叠 ?????? 如何加深网络

## 08-07 19_HybridSN_Exploring 3D-2D CNN Feature Hierarchy for Hyperspectral Image Classification

3D接2D

达到过State of the art精确度

python + keras

时间快

### 3D接2D

**所提出的HybridSN模型以3D接2D卷积的形式结合了时空光谱和光谱的互补信息**。

此外，与单独的3D-CNN相比，混合CNN的使用**降低了模型的复杂性**。





![image-20200807092605893](/img/in-post/20_07/image-20200807092605893.png)

<img src="/img/in-post/20_07/image-20200807095655220.png" alt="image-20200807095655220" style="zoom:33%;" />

<img src="/img/in-post/20_07/image-20200807095711504.png" alt="image-20200807095711504" style="zoom: 33%;" />

> 
>
> HybridSN使用了频谱空间3D-CNN和空间2D-CNN。 3D-CNN可以**联合空间光谱特征表示**。在3D-CNN之上的2D-CNN**进一步学习了更多抽象级别的空间表示**。
>
> 2D卷积公式, 3D卷积公式. 2η+ 1是沿着光谱维的核深度. **为了同时增加频谱-空间特征图的数量**，**三次使用三维卷积**，可以将输入HSI数据的**光谱信息**保存在输出体中。考虑到二维卷积可以很好地区分不同光谱波段内的**空间信息**而不会造成光谱信息的大量损失，因此在平化层之前进行一次二维卷积，这对于HSI数据非常重要。

### 网络层参数分析

<img src="/img/in-post/20_07/image-20200807091902257.png" alt="image-20200807091902257" style="zoom: 33%;" />

> 3D convolu- tion kernels are 8 × 3 × 3 × 7 × 1 , 16×3×3×5×8 and 32×3×3×3×16  where **16×3×3×5×8** means 16 3D-kernels of dimension 3×3×5 (i.e., two spatial and one spectral dimension) for all 8 3D input feature maps. **( 注意16和8 分别是输出和输入特征图的数量, 3 * 3 * 5才是卷积核尺寸 )**
>
> the dimension of 2D convolution kernel is 64 × 3 × 3 × 576 , where 64 is the number of 2D-kernels, 3 × 3 represents the spatial dimension of 2D-kernel, and 576 is the number of 2D input feature maps.
>
> 30->24, 因为卷积核(3,3,7)通道上的卷积是7,没有padding操作. 就会减少6个通道
>
> 18，32相乘->576

实验结果

<img src="/img/in-post/20_07/image-20200807103432076.png" alt="image-20200807103432076" style="zoom:50%;" />

> 精度当时达到了SOTA

<img src="/img/in-post/20_07/image-20200807103911155.png" alt="image-20200807103911155" style="zoom:50%;" />

<img src="/img/in-post/20_07/image-20200807104231519.png" alt="image-20200807104231519" style="zoom: 50%;" />

> 由表证明了其效率高. 可以看出，在大约50个epochs内实现了收敛，这表明我们方法的快速收敛。 HybridSN模型的计算效率在训练和测试时间方面列于表, 高于3D CNN

### `有什么缺点?还能加什么?`

`加入resnet`

`加入mean-teacher???`

`作为对比学习的网络???SimCLR` 做无监督板监督学习试试

#### `解决过拟合??`

PU 1%时, 测试集准确率能到100%

```
23.513559997081757 Test loss (%)
95.65278887748718 Test accuracy (%)

94.22477463136445 Kappa accuracy (%)
95.65278991239462 Overall accuracy (%)
92.05721081042722 Average accuracy (%)
[94.02894136 99.74002058 71.8479307  88.09759314 98.42342342 99.41755373
 98.32953683 93.25102881 85.37886873] Class accuracy (%)

                      precision    recall  f1-score   support

             Asphalt       0.95      0.94      0.95      6565
             Meadows       0.99      1.00      0.99     18463
              Gravel       0.95      0.72      0.82      2078
               Trees       0.98      0.88      0.93      3033
Painted metal sheets       0.99      0.98      0.99      1332
           Bare Soil       0.98      0.99      0.99      4979
             Bitumen       0.86      0.98      0.92      1317
Self-Blocking Bricks       0.82      0.93      0.87      3645
             Shadows       0.90      0.85      0.88       937

            accuracy                           0.96     42349
           macro avg       0.94      0.92      0.93     42349
        weighted avg       0.96      0.96      0.96     42349

[[ 6173    45     4    15     0     0   105   213    10]
 [   20 18415     0    26     0     0     0     2     0]
 [   49     4  1493     2     0     0    21   485    24]
 [   96   167    14  2672     0     8    25    49     2]
 [    0     0     0     0  1311     0     0     0    21]
 [    0    29     0     0     0  4950     0     0     0]
 [   11     0     0     0     0     6  1295     0     5]
 [   79     5    48    19     0    61     5  3399    29]
 [   37     2     6     0     9     7    58    18   800]]
```

## 11-07 17_rs_li_3DCNN Spectral–Spatial Classification of Hyperspectral Imagery with 3D Convolutional Neural Network 

```
----------------------------------------------------------------

    Layer (type)        Output Shape     Param #

======================================================

      Conv3d-1    [-1, 16, 196, 3, 3]      1,024

      Conv3d-2    [-1, 32, 196, 1, 1]     13,856

      Linear-3          [-1, 17]     106,641
```



![image-20201107194912491](/img/in-post/20_07/image-20201107194912491.png)

### 优势

1. 完全**不依赖任何预处理或后处理**即可查看HSI多维数据集数据，从而有效地提取了深光谱空间组合特征
2. 更少的参数
   1. lighter
   2. less likely to over-fit
   3. easiler to train
3. 网络层内可视化

### 缺点

需要50%的训练样本, 可以达到99的accuracy, 需要的训练集太多

### 收获和点子

**解决分辨率低: 不使用池化操作的原因**: Reducing the spatial resolution in HSI

**Pixel-level**: 在HSI的卷积操作相对于其他的3D-CNN的应用

`**解决样本少的问题:  研究基于3D-CNN的无监督和半监督**分类方法的集成. 研究可以利用未标记样本的3D-CNN的HSI分类技术, 可以解决高光谱图像标记样本难以获取的问题.` (与度量学习目的同, 度量学习算是啥? 

## 15_JS_hu_1DCNN_Deep Convolutional Neural Networks for Hyperspectral Image Classification

```
        Layer (type)               Output Shape         Param #
======================================================
            Conv1d-1              [-1, 20, 178]             480
         MaxPool1d-2               [-1, 20, 35]               0
            Linear-3                  [-1, 100]          70,100
            Linear-4                   [-1, 17]           1,717
```

<img src="/img/in-post/20_07/image-20201108084557806.png" alt="image-20201108084557806" style="zoom:50%;" />





## 17_icip _he_Multi-scale3DCNN_MULTI-SCALE 3D DEEP CONVOLUTIONAL NEURAL NETWORK FOR HYPERSPECTRAL IMAGE CLASSIFICATION

```
        Layer (type)               Output Shape         Param #
======================================================
            Conv3d-1         [-1, 16, 64, 5, 5]           1,600
            Conv3d-2         [-1, 16, 64, 5, 5]             272
            Conv3d-3         [-1, 16, 64, 5, 5]             784
            Conv3d-4         [-1, 16, 64, 5, 5]           1,296
            Conv3d-5         [-1, 16, 64, 5, 5]           2,832
            Conv3d-6         [-1, 16, 64, 5, 5]             272
            Conv3d-7         [-1, 16, 64, 5, 5]             784
            Conv3d-8         [-1, 16, 64, 5, 5]           1,296
            Conv3d-9         [-1, 16, 64, 5, 5]           2,832
           Conv3d-10         [-1, 16, 62, 4, 4]           3,088
          Dropout-11                [-1, 15872]               0
           Linear-12                   [-1, 17]         269,841
```

MS-CNN标准结构:

<img src="/img/in-post/20_03/image-20200318093520333.png" alt="image-20200318093520333" style="zoom:50%;" />

提出的结构M3D-DCNN:

<img src="/img/in-post/20_03/image-20200318093535883.png" alt="image-20200318093535883" style="zoom:50%;" />

### 效果 

尽管没有任何手工功能或PCA，稀疏编码等预处理/后处理功能，但我们在标准数据集上获得了最先进的结果

### 优势

1. 提出**多尺度3-d卷积块**,以满足空间域中的多尺度目标
2. 由于没有任何手工功能以及PCA，稀疏表示等前/后处理功能，我们提出的M3D-DCNN可以在标准数据集上实现最新的结果。 更重要的是，我们的方法完全是一种端到端的方法，有望在将来使用大规模数据集获得更好的结果

### 还可以改进的地方

把多尺度和强表示深度网络结合

## 17_icpr_luo_3dcnn_HSI-CNN_ A Novel Convolution Neural Network for Hyperspectral Image

感觉和3d卷积区别不大,但是画图好

缺点: 训练集需求多: 80%

![image-20200318092913587](/img/in-post/20_07/image-20200318092913587-4494955-4845695.png)



![image-20201108222724685](/img/in-post/20_07/image-20201108222724685.png)

```
        Layer (type)               Output Shape         Param #
======================================================            
						 Conv3d-1         [-1, 90, 20, 1, 1]          19,530
            Conv2d-2           [-1, 64, 18, 88]             640
            Linear-3                 [-1, 1024]     103,810,048
            Linear-4                   [-1, 17]          17,425
```

## sharma

```
        Layer (type)               Output Shape         Param #
======================================================
            Conv3d-1        [-1, 96, 1, 30, 30]         691,296
       BatchNorm3d-2        [-1, 96, 1, 30, 30]             192
         MaxPool3d-3        [-1, 96, 1, 15, 15]               0
            Conv3d-4         [-1, 256, 1, 7, 7]         221,440
       BatchNorm3d-5         [-1, 256, 1, 7, 7]             512
         MaxPool3d-6         [-1, 256, 1, 3, 3]               0
            Conv3d-7         [-1, 512, 1, 1, 1]       1,180,160
            Linear-8                 [-1, 1024]         525,312
           Dropout-9                 [-1, 1024]               0
           Linear-10                   [-1, 17]          17,425
```

## 17_RSL_liu_A semi-supervised convolutional neural network for hyperspectral image classification

```
        Layer (type)               Output Shape         Param #
======================================================
            Conv2d-1             [-1, 80, 7, 7]         144,080
       BatchNorm2d-2             [-1, 80, 7, 7]             160
         MaxPool2d-3             [-1, 80, 3, 3]               0
            Linear-4                   [-1, 17]          12,257
            Linear-5                  [-1, 720]         519,120
            Linear-6                  [-1, 720]         519,120
       BatchNorm1d-7                  [-1, 720]           1,440
            Linear-8                 [-1, 3920]       2,826,320
       BatchNorm1d-9                 [-1, 3920]           7,840
           Linear-10                  [-1, 200]         784,200
```

![image-20201108224521605](/img/in-post/20_07/image-20201108224521605.png)

![image-20201108224554518](/img/in-post/20_07/image-20201108224554518.png)













## boulch

```
        Layer (type)               Output Shape         Param #
======================================================
            Conv1d-1              [-1, 32, 200]             128
         MaxPool1d-2              [-1, 32, 100]               0
              ReLU-3              [-1, 32, 100]               0
       BatchNorm1d-4              [-1, 32, 100]              64
            Conv1d-5              [-1, 16, 100]           1,552
         MaxPool1d-6               [-1, 16, 50]               0
              ReLU-7               [-1, 16, 50]               0
       BatchNorm1d-8               [-1, 16, 50]              32
            Conv1d-9               [-1, 16, 50]             784
        MaxPool1d-10               [-1, 16, 25]               0
             ReLU-11               [-1, 16, 25]               0
      BatchNorm1d-12               [-1, 16, 25]              32
           Conv1d-13               [-1, 16, 25]             784
        MaxPool1d-14               [-1, 16, 12]               0
             ReLU-15               [-1, 16, 12]               0
      BatchNorm1d-16               [-1, 16, 12]              32
           Conv1d-17               [-1, 16, 12]             784
        MaxPool1d-18                [-1, 16, 6]               0
             ReLU-19                [-1, 16, 6]               0
      BatchNorm1d-20                [-1, 16, 6]              32
           Conv1d-21                [-1, 16, 6]             784
        MaxPool1d-22                [-1, 16, 3]               0
             ReLU-23                [-1, 16, 3]               0
      BatchNorm1d-24                [-1, 16, 3]              32
           Conv1d-25                [-1, 16, 3]             784
        MaxPool1d-26                [-1, 16, 1]               0
             ReLU-27                [-1, 16, 1]               0
      BatchNorm1d-28                [-1, 16, 1]              32
           Conv1d-29                 [-1, 3, 1]             147
             Tanh-30                 [-1, 3, 1]               0
           Linear-31                   [-1, 17]              68
           Linear-32                  [-1, 200]             800

```

## mou

```
        Layer (type)               Output Shape         Param #
======================================================
               GRU-1  [[-1, 2, 64], [-1, 2, 64]]               0
       BatchNorm1d-2                [-1, 12800]          25,600
              Tanh-3                [-1, 12800]               0
            Linear-4                   [-1, 17]         217,617
```



## 17_TIP_DCNN_Contextual CNN+MS+ResNet+FCN

**17_TIP_DCNN_Going Deeper with Contextual CNN for Hyperspectral Image Classification**

**IEEE Transactions on Image Processing(Impact factor 6.79)**

**实现:**

[paperwithcode-pytorch](https://paperswithcode.com/paper/going-deeper-with-contextual-cnn-for#code)

### 收获和点子

1. `考虑利于FCN进行HSI图像分割`
   1. 笔记: [FCN](/blog/_posts/2020-03-07-CNN知识点总结.md)
   2. 能不能通过FCN使用少数标记点(结合语义分割)来标记其他像素点进行训练
   3. `看语义分割能不能转化为像素点分类`

2. `为了避免过拟合，**作者对patch做了水平和竖直方向及对角线方向的镜像**，从而实现了4倍率的augmentation。`

### 优势

1. **multi-scale: 实现了deeper and wider** than other existing deep networks for hyperspectral image classification
2. `FCN: 使用1*1卷积替代全连接网络, 其实就是对**将卷积核上同一点的所有channel进行FC操作**。`
   1. 能不能通过FCN使用少数标记点(结合语义分割)来标记其他像素点进行训练
   2. `看语义分割能不能转化为像素点分类`
   3. 声称是第一次用比较**深的网络**来进行高光谱分类

### 整体流程

<img src="https://img-blog.csdnimg.cn/20190320194547550.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9zaGVybG9jay5ibG9nLmNzZG4ubmV0,size_16,color_FFFFFF,t_70" alt="在这里插入图片描述"  />

较深的网络:

![img](https://img-blog.csdnimg.cn/20190320202016861.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9zaGVybG9jay5ibG9nLmNzZG4ubmV0,size_16,color_FFFFFF,t_70)
训练方法:

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190320211214652.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9zaGVybG9jay5ibG9nLmNzZG4ubmV0,size_16,color_FFFFFF,t_70)



200 training samples

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190320211517900.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9zaGVybG9jay5ibG9nLmNzZG4ubmV0,size_16,color_FFFFFF,t_70)





`有什么缺点?还能加什么?`





`有什么缺点?还能加什么?`





`有什么缺点?还能加什么?`





`有什么缺点?还能加什么?`





`有什么缺点?还能加什么?`





`有什么缺点?还能加什么?`



























# 出现的问题

## [√]undefined有预测, 且准确率不够

![f8839b69-9f1d-4ed4-b7ab-cbf4b5bcad02](/img/in-post/20_07/f8839b69-9f1d-4ed4-b7ab-cbf4b5bcad02.svg)

![f71a8468-014d-406d-8f24-770fefdaf79b](/Volumes/StuFile/Sqh/HSIClassificationCode/%E9%97%AE%E9%A2%98/f71a8468-014d-406d-8f24-770fefdaf79b.svg)

## padding方案问题

https://github.com/eecn/Hyperspectral-Classification/issues/7

> 打扰一下，这个工具箱里用到padding 的方法都有点问题，边缘像素分类结果全是黑色的背景，导致精度低，不知道有没有解决办法

这个问题实际上是深度网络处理的常用方案，只是说高光谱数据对padding更敏感一些<理论上和实际实验中我都试过>，torch、numpy等也有其他的padding方案可以试一试效果。



## 制约准确率的原因主要就是第二类meadows预测为第一类asphalt和第六类bare soil的原因

![5634194f-7ea6-4aa3-9000-0a349d03572a](/img/in-post/20_07/5634194f-7ea6-4aa3-9000-0a349d03572a.svg)





















