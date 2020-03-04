---
layout: post
title: "A CNN With Multiscale Convolution and Diversified Metric for Hyperspectral Image Classification"
subtitle: 'From Vim to Spacemacs'
author: "Sun"
header-style: text
tags:
  - Paper
  - DPP
  - DML
  - CNN
---

•IEEE Transactions on Geoscience and Remote Sensing, 2019, 57(6): 3599-3618.

•提出了一种DPP-DML-MS-CNN方法, 同时利用了**基于DPP的促进多样性的深度度量**和**多尺度特征**，用于高光谱图像分类。
![image-20200302104353207](/img/in-post/20_03/image-20200302104353207.png)

## index

multi-scale feature → MS-CNN

determinantal point process (DPP) 

deep metric learning(DML) 

# MS-CNN

## 提出原因

由于训练样本数量有限，特别是对于具有较大类内方差和低类间方差的图像，学习的深度模型可能不是最优的。

## 创新点

在本文中，提出了具有多尺度卷积（MS-CNN）的新型卷积神经网络，通过从高光谱图像中提取深层多尺度特征来解决这一问题。

设计了三种类型的MS-CNN用于高光谱图像分类。通过将多尺度滤波器组合并到CNN中以从图像中提取多尺度特征的CNN结构。 

![image-20200302112453920](/img/in-post/20_03/image-20200302112453920.png)

![image-20200302112510439](/img/in-post/20_03/image-20200302112510439.png)

![image-20200302112517053](/img/in-post/20_03/image-20200302112517053.png)

# **Deep metric learning** (DML)

但是，在**度量转换**中用一般的结构损失来学习度量参数因子通常会使学习到的因子之间具有**相似性**。这种相似性将负面影响深度度量模型对高光谱图像的表示能力。

•HSI 没有足够标记的样本

•不同类别样本表现相似特征

因此学习到的用于表示HSI的metric factors明显冗余,不同metric factors从图像提取相似特征, 导致分类性能降低

![image-20200302112800844](/img/in-post/20_03/image-20200302112800844.png)

# DPP prior

![image-20200302113857348](/img/in-post/20_03/image-20200302113857348.png)

>  θ在此处是连续空间, B是θ的任意子集
>
>  𝑝(B⊂θ) 表示B中元素采样中被命中的概率, 即B的DPP先验
>
>  λ可以视为权衡参数，该参数衡量多样化惩罚的权重。 二维平面随机采样

•矩阵K 被称作 DPP kernel，是一个 𝑁×𝑁的实对称方阵。

>  K~B~是根据B中的元素从 𝐾中按行按列索引得到的方阵，也即K~B~是 𝐾的主子式
>
>  det(K~B~) 是矩阵K~B~的行列式值。
>
> 𝐾 的元素 𝐾𝑖𝑗 可以看做集合θ中第𝑖,𝑗个元素之间的**相似度** 
>
> 𝐾𝑖𝑖 越大的样本，被采样出来的概率越大
>
> 𝐾𝑖𝑗越大的的两个样本 {𝑖,𝑗} 越相似，
>
>  被同时采样出来的概率越低

二维平面随机采样:

![image-20200302120120989](/img/in-post/20_03/image-20200302120120989.png)

采样概率计算:																					

![image-20200302120126678](/img/in-post/20_03/image-20200302120126678.png)

# Determinantal point process(DPP) based DML

•通过**多样化（diversification）**，不同的因子(factors)倾向于对不同的特征(features)做出反应, 可以根据图像建模更多的特征(features)。因此需要加入促进多样性的先验(diversity-promoting priors)。

•为了使学习到的**度量参数因子多样化**，并进一步提高深度度量的表示能力，提出了一种**基于DPP的结构损失**, 对学习到的度量参数因子施加确定性点过程（DPP）先验，以**鼓励学习到的度量因子相互排斥**。在本文中，使用这种**具有特殊结构损失的深度度量学习方法**，以**联合训练**所提出的模型。

•**结构损失**L用于**最小化所有正对之间的距离,并惩罚了相应的负对**

![image-20200302112957361](/img/in-post/20_03/image-20200302112957361.png)

•特征距离函数:

![image-20200302113240820](/img/in-post/20_03/image-20200302113240820.png)

•负对惩罚函数:

![image-20200302113246170](/img/in-post/20_03/image-20200302113246170.png)

> 其中, m是一个正值，表示对负对距离的惩罚边界。
>  xi, xj: 同一类别的样本对(positive pairs) xi, xk：不同类别样本对(negative pairs)
> 可以注意到，B可以看成是线性映射，可以训练成MS-CNN中的全连接层。

•通过最大后验概率(MAP)估计：![image-20200302122417727](/img/in-post/20_03/image-20200302122417727.png)

>  其中X是训练样本的集合。

•等效对数似然方程可表示为:

![image-20200302122427819](/img/in-post/20_03/image-20200302122427819.png)

•转化为约束优化:

![image-20200302122640399](/img/in-post/20_03/image-20200302122640399.png)

•通过通过拉格朗日乘数，重新化为无约束优化:

![image-20200302122648104](/img/in-post/20_03/image-20200302122648104.png)

•用于训练MS-CNN的**基于DPP的结构损失** :

![image-20200302122652956](/img/in-post/20_03/image-20200302122652956.png)

•通过随机梯度下降方法联合训练MS-CNN和深度度量。训练过程可以看作是最小化基于DPP的结构损失。W表示MS-CNN中的参数, B表示度量参数因子。可以通过以下方式估算所提出模型中的参数

![image-20200302123010188](/img/in-post/20_03/image-20200302123010188.png)

•基于DPP的结构化损失L相对于度量因子B的梯度可以计算为: 

![image-20200302123016706](/img/in-post/20_03/image-20200302123016706.png)

•从流程图可以看出, 基于DPP的结构损失的学习模型用于从高光谱图像中提取特征**，**然后使用Softmax分类器用于分类**。**

•该模型通过确定性点过程（DPP）先验而非独立先验来使深度度量多样化, 开发了一种具有**多尺度卷积和多样化度量的****CNN**，以获得高光谱图像分类的判别特征。

![image-20200302123035175](/img/in-post/20_03/image-20200302123035175.png)

# Result

## A.Computational Performance

![image-20200302123208238](/img/in-post/20_03/image-20200302123208238.png)

> For pre-training procedure:
>
>  the training epoch, the base learning rate, the weight decay, and the momentum value 
>
> were set to 20000, 0.0001, 5e-5, and 0.9, respectively.
>
> For fine-tuning procedure: 
>
> the training epoch, the base learning rate, the weight decay, and the momentum value 
>
> were set to 1000, 1e-5, 5e-5, and 0.9, respectively.

## **B. Model’s Diversity**

![image-20200302123314790](/img/in-post/20_03/image-20200302123314790.png)

## **C. Effects of Diversity Weight** **λ**

![image-20200302123333473](/img/in-post/20_03/image-20200302123333473.png)

> 从四个数据集的趋势可以看出，λ值越大，分类精度越高，而λ值过大会导致性能下降。
>
> 在实际应用中，采用交叉验证技术来选择合适的λ来满足不同任务的特殊要求。

## **D. Effects of Neighbor Size**  

![image-20200302123402241](/img/in-post/20_03/image-20200302123402241.png)

## **E. Classification Results and Classification Maps**

## ![image-20200302123408751](/img/in-post/20_03/image-20200302123408751.png)

![image-20200302123432385](/img/in-post/20_03/image-20200302123432385.png)

