---

layout: post
title: "CNN知识点总结"
subtitle: ''
author: "Sun"
header-style: text
tags:
  - CNN
  - 
---

# 预处理

## PCA

一般用pca降维 是因为网络比较复杂 降低维度 防止维度灾难





3d 就比较复杂 可以用

但是用了 也可能会使精度降低



## 放射变换网络

![image-20200401150058867](/img/in-post/20_03/image-20200401150058867.png)



## 应用/数据集

**水体提取**

> 全卷积神经网络用于遥感影像**水体提取**[J].http://html.rhhz.net/CHTB/html/2018-6-41.htm

## CNN结构



#### FCN

这些典型CNN结构适合于图像级的分类和回归任务。由于最后得到的是整个输入图像的一个数值描述的**概率值**，它会将原来二维的矩阵(图像)压扁成一维的，从而使**后半段没有空间信息**。那么, 如何从抽象的特征中恢复出每个像素的所属类别，即从图像级别的分类进一步延伸到像素级别的分类，Jonathan Long等提出了全卷积神经网络(FCN)[[12](http://html.rhhz.net/CHTB/html/2018-6-41.htm#b12)]。

<img src="http://html.rhhz.net/CHTB/html/PIC/chtb-2018-6-41-1.jpg" alt="chtb-2018-6-41-1" style="zoom:50%;" />

产生像素热力图, 常用于物体检测, **语义分割**

**传统的基于CNN的分割方法**

为了对一个像素分类，使用该像素周围的一个图像块作为CNN的输入用于训练和预测。这种方法有几个缺点:

一是存储开销很大

二是计算效率低下

三是像素块大小的限制了感知区域的大小

而全卷积网络(FCN)则是从抽象的特征中恢复出每个像素所属的类别。即从图像级别的分类进一步延伸到像素级别的分类。

## Inception结构(多尺度卷积核)

Inception V1的论文中指出，Inception Module可以让网络的深度和宽度高效率地扩充，提升准确率且不致于过拟合。

![深度学习之四大经典CNN技术浅析 | 硬创公开课](https://static.leiphone.com/uploads/new/article/740_740/201702/58b53ff73e987.png?imageMogr2/format/jpg/quality/90)

Inception-ResNet-V2则是把这几种网络的优势都融为一体。

## 多分支结构(一开始取不同大小的输入块)