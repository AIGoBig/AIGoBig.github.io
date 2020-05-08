---
layout: post
comments: true
mathjax: false
title: "Self-supervised Hyperspectral Image Restoration using Separable Image Prior(利用可分图像先验进行自监督高光谱图像复原)"
subtitle: '19_arxiv_(0)'
author: "Sun"
header-style: text
tags:
  - Self-supervised
  - Hyperspectral Image Restoration
  - structual prior
---

![image-20200507095733356](/img/in-post/20_03/image-20200507095733356.png)

### Abstract

用于图像恢复的监督学习深度学习方法取得了很好的效果, 但是其在像HSI这种非灰度/彩色图像上效果不佳. 

我们提出了一个新型的自监督学习策略, 其通过一个被破坏图像自动产生一个训练集, 并使用干净图片生成一个去噪网络, 

另一个值得注意的地方是我们方法中使用了一个可分离的卷积层, 实验证明其可以获得更多的HSI先验用以进行图像恢复

### Introduction

Hyperspetral Image Restoration -- 用degraded image 产生 clear image 的过程



![image-20200507114728036](/img/in-post/20_03/image-20200507114728036.png)

> 自监督恢复方法: 通过使用网络将加了噪声的Noisy input作为输入, 恢复为原Noisy input

![image-20200507124946066](/img/in-post/20_03/image-20200507124946066.png)

> (左)HSI的Tucker分解，(右)各模式的奇异值图(以对数尺度表示)