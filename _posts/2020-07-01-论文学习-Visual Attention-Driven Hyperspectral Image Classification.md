---
layout: post
comments: true
mathjax: false
subtitle: ''
author: "Sun"
header-style: text
tags:
  - attention
  - paper
  - 
---

# Reference

Haut J M , Paoletti M E , Plaza J , et al. Visual Attention-Driven Hyperspectral Image Classification[J]. IEEE Transactions on Geoscience and Remote Sensing, 2019, PP(99):1-16.

https://blog.csdn.net/qq_40721337/article/details/106408155

# 贡献点

  ①在ResNet中引入了视觉注意力机制（新网络叫做A-ResNet）。A-ResNet的基础模块为dual data-path attention module，该模块同时考虑了bottom-up和top-down两种成分。
  ②考虑了在引入噪声的情况下，A-ResNet和ResNet各自的表现。

# 内容

​	   跟ResNet类似，A-ResNet也采用了一个基础模块——attention module（A(l)），包括“the trunk branch”和“the mask branch”两个分支。
  输入X(l-1)，经过“attention module”（A(l)）后，输出与X(l-1)光谱和空间维度一致的X(l)，其中X(l)=(1+X(lmask))*X(ltrunk)。
