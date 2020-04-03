---
layout: post
comments: true
title: "(Hinton|CL)A Simple Framework for Contrastive Learning of Visual Representations"
subtitle: '20_CVPR_Deep Learning先驱Hinton最新论文学习及对SimCLR解读'
author: "Sun"
mathjax: true
header-style: text
tags:
  - Contrastive Learning
  - metric
  - Master
  - VR
  - CVPR
  - data augmentation
  - paper
---

20_CVPR_(Hinton|CL)A Simple Framework for Contrastive Learning of Visual Representations

## Index

**Contrastive Learning**

Visual Representation 

## 收获

与度量学习结合, 训练网络, 

1. ==无监督的方式学习表示网络(度量网络),== 
2. 少量样本微调网络
3. 卷积神经网络提取特征

样本增强方式

1. 随机裁剪
2. ==颜色增强==



## Abstract

> This paper presents SimCLR: a simple framework for contrastive learning of visual representations. 
> We simplify recently proposed contrastive self- supervised learning algorithms without requiring specialized architectures or a memory bank. In order to understand what enables the contrastive prediction tasks to learn useful representations, we systematically study the major components of our framework. We show that (1) composition of data augmentations plays a critical role in defining effective predictive tasks, (2) introducing a learnable nonlinear transformation between the representation and the contrastive loss substantially improves the quality of the learned representations, and (3) contrastive learning benefits from larger batch sizes and more training steps compared to supervised learning. By combining these findings, we are able to considerably outperform previous methods for self-supervised and semi-supervised learning on ImageNet. A linear classifier trained on self-supervised representations learned by Sim- CLR achieves 76.5% top-1 accuracy, which is a 7% relative improvement over previous state-of- the-art, matching the performance of a supervised ResNet-50. When fine-tuned on only 1% of the labels, we achieve 85.8% top-5 accuracy, outperforming AlexNet with 100× fewer labels.

本文介绍了SimCLR：用于视觉表示的对比学习的简单框架。

我们简化了最近提出的的对比自监督学习算法, 使其**无需专门的架构或存储库** 。为了了解什么使对比预测任务能够学习有用的表示形式，我们系统地研究了框架的主要组成部分。

我们证明

（1）**多个数据扩充方法的组合**对于有效的预测任务起着至关重要的作用，且数据增强相比于有监督学习, 其对于无监督学习更加有用.

（2）==在表示和对比损失之间引入**可学习的非线性变换**，可以大幅度提高模型学到的表示的质量，== 

（3）与有监督的学习相比，对比学习**得益于更大的批处理数量和更多的训练次数**。

通过结合这些发现，我们能够大大胜过ImageNet上用于自我监督和半监督学习的先前方法。 **使用通过Sim-CLR学习到的自监督表示来训练线性分类器**，可达到76.5％的top-1准确性，与以前的最新技术相比，相对**提高了7％**，==与监督的ResNet-50的性能相匹配。当**使用1％的标签进行微调时**，我们的top-5准确性达到了85.8％，比AlexNet少了100倍的标签?????==

#### Q: 到底使用了多少标签和准确率

#### Q: 自监督和无监督的区别

#### Q: 怎么理解"**既不需要专门的架构，也不需要特殊的存储库。**"

#### ==Q: 无监督表示学习??/啥意思==

A: 只是学习一个视觉表示, 然后再用线性分类器分类?

## 评价

SimCLR 是一种简单而清晰的方法，无需类标签即可让 AI 学会视觉表示，而且可以达到有监督学习的准确度。

它不仅优于此前的所有工作，也优于最新的对比自监督学习算法，而且结构更加简单：**既不需要专门的架构，也不需要特殊的存储库。**

<img src="/img/in-post/20_03/image-20200319091344678.png" alt="image-20200319091344678" style="zoom:50%;" />

> *SimCLR 与此前各类自监督方法在 ImageNet 上的 Top-1 准确率对比（以 ImageNet 进行预训练），以及 ResNet-50 的有监督学习效果（灰色×）*

基于这些发现，他们在 **ImageNet ILSVRC-2012** 数据集上实现了一种新的**半监督、自监督学习** SOTA 方法——SimCLR。在线性评估方面，SimCLR 实现了 76.5% 的 top-1 准确率，比之前的 SOTA 提升了 **7%**。在仅使用 1% 的 ImageNet 标签进行微调时，SimCLR 实现了 85.8% 的 top-5 准确率，比之前的 SOTA 方法提升了 **10%**。在 12 个其他自然图像分类数据集上进行微调时，SimCLR 在 10 个数据集上表现出了与强监督学习基线相当或更好的性能。

无监督学习的快速发展让科学家们看到了新的希望，==DeepMind 科学家 Oriol Vinyals 表示：感谢对比损失函数，无监督学习正在逼近监督学习！==



## 整体流程

受到最近对比学习算法（contrastive learning algorithm）的启发，SimCLR 通过**隐空间中的对比损失来最大化同一数据示例的不同增强视图之间的一致性，从而学习表示形式**。具体说来，这一框架包含**四个主要部分**：

- **随机数据增强模块**，可随机转换任何给定的数据示例，从而**产生同一示例的两个相关视图，分别表示为 $x_i$ 和 $x_j$，我们将其视为正对**；
- 一个**基本的神经网络编码器 $f(·)$**，从增强数据中==**提取表示向量**(类似度量因子???)；==
- 一个小的**神经网络投射头（projection head）$g(·)$**，**将表示映射到对比损失的空间**；
- 为对比预测任务定义的**对比损失函数**。

<img src="https://image.jiqizhixin.com/uploads/editor/f4bfa740-d1e5-4f61-bc64-bbca43f25686/640.png" alt="img" style="zoom:67%;" />

论文的作者之一，谷歌资深研究科学家 Mohammad Norouzi 的**简化总结**: 

- 随机抽取一个小批量
- 给每个例子绘制两个独立的增强函数
- 使用两种增强机制，为每个示例生成两个互相关联的视图
- 让相关视图互相吸引，同时排斥其他示例

<img src="https://image.jiqizhixin.com/uploads/editor/447eb3c3-1342-49bb-9542-3a870150560b/640.jpeg" alt="img" style="zoom: 33%;" />

SimCLR 的主要**学习算法**如下：

<img src="https://image.jiqizhixin.com/uploads/editor/6d605f25-111c-4c07-8d5e-920a101e60d5/640.jpeg" alt="img"  />

注意: 

不同且独立的增强函数$t(x)$

编码器网络(encoder network) $f(·)$, ==**投射头网络（projection head）$g(·)$**???==为可训练的神经网络, 其中每个 f 和每个 g 都是一样的

==**利用对比损失来最大化一致性(maximize agreement)训练f, g**, 训练完成后，我们将投影头g 拿走，并使用**编码器网络f**和represention **h** 进行后面阶段的任务。==

**用更大的批大小进行训练**

作者将训练批大小 N 分为 256 到 8192 不等。批大小为 8192 的情况下，增强视图中每个正对（positive pair）都有 16382 个反例。当使用标准的 SGD/动量和线性学习率扩展时，大批量的训练可能不稳定。为了使得训练更加稳定，研究者在所有的批大小中都采用了 ==LARS 优化器???==。==**他们使用 Cloud TPU 来训练模型**，根据批大小的不同，使用的核心数从 32 到 128 不等/。==

#### Q: 我们的服务器规模好像不够, 可以使用降维或者什么操作??==

#### Q: 提取表示向量, 类似度量因子???

#### Q: ==LARS 优化器???==

**数据增强**

虽然数据增强已经广泛应用于监督和无监督表示学习，但它还没有被看做一种定义对比学习任务的系统性方法。许多现有的方法通过改变架构来定义对比预测任务。

本文的研究者证明，通过对目标图像执行**简单的随机裁剪（调整大小）**，可以避免之前的复杂操作，从而创建包含上述两项任务的一系列预测任务，如图 3 所示。**这种简单的设计选择方便得将预测任务与其他组件（如神经网络架构）解耦**。

随机裁剪

<img src="https://image.jiqizhixin.com/uploads/editor/7fc95486-9729-4459-ab38-904fa3de8e57/640.png" alt="img" style="zoom:50%;" />



#### Q: 将预测任务与其他组件（如神经网络架构）解耦????。

## 多种数据增强操作的组合是学习良好表示的关键

<img src="https://image.jiqizhixin.com/uploads/editor/8b86985a-f53a-4826-8dd5-291d10db8022/640.jpeg" alt="img" style="zoom: 50%;" />

==颜色增强的重要性???, 如下调整了颜色的强度==

<img src="https://image.jiqizhixin.com/uploads/editor/a1350b5a-59e3-4562-80ae-ba6199c70b6b/640.jpeg" alt="img" style="zoom: 33%;" />

**随参数增加Top1变换**

如图 7 所示，增加深度和宽度都可以提升性能。监督学习也同样适用这一规律。但我们发现，随着模型规模的增大，监督模型和在无监督模型上训练的线性分类器之间的差距会缩小。这表明，与监督模型相比，无监督学习能从更大规模的模型中得到更多收益。

<img src="https://image.jiqizhixin.com/uploads/editor/caf6b2b5-fb26-4011-9fda-19a1b580ffc0/640.jpeg" alt="img" style="zoom:50%;" />

**非线性的投射头**可以改善之前的层的表示质量，图 8 展示了使用三种不同投射头架构的线性评估结果。

<img src="https://image.jiqizhixin.com/uploads/editor/b3d4e578-c2b3-4bba-a76c-a156b9528cd5/640.png" alt="img" style="zoom:50%;" />

 **损失函数和批大小**

可调节温度的归一化交叉熵损失比其他方法更佳。研究者对比了 NT-Xent 损失和其他常用的对比损失函数，比如 logistic 损失、margin 损失。==表 2 展示了**目标函数和损失函数输入的梯度**????。==

![640](https://image.jiqizhixin.com/uploads/editor/b6cfc463-c1cd-484f-a7fb-9c1a75e93f2e/640.jpeg)

对比学习（Contrastive learning）**能从更大的批大小和更长时间的训练中受益更多**。图 9 展示了在模型在不同 Epoch 下训练时，不同批大小所产生的影响。 



**与当前最佳模型的对比**

**线性估计**

表 6 显示了 SimCLR 与之前方法在线性估计方面的对比。此外，上文中的表 1 展示了不同方法之间更多的数值比较。从表中可以看出，用 SimCLR 方法使用 ResNet-50 (4×) 架构能够得到与监督预训练 ResNet-50 相媲美的结果。

<img src="https://image.jiqizhixin.com/uploads/editor/62861fdb-95e5-4e2c-9782-1a3d1a0dd497/640.jpeg" alt="img" style="zoom:50%;" />

**半监督学习**

下表 7 显示了 SimCLR 与之前方法在**半监督学习**方面的对比。从表中可以看出，无论是**使用 1% 还是 10% 的标签**，本文提出的方法都显著优于之前的 SOTA 模型。

<img src="https://image.jiqizhixin.com/uploads/editor/8336d657-a4d2-4d13-b5a1-958b230f6057/640.jpeg" alt="img" style="zoom:50%;" />

**迁移学习**

研究者在 12 个自然图像数据集上评估了**模型的迁移学习性能**。下表 8 显示了使用 ResNet-50 的结果，与监督学习模型 ResNet-50 相比，SimCLR 显示了良好的迁移性能——两者成绩互有胜负。

![img](https://image.jiqizhixin.com/uploads/editor/86321c8b-362f-44e2-8106-389259185911/640.jpeg)

 



的对比。此外，上文中的表 1 展示了不同方法之间更多的数值比较。从表中可以看出，用 SimCLR 方



的对比。此外，上文中的表 1 展示了不同方法之间更多的数值比较。从表中可以看出，用 SimCLR 方

## Reference

[Hinton组力作：ImageNet无监督学习最佳性能一次提升7%，媲美监督学习](https://www.jiqizhixin.com/articles/2020-02-15-3)

