---
layout: post
comments: true
title: "(HInton|CL)A Simple Framework for Contrastive Learning of Visual Representations"
subtitle: '20年Deep Learning先驱Hinton最新论文学习及对SimCLR解读'
author: "Sun"
header-style: text
tags:
  - Contrastive Learning
  - Master
---

**Contrastive Learning** 

20_(Hinton|CL)A Simple Framework for Contrastive Learning of Visual Representations

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

## 优势

SimCLR 是一种简单而清晰的方法，无需类标签即可让 AI 学会视觉表示，而且可以达到有监督学习的准确度。

它不仅优于此前的所有工作，也优于最新的对比自监督学习算法，而且结构更加简单：**既不需要专门的架构，也不需要特殊的存储库。**

<img src="/img/in-post/20_03/image-20200319091344678.png" alt="image-20200319091344678" style="zoom:50%;" />

> *SimCLR 与此前各类自监督方法在 ImageNet 上的 Top-1 准确率对比（以 ImageNet 进行预训练），以及 ResNet-50 的有监督学习效果（灰色×）*

基于这些发现，他们在 **ImageNet ILSVRC-2012** 数据集上实现了一种新的**半监督、自监督学习** SOTA 方法——SimCLR。在线性评估方面，SimCLR 实现了 76.5% 的 top-1 准确率，比之前的 SOTA 提升了 **7%**。在仅使用 1% 的 ImageNet 标签进行微调时，SimCLR 实现了 85.8% 的 top-5 准确率，比之前的 SOTA 方法提升了 **10%**。在 12 个其他自然图像分类数据集上进行微调时，SimCLR 在 10 个数据集上表现出了与强监督学习基线相当或更好的性能。





The president is turning to racist rhetoric to distract from his failures to take the coronavirus seriously early on, make tests widely available, and adequately prepare the country for a period of crisis. 

Hillary, 3 comments on your statement: -The Virus originated in China so it's a Chinese virus. Are you REALLY this dumb? -Every person who knows dirt on you or Bill end up in the graveyard. -You're STILL a sore loser. You're really not it a position of moral authority. 



