---
layout: post
comments: true
mathjax: false
title: "Deep Learning for Hyperspectral Image Classification: An Overview"
subtitle: '19_TGRS_(55)'
author: "Sun"
header-style: text
tags:
  - overview
  - 
  - 
---

## Abstract

摘要—高光谱图像（HSI）分类已成为遥感领域的热门话题。总的来说，高光谱数据的复杂特性使得这种数据的准确分类对于传统的机器学习方法而言具有挑战性。此外，高光谱成像通常处理捕获的光谱信息和相应材料之间的固有非线性关系。近年来，深度学习已被公认为是**有效解决非线性问题**的强大功能提取工具，并已广泛用于许多图像处理任务中。受这些成功应用程序的激励，还引入了深度学习来对HSI进行分类并表现出良好的性能。本调查论文对基于深度学习的HSI分类文献进行了系统综述，并比较了该主题的几种策略。具体而言，我们首先总结了传统机器学习方法无法有效克服的HSI分类的主要挑战，并介绍了深度学习解决这些问题的优势。然后，我们建立了一个框架，将相应的工作分为**频谱特征网络**，**空间特征网络**和**频谱空间特征网络**，以系统地回顾基于深度学习的HSI分类的最新成果。
此外，考虑到以下事实：遥感领域中可用的**训练样本通常非常有限**，而训练深度网络需要大量样本，因此，我们提供了一些==**提高分类性能的策略**==，这可以为该主题的未来研究提供一些指导。最后，在我们的实验中，对真实的HSI进行了几种基于深度学习的代表性分类方法。

一方面，本文有望阐明现有**基于深度学习的分类方法背后的机制。**在框架内，我们系统地回顾了大量文献，其中，根据提取的特征类型，将其分为光谱特征网络，空间特征网络和光谱空间特征网络。采用的深层网络。
另一方面，我们打算包括一些策略来**处理可用样本有限的问题**（即上述第二个挑战），这对于设计用于HSI分类的深度学习方法非常重要。

## Introduction

高光谱分类的2大**挑战**：
（1）spectral signatures的空间变异大；

（2）样本数量有限，而高光谱数据维数高。

传统机器学习方法的不足：
（1）依赖于hand-crafted or shallow-based descriptors，而人工特征是为了**完成特定的任务而设计的**，并且在参数初始化阶段需要专家知识，这限制了模型应用的场景。
（2）人工特征的表示能力不足以识别不同类间的小变异和类内的大变异。

深度学习的优势：
（1）能够提取信息丰富的特征
（2）深度学习是自动的，所以模型应用的场景更加灵活。

## 深度模型:

### stacked autoencoders (SAEs)

**AE**

AE是SAEs的基本单元，AE网络由input layer（x），hidden layer（h），output layer（y）构成。由x到h的过程称为“**encoder**”，由h到y的过程称为“decoder”。我们注重“encoder”这个过程，希望将x做某种变换后得到的h可以最大程度复原为y，这样就提取出了x的某种特征h。

<img src="/img/in-post/20_03/image-20200514102908497.png" alt="image-20200514102908497" style="zoom:50%;" />

> 自编码网络AE

**SAE**

[自编码器与堆叠自编码器简述](https://blog.csdn.net/u013550000/article/details/80747109)

训练完AE提取出h特征以后，<u>我们会去掉output layer这一层</u>，然后进行<u>堆叠</u>，以便提取出更深层的特征。<u>最后加上一个Logistic regression classifier</u>，便得到SAEs。

<u>输入pixel vector，输出class labels</u>，便可将SAEs视为一个spectral classifier。

<img src="/img/in-post/20_03/image-20200514103204765.png" alt="image-20200514103204765" style="zoom:50%;" />

#### Q: 自编码器变形形式?

![img](https://img-blog.csdn.net/20160530044431469?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

### deep belief networks (DBNs), 

**RBM**

<img src="/img/in-post/20_03/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwNzIxMzM3,size_16,color_FFFFFF,t_70.png" alt="RBM" style="zoom:67%;" />

**DBNs**

DBN是由多层RBM构成的，可看作生成模型或判别模型。加上Logistic regression classifier即构成一个spectral classifier。

<img src="/img/in-post/20_03/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwNzIxMzM3,size_16,color_FFFFFF,t_70-20200514113225493.png" alt="DBN" style="zoom:67%;" />



### convolutional neural networks (CNNs), 

### recurrent neural networks (RNNs), 

RNNs中隐藏层中的结点是有连接的，隐藏层的输入不仅包括输入层的输出还包括上一时刻隐藏层的输出。所以RNNs适合于处理与时间序列相关的问题。

又因为在光谱空间中，高光谱数据的<u>每一个pixel vector能被看作一系列有序而连续的光谱序列，所以RNNs适合于高光谱分类。</u>

对于梯度消失或者梯度爆炸的问题，也提出了RNNs的**改进算法**：LSTM 和 GRU。

###  generative adversarial networks (GANs)

<img src="/img/in-post/20_03/image-20200514151431037.png" alt="image-20200514151431037" style="zoom:50%;" />

> GAN模型

模型可大体上分为2类：生成模型和判别模型。

生成模型从数据中**学习到参数分布**，然后根据已学习的模型**产生新样本**；

判别模型关注数据之间的差异，根据样本数据建立一个由x到y的映射，再根据这个映射进行预测。

生成模型通过噪音生成以假乱真的false data，通过判别模型判断true data和false data来**提升 生成模型生成假数据的能力。**



<img src="/img/in-post/20_03/image-20200514152147571.png" alt="image-20200514152147571"  />

[博客 Generative Adversarial Networks for Hyperspectral Image Classification](https://blog.csdn.net/a601165974/article/details/105640549)

> 在提出的GAN中，设计了一个卷积神经网络(convolutional neural network, CNN)对输入进行区分，并使用另一个CNN生成所谓的假的输入。 上述提到的两个CNN是一起训练的: 生成式CNN试图生成尽可能真实的虚假输入，而有鉴别力的CNN则试图对真实和虚假输入进行分类。**这种对抗训练提高了判别CNN的泛化能力，这在训练样本有限的情况下是非常重要的。**
>
> 此外, 将生成的对抗样本与真实训练样本结合使用，对判别CNN进行微调，提高了最终的分类性能。
>
> 我们的网络是根据ACGAN理论形成的，而目标函数则根据正确来源的可能性和正确HSI类的可能性进行了修改。该算法根据多分类损失对参数进行优化，较传统的遗传算法能更合理地优化损失函数。将标记信息作为发生器和鉴别器的输入，**鉴别器D包含一个辅助解码器网络，可以输出各自的训练数据标签**。显然，额外的标签信息可以同时利用鉴别器D的分类能力和生成器g的生成能力。
>
> 该方法不仅可以对真实输入和虚假输入进行分类，**而且可以对输入数据的相应标签进行预测。**G的形式是小型的CNN, D的形式是CNNs。最后，使用sigmoid分类器和softmax分类器并行进行分类，分别对真实/虚假样本和HSIs进行分类。
>
> 随着训练过程的继续，当G可以生成与真实数据最相似的假数据，D无法区分假数据与真实数据时，理论上G和D都会得到最优的结果。这样，我们可以证明整个网络达到了纳什均衡条件，两个网络之间的敌对行为和竞争可以提高分类性能。因此，GAN的关键思想在于对抗性训练，通过不断的竞争，我们可以获得优于传统CNN方法的分类结果。

一个基于gan的HSI分类框架如图8所示。从图8可以看出，除噪声z外，发生器G**还接受HSI类标签c**, G的输出可以用xfake = G(z)来定义。**将带有相应类标签的训练样本和G生成的伪数据作为判别器d的输入**。

目标函数包括2部分: 正确的输入数据源的对数似然 Ls 和正确的类标签的对数似然 Lc, **D 最大化Ls+Lc, G最大化Lc-Ls**



## 基于深度网络的的HSI分类

![image-20200514183957888](/img/in-post/20_03/image-20200514183957888.png)

光谱空间特征网络的范型，根据特征融合阶段可进一步分为三类:、基于预处理的网络、集成网络和基于后处理的网络。





### Spectral-Feature Networks：

  早期，用原始的spectral vector直接以非监督方式训练SAE或DBN。
  后来，提出deep learning和active learning结合的分类框架，其中DBN用来提取deep spectral feature，**深度学习算法用来选择高质量的训练样本待人标注**；提出diversified DBN，即正则化DBN的预处理和微调程序。
  此外，1-D CNN，1-D GAN，RNN也被用于提取spectral feature；在spectral domain中执行conv提取pixel-pair features（PPFs）
  最后，用字典学习训练深度网络被reformulated。

### Spatial-Feature Networks：

  学习到的spatial features将会和通过别的特征提取的方法提取出的spectral features融合在一起，来进行更精确的高光谱分类。
  **PCA（使原始数据降维）**+2-D CNN（提取空间领域中的spatial information）；
  采用sparse representation将CNN提取到的deep spatial feature进行编码，**使其变成低维的sparse representation；**
  直接采用AlexNet和GoogleNet的CNNs来提取deep spatial feature；
  SSFC框架，即用balanced local discriminant embedding（BLDE）和CNN来提取spectral and spatial features，然后将特征融合来训练multiple-features-based classifier；
  multiscale spatial-spectral feature extraction algorithm，训练好的FCN-8被用来提取deep multiscale spatial features，然后采用加权融合法融合original spectral features和deep multiscale spatial features，最后将融合的特征放入classifier中；

### Spectral-Spatial-Feature Networks：

  这种网络不是用来提取spectral features或者spatial features，而是用来**提取joint deep spectral-spatial features。**
  获取joint deep spectral-spatial features有3种方式：（1）通过深度网络将low-level spectral-spatial features映射为high-level spectral-spatial features；（2）直接从原始数据或几个主成分中提取deep spectral-spatial features；（3）融合2个deep features，即deep spectral features和deep spatial features。
  **因此根据将光谱信息和空间信息融合的处理阶段不同**，将spectral-spatial-feature networks也分为3类：preprocessing-based networks，integrated networks，postprocessing-based networks。

 **（1）preprocessing-based networks**
  在连接deep network之前spectral features和spatial features就已经融合了。
  处理步骤：①low-level spectral-spatial feature fusion；②用deep networks提取high-level spectral-spatial feature；③将deep SSFC与simple classifier（如SVM、ELM极限学习机、多项式Logistic回归）连接起来。
  由于全连接网络（如DBN、SAE等）只能处理1维输入，所以我们想到把空间邻域reshape为1维向量，与1维的光谱信息叠加之后放入全连接网络；
  将一个空间邻域内所有像元的光谱信息取平均变为一个光谱向量，这个平均光谱向量其实就包括了空间信息在内，我们再将其放入接下来的deep network中；
  也有一些不是获取邻域内的spatial information的滤波方法（如用Gobar滤波、attribute滤波、extinction滤波、rolling guidance滤波来提取更有效的spatial features），它们结合了deep learning和spatial-feature提取方法。

**（2）integrated networks**
  不是分别获取spectral features和spatial features再融合，而是**直接获取joint deep spectral-spatial features。**
  通过2-D CNN或3-D CNN在原始数据中直接提取joint deep spectral-spatial features；
  FCN（fully convolutional network）能够通过监督方式或非监督方式学习到高光谱的deep features；
  RL（residual learning）可以帮助建立deep和wide的网络，一个残差单元中输出是输入和输入的卷机值的和；
  用3-D GAN来提取特征，其中一个CNN是判别模型，另一个CNN是生成模型；
  修正CapsNets作为一个spectral-spatial capsules来提取特征；
------------------------------------------------提出hybrid deep networks------------------------------------------------
  three-layer stacked convolutional autoencoder可通过未标记的像元来学习generalizable features；
  convolutional RNNS（CRNNs）中用CNN从高光谱序列中提取middle-level features，用RNN从middle-level features中提取contexture information；
  用spectral-spatial cascaded RNN model来提取特征。

**（3）postprocessing-based networks**
  处理步骤：①用2个deep networks提取deep spectral features和deep spatial features；②在全连接层将2种特征融合产生seep spectral-spatial features；③将deep SSFC与simple classifier（如SVM、ELM极限学习机、多项式Logistic回归）连接起来。
  第①步中的2个deep networks可能共享权重，也可能不共享。
  **1-D CNN用来提取spectral features，2-D CNN用来提取spatial features**。然后这2种特征被融合并与全连接层连接提取spectral-spatial特征用于分类；
  对于每个输入像元，用堆叠去噪自编码器编码提取spectral features。对于相应的图片块，用deep CNN提取spatial features，然后2个预测概率被融合；
  2个主成分分析网络用来提取spectral features，2个堆叠稀疏自编码器用来提取spatial features+全连接层+SVM；
  multiple CNNs同时处理multiple spectral sets（共享权值），each CNN提取相应的spatial feature+在全连接层里将individual spatial features融合起来。 有限样本的策略

## 有限样本应对策略

## 有限样本的策略

### 数据增强(Data Augmentation)

DA通过training samples产生virtual samples，包括2种方式：（1）transform-based sample generation；（2）mixture-based sample generation
（1）transform-based sample generation
  由于同类物体在不同的光照条件下会有不同的辐射率，因此我们可以通过现有训练样本的旋转、镜像操作来产生新数据。
（2）mixture-based sample generation
  通过2个同类样本的线性组合即可产生1个新样本。

### 迁移学习(Transfer Learning)

TL可以从其他已训练好的网络中复制参数来初始化参数。directly transfer the network parameters of low and middle layers to a new network that has the same architecture as the previous network。
  top layers的参数还是随机初始化的，以便处理特定的问题。
  一旦迁移了网络参数，后面的分类也可以被分为非监督和监督方法。非监督方法则直接用迁移网络提取的特征进行分类；非监督分类还要加入少量training samples被fine-tuned。

![transfer learning](/img/in-post/20_03/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQwNzIxMzM3,size_16,color_FFFFFF,t_70-20200514210814300.png)

### 无监督/半监督特征学习(Unsupervised/Semisupervised Feature Learning)

非监督特征学习只用到无标签数据集，可以把它当做一个encoder-decoder过程。半监督分类特征学习可以通过迁移训练好网络的参数和标记数据集的微调来提取特征。
  用全连接网络来进行HSIs分类，其中网络的训练分为非监督式的pretraining和fine-tuning with labeled data。在pretraining时期，参数的学习可以看成一个encoder-decoder过程（unlabled data–intermediate feature–reconstruction）。但是这样比较低效，所以一个unsupervised end-to-end training framework被提出，它将卷积网络视为encoder，将解卷积网络视为decoder。
  GAN也被用来构建半监督特征学习框架。
  用大量unlabeled data和用无参数贝叶斯聚类算法获得的pseudo labels来预训练CRNN。

### 网络优化(Network Optimization)

通过采取更有效的模块或函数，网络优化可以改善网络性能。
  ReLU激活函数可以缓解训练过程中的过拟合。
  RL有2中映射方式：identity mapping、residual mapping。通过引入残差函数G（X）=F（X）-X使得F（X）=X转化为G（X）=0。
  正则化pre-training和fine-tuning程序可以改善DBN的性能； 
  SAE训练中加入了label consistency constraint；
  考虑样本之间的相关性。

## 实验

![20190818084846846](/img/in-post/20_03/20190818084846846.png)

### Effectiveness Analysis of Strategies for Limited Samples

  我们建立了一个简单的CNN并分别运用3种策略（DA、TL、RL）来检验其有效性。
  CNN-Original、CNN-DA、CNN-TL、CNN-RL被用于Salinas数据集，其中training sets分别包括5、10、15、20、25、30个标记样本/类，其他的标记样本就被当做测试集数据了。
  CNN-TL中的CNN在Indian Pines中预训练，因为Indian Pines和Salinas images有一样的传感器。
  观察分类结果，我们发现这3种策略均能改善OA，CNN-RL在大部分情况都表现得最好，这表明了RL是一个非常有用的网络优化方法。

## 结论

通过不同方法获得的分类精度表明，基于**深度学习**的方法总体上优于基于非深度学习的方法，而结合了**RL和特征融合的DFFN**则实现了最佳的分类性能。

此外，**可视化**了深层功能和网络权重，这对于分析网络性能和进一步设计深层架构很有用。

此外，考虑到遥感中可用的训练**样本通常非常有限**，而训练深度网络需要大量样本这一事实，我们还包括一些提高分类性能的策略。我们还进行了实验，以验证和比较这些策略的有效性。最终结果表明，**RL**在所有方法中获得了最高的改进。该实验结果可能为将来对该主题的研究提供一些指导。



首先，介绍了几个在HSIs分类中常用的deep models——SAE、DBN、CNN、RNN、GAN、
  然后，将deep learning-based分类方法分为3类：spectral-feature networks、spatial-feature networks、spectral-spatial networks。
  其次，我们比较了深度学习分类和传统分类方法的分类精度，发现深度学习分类方法总体都比传统分类方法表现更好，其中结合了RL和特征融合的DFFN方法表现得最好。
  还有，deep features和network weights被可视化。
  此外，我们验证了针对样本量少提出的几个策略，发现使用RL能够更好地改善网络性能。

