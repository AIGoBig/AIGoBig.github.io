---
layout: post
author: "Sun"
header-style: text
tags:
  - Paper
---

Fuzzy C-Means Clustering With **Local Information** and **Kernel Metric** for Image Segmentation(基于使用本地信息和核度量的模糊C-均值聚类的图像分割)

13_IEEE TRANSACTIONS ON IMAGE PROCESSING

> Abstract—In this paper, we present an improved fuzzy C-means (FCM) algorithm for image segmentation by introducing a tradeoff weighted fuzzy factor and a kernel metric. The tradeoff weighted fuzzy factor depends on the space distance of all neighboring pixels and their gray-level difference simultaneously. By using this factor, the new algorithm can accurately estimate the damping extent of neighboring pixels. In order to further enhance its robustness to noise and outliers, we introduce a kernel distance measure to its objective function. The new algorithm adaptively determines the kernel parameter by using a fast bandwidth selection rule based on the distance variance of all data points in the collection. Furthermore, the tradeoff weighted fuzzy factor and the kernel distance measure are both parameter free. Experimental results on synthetic and real images show that the new algorithm is effective and efficient, and is relatively independent of this type of noise.
> Index Terms—Fuzzy clustering, gray-level constraint, image segmentation, kernel metric, spatial constraint.

摘要 — 本文通过引入权衡加权模糊因子和核度量，提出了一种改进的图像分割模糊C均值（FCM）算法。**权衡加权模糊因子(factor)**取决于所有相邻像素的空间距离及其灰度差。利用这个因子，新算法可以准确估计相邻像素的阻尼范围????。为了进一步提高对噪声和异常值的鲁棒性，我们引入了**核距离度量**到其**目标函数**。新算法使用基于**集合中所有数据点的距离方差的快速带宽选择规则**自适应地确定核参数。此外，权衡加权模糊因子和内核距离测量均无参数。合成图像和真实图像的实验结果表明，该算法有效、高效，与此类噪声相对无关。

索引术语 — 模糊聚类、灰级约束、图像分割、内核度量、空间约束。

## Introduction

最常见的图像分割方法之一是模糊聚类，在某些情况下，模糊聚类可以保留比硬聚类更多的图像信息

**Fuzzy c-means (FCM) algorithm**

尽管传统的FCM算法在大多数无噪声图像上效果良好，但它无法对被**噪声、异常值**和其他成像伪影损坏的图像进行分割。造成其鲁棒性弱的主要原因是其忽略了图像中的**空间上下文信息**以及使用非鲁棒的**欧氏距离**

**improved FCM algorithms**

 have been proposed by **incorporating local spatial information** into original FCM objective function

我们[12]提出了FLICM算法的变体（RFLICM）`，它采用**局部变化率(local coefficient of variation)**代替空间距离作为局部相似性度量。?????`而且有更强的鲁棒性。

**Motivation1**

`尽管RFLICM算法通过使用局部变化率可以利用更多的局部上下文信息来估计邻域像素之间的关系，**但忽视空间约束对中心像素和邻域像素之间关系的影响仍然是不合理的。**????????`

因此，在本研究中，我们的**动机之一**是设计一个权衡加权模糊因子(trade-off weighted fuzzy factor)，以`**自适应地控制局部空间关系????**`。此因子取决于所有相邻像素的空间距离及其灰度差异。

为了进一步提高FLICM在抗噪声方面的性能, 本研究的另一个新奇之处是**将内核距离测量引入其目标函数**

**核方法**

近年来，核方法在机器学习界得到了极大的关注。其**主要思想**是将原始低维特征空间中复杂的非线性问题转化为在变换空间中容易解决的问题。如: support vector machines (SVM) [13], [14] kernel principle component analy- sis (KPCA) [15] and kernel perceptron algorithm [16]

基于内核方法的聚类算法已被证明对数据集的异常值或噪声是可靠的 [17]。因此，基于内核方法的聚类算法已应用于图像分段的多个领域[18][21]。

## Motivation

#### FLICM 

FLICM 提出了一`种新的模糊因子Gki作为在其目标函数中的模糊局部相似性测量，旨在保证噪声不敏感和图像细节保存????`。其将数据集 [xi_N i=1（在灰级空间中）分区到 c 群集的目标函数在

把数据集$\{x_i\}_{i=1}^{N}$聚为c簇的目标函数是: 

<img src="/img/in-post/20_03/image-20200324120559957.png" alt="image-20200324120559957" style="zoom:50%;" />

其中$U=\{u_{ki}\}$代表隶属度矩阵满足:

<img src="/img/in-post/20_03/image-20200324153543466.png" alt="image-20200324153543466" style="zoom:50%;" />

> **有2个约束**, 前者是每一个像素点i属于所有类的隶属度之和为1, 后者是属于某一类的所有像素点的隶属度之和是不能取0和N的, 因为这样会导致某一类没有或者只有这一类

模糊因子$G_{ki}$:

<img src="/img/in-post/20_03/image-20200324153646649.png" alt="image-20200324153646649" style="zoom:50%;" />

> 第i个像素xi是本地窗口(local window)的中心，第j个像素 xj 表示沿 xi 窗口的相邻像素，dij 是像素 i 和 j 之间的**空间欧氏距离**，Ni 表示 xi 周围窗口中的邻域集。vk 表示第 k 个聚类中心的**像素**????????，ukj 表示与第 k 个聚类有关的灰度值 xj 的模糊隶属度。

`然后，通过更新隶属度$\left\{u_{ki}\right\}$和聚类中心$\left\{v_{k}\right\}_{k=1}^{c}$，可以获得最小值`

<img src="/img/in-post/20_03/image-20200324171259526.png" alt="image-20200324171259526" style="zoom: 67%;" />

为了将模糊的分区矩阵转换为清晰的分区，将发生解构过程。一般来说，采用最大成员资格程序方法。此过程将像素 i 分配给具有最高成员资格的 Ck 类

使用最高隶属度方法确定像素 i 的类别 $C_k$:

<img src="/img/in-post/20_03/image-20200324210500378.png" alt="image-20200324210500378" style="zoom:50%;" />

#### A. Motivation of Introducing the Trade-Off Weighted Fuzzy Factor

此外，Gki本地窗口中像素的影响是通过利用它们与中心像素之间的空间欧几里得距离来灵活发挥的。因此，Gki可以以距中心像素的空间距离反映邻居的衰减程度。

![image-20200325090030503](/img/in-post/20_03/image-20200325090030503.png)

> 如图(a), 噪声点 A 处与中间像素的灰度差比 B 处大, 但是阻尼程度是相反的.  如图1（b）所示，尽管中心像素是噪声像素，但它无法分析每个相邻像素的影响。它与附近像素之间的差异不一致。在[12]中，我们提出了一个`因数Gki的变体`，它**利用局部变化系数代替空间距离来克服上述缺点**，并有助于使用**更多的局部上下文信息**。描述如下

<img src="/img/in-post/20_03/image-20200325092137300.png" alt="image-20200325092137300" style="zoom: 67%;" />

> 式中，Cu表示中心像素的局部变化率。Cuj表示第j个邻居的局部变化率，cu^是位于局部窗口的Cj u的均值。
>
> 从视觉上看，G？相邻像素和中心像素之间的关系相对符合它们之间的灰色水平差异。但是，当窗口大小扩大时，忽视**空间距离约束对附近中心像素和像素之间重拉性的影响仍然是不合理的**。此外，由于灰度分布和空间约束不同，不能准确计算邻域的阻尼范围, 作为相同的灰级分布和不同的空间约束。**此因子取决于所有相邻像素的空间距离及其灰度差异。**在第三节中。B，我们将详细介绍这种权衡加权模糊因子。??????

#### B. Motivation of Using Non-Euclidean Distance

可以看出，在FLICM的目标函数中使用的度量仍然是FCM中的**欧几里德度量**。虽然这种测量方法在计算上很简单，但是使用欧几里德距离会导致**被噪声、离群值和其他成像伪影破坏的图像的分割结果不可靠**。因此，一些研究者采用所谓的鲁棒的距离度量，如Lp范数(0<p<1)等来代替FCM目标函数中的L2 范数, 以减少离群值对聚类结果的影响。

最近的机器学习工作中存在一种**使用“核方法”构造线性算法的非线性版本**的趋势，该方法旨在将原始低维特征空间中的复杂非线性问题转化为问题。在转换后的空间中可以轻松解决。而且该方法也可以用于聚类

特征空间中的内核可以表示为以下函数K特征空间中的内核可以表示为以下函数K

<img src="/img/in-post/20_03/image-20200325105549653.png" alt="image-20200325105549653" style="zoom:67%;" />

Gaussian Radial basis function (GRBF) kernel is a commonly- used method.



<img src="/img/in-post/20_03/image-20200325111230244.png" alt="image-20200325111230244" style="zoom:50%;" />

> 其中d是向量x的维数； σ是内核带宽，一个自定义参数； a≥0; 1≤b≤2。显然，对于所有x和以上RBF内核，K（x，x）= 1。请注意，高斯RBF中的参数σ对算法的性能有非常重要的影响。但是，为基于内核的聚类算法选择合适的带宽值可能会很麻烦，因为所有数据点都未标记，并且它们的真实类是未知的。

根据以上所述，在本研究中，**我们基于集合中所有数据点的距离方**差使用**快速带宽选择规则来确定参数σ**。基于核方法的自适应距离法将在第III.C节中详细介绍。

## METHODOLOGY

我们通过引入**权衡加权模糊因子和核方法**来改进FLICM。(a trade-off weighted fuzzy factor and kernel method.)

#### A. General Framework of `KWFLICM` Algorithm

目标函数:

<img src="/img/in-post/20_03/image-20200325121635717.png" alt="image-20200325121635717" style="zoom:50%;" />

重新表述的模糊因子:

<img src="/img/in-post/20_03/image-20200325121803501.png" alt="image-20200325121803501" style="zoom:50%;" />

> where Ni stands for the set of neighbors in a window around xi, wij is the trade-off weighted fuzzy factor of jth in a local window around xi, 1 − K (xi, vk) represents a non-Euclidean distance measure based on kernel method, (1 − uki)m is a penalty which can accelerate the iterative convergence to some extent. {vk}c k=1is the centers or prototypes of the clusters and the array {uki} represents a membership matrix which also must satisfy the Eq. (2).
>
> wij是围绕xi的局部窗口中jth的权衡加权模糊因子, 1-K是基于核方法的非欧距离度量, 1-u是可以一定程度上加速迭代收敛的惩罚.  vk是聚类中心. u代表也满足2式的隶属度矩阵

关于uki和vk最小化Jm的两个更新公式

<img src="/img/in-post/20_03/image-20200325133435651.png" alt="image-20200325133435651" style="zoom:50%;" />

the `proposed algorithm` can be summarized as follows

1. init 类别个数c, 模糊化参数m, 窗口尺寸Ni, 停止条件ε

2. 随机机初始化模糊聚类原型(prototypes)??

3. 设置循环计数器 b=0

4. 计算权衡加权模糊因子wij和修改的距离度量Dik^2???

5. 使用公式12更新分割矩阵

6. 使用公式13更新聚类中心

7. if max |Vnew− Vold| < ε then stop, otherwise, set b = b+ 1 and go to step 4.

   where V = [v1, v2, . . . , vc] are the vectors of the cluster prototypes.



当算法收敛后，采用去模糊化处理将模糊图像转化为清晰的分割图像。

#### B. Trade-Off Weighted Fuzzy Factor( 权衡加权模糊因子)

提出的KWFLICM的抗噪声性能主要依赖于模糊因子G, 如公式11 自适应加权模糊因子依赖于局部空间约束和局部灰度约束

对于每个具有坐标(pi, qi)的像素xi，**空间约束**通过与中心像素的空间距离反映相邻像素的阻尼程度，定义为:

<img src="/img/in-post/20_03/image-20200325144048272.png" alt="image-20200325144048272" style="zoom:50%;" />

然后，我们得到每个像素j的**局部变化率Cj**:

<img src="/img/in-post/20_03/image-20200325145739295.png" alt="image-20200325145739295" style="zoom:50%;" />

> 其中var (x)和x分别是图像局部窗口的强度方差和均值。

接下来，我们将Cj投影到内核空间中。然后，将权值标准化。由于**指数核的快速衰减**，Cj与这些局部变异系数的均值之间的较大距离将**导致权值趋近于零**。最后，根据Cj与C(局部窗口中Cj的均值)的比较，**对Cj进行了不同程度的补偿**，从而增大了阻尼范围在邻域内的差异。

方案为: 

<img src="/img/in-post/20_03/image-20200325150913160.png" alt="image-20200325150913160" style="zoom: 33%;" />

权衡加权模糊因子可以写作:

<img src="/img/in-post/20_03/image-20200325153608008.png" alt="image-20200325153608008" style="zoom: 33%;" />

#### C. Non-Euclidean Distance Based on Kernel Metric

objective function in KWFLICM is

<img src="/img/in-post/20_03/image-20200325154129342.png" alt="image-20200325154129342" style="zoom: 33%;" />

然后，通过核替换，我们得到

<img src="/img/in-post/20_03/image-20200325154252114.png" alt="image-20200325154252114" style="zoom: 33%;" />

## IV. EXPERIMENTAL STUDY