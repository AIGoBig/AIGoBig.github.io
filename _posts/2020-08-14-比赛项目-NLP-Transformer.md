---
layout: post
comments: true
mathjax: false
subtitle: ''
author: "Sun"
header-style: text
tags:
  - 比赛项目
  - 
  - 
---

# TODO:

word2vec

GPT

bert

eimo

# 就业相关

## 岗位要求

![image-20200817211236832](/img/in-post/20_07/image-20200817211236832.png)

![image-20200817211133136](/img/in-post/20_07/image-20200817211133136.png)

![image-20200817211617930](/img/in-post/20_07/image-20200817211617930.png)



## 就业方向

对话系统

舆情监控

==**推荐系统**==

搜索

机器翻译

# 预训练模型介绍

## 发展历程

<img src="/img/in-post/20_07/image-20200817171443663.png" alt="image-20200817171443663" style="zoom:50%;" />

NLP=NLU+NLG

<img src="/img/in-post/20_07/image-20200817222354651.png" alt="image-20200817222354651" style="zoom:50%;" />



## 技术演化路径

### Word2vec



![image-20200817234716393](/img/in-post/20_07/image-20200817234716393.png)

> 有上下文信息, 
>
> CBOW: 用中间预测前后预测中间词
>
> 



![image-20200818000802966](/img/in-post/20_07/image-20200818000802966.png)

### 预训练模型

![image-20200818001101464](/img/in-post/20_07/image-20200818001101464.png)



![image-20200818001246858](/img/in-post/20_07/image-20200818001246858.png)

> bert 用的是encoder
>
> GPT用的decoder, 去掉了中间一层
>
> bert 用的 masked Language Modeling的结构(隐藏中间并预测这个辅助任务), 前向后相都考虑了
>
> gpt 只能从左到右
>
> ELMo concat了左到右和右到左
>
> 

## 学习路径

<img src="/img/in-post/20_07/image-20200817222650621.png" alt="image-20200817222650621" style="zoom:50%;" />

<img src="/img/in-post/20_07/image-20200817222728605.png" alt="image-20200817222728605" style="zoom:50%;" />

<img src="/img/in-post/20_07/image-20200817222828941.png" alt="image-20200817222828941" style="zoom:50%;" />

<img src="/img/in-post/20_07/image-20200817222909371.png" alt="image-20200817222909371" style="zoom:50%;" />

资料推荐

![image-20200817223331474](/img/in-post/20_07/image-20200817223331474.png)























# 总览

前期知识

![image-20200814210848408](/img/in-post/20_07/image-20200814210848408.png)



![image-20200814211006951](/img/in-post/20_07/image-20200814211006951.png)

#### WMT数据集

语言翻译

<img src="/img/in-post/20_07/image-20200814211210793.png" alt="image-20200814211210793" style="zoom:50%;" />

#### 参考指标bleu

![image-20200814211559646](/img/in-post/20_07/image-20200814211559646.png)

<img src="/img/in-post/20_07/image-20200814211749924.png" alt="image-20200814211749924" style="zoom:50%;" />

#### transform big

#### self-attention

可以降低时间复杂度

具有更强的可解释性, 显示了不同词语间的关联信息.

![image-20200814212529964](/img/in-post/20_07/image-20200814212529964.png)

## transformer 历史意义

![image-20200814212846684](/img/in-post/20_07/image-20200814212846684.png)

1.  提出self-attention, 拉开非序列化模型序幕
2. 为预训练模型到来打下基础
3. bert等

![image-20200814213158860](/img/in-post/20_07/image-20200814213158860.png)



