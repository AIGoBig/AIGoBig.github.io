---
shortcutlayout: post
comments: true
mathjax: false
title: ""
subtitle: ''
author: "Sun"
header-style: text
tags:
  - 
  - 
  - 

---

##  自定义层

几个必须:

```python
class Linear(nn.Module): # 必须继承nn.Module
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__() # 必须调用nn.Module的构造函数,等价于nn.Module.__init__(self)
        # 在构造函数__init__中必须自己定义可学习的参数，并封装成Parameter, 如此处的w和b
        self.w = nn.Parameter(t.randn(in_features, out_features)) 
        self.b = nn.Parameter(t.randn(out_features))
    
    def forward(self, x):
        x = x.mm(self.w) # x.@(self.w)
        return x + self.b.expand_as(x)
      
layer = Linear(4,3)
input = t.randn(2,4)
output = layer(input)  # 调用layer(input)即可得到input对应的结果。它等价于`layers.__call__(input)`，在`__call__`函数中，主要调用的是 `layer.forward(x)`
output
```

### `nn.Parameter`

Build Parameter matrix like weight or bias   

`parameter`是一种特殊的`Tensor`，但其默认需要求导（requires_grad = True），感兴趣的读者可以通过`nn.Parameter??`，查看`Parameter`类的源代码。

### `forward()` - 前项传播函数

实现前项传播过程, 构建所有层, 输入是一个或多个tensor, 最终返回结果

无需写反向传播函数，nn.Module能够利用autograd自动实现反向传播，这点比Function简单许多。

使用时，直观上可将layer看成数学概念中的函数，==调用layer(input)即可得到input对应的结果。它等价于`layers.__call__(input)`，在`__call__`函数中，主要调用的是 `layer.forward(x)`==，另外还对钩子做了一些处理。所以在实际使用中应==尽量使用`layer(x)`而不是使用`layer.forward(x)`==，关于钩子技术将在下文讲解。

### `named_parameters()`

`Module`中的可学习参数可以通过`named_parameters()`或者`parameters()`返回迭代器，前者会给每个parameter都附上名字，使其更具有辨识度。

### Module

可见利用Module实现的全连接层，比利用`Function`实现的更为简单，因其不再需要写反向传播函数。

这些自定义layer对输入形状都有假设：==输入的不是单个数据，而是一个batch, 则必须调用`tensor.unsqueeze(0)` 或 `tensor[None]`将数据伪装成batch_size=1的batch==

### 子Module

Module能够==自动检测到自己的`Parameter`，并将其作为学习参数。除了`parameter`之外，Module还包含子`Module`，主Module能够递归查找子`Module`中的`parameter`。==下面再来看看稍微复杂一点的网络，多层感知机。

构造函数`__init__`中，==可利用前面自定义的Linear层(module)，作为当前module对象的一个子module，它的可学习参数，也会成为当前module的可学习参数。==

### Sequential ??? 

```
return nn.Sequential(*layers)
```

