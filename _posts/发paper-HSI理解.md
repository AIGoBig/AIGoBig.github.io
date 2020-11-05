---
layout: post
title: "论文整理-03.08"
subtitle: ''
author: "Sun"
header-style: text
tags:
  - Paper
  - Summary

---

# HSI数据

## Indian_1

10249像素

# 评价指标

**总体精度是模型在所有测试集上预测正确的与总体数量之间的比值**，**平均精度是每一类预测正确的与每一类总体数量之间的比值，最终再取每一类的精度的平均值**。

![img](/img/in-post/20_07/Center.jpeg)

## 混淆矩阵

*混淆矩阵*也称误差*矩阵*，是表示精度评价的一种标准格式，用n行n列的*矩阵*形式来表示。

|                                   | Predicted as Positive（预测-正例） | Predicted as Negative（预测-反例） |
| --------------------------------- | ---------------------------------- | ---------------------------------- |
| Labeled as Positive （真实-正例） | True Positive(TP-真正例)           | False Negative(FN-假反例)          |
| Labeled as Negative （真实-反例） | False Positive(FP-假正例)          | True Negative(TN-真反例)           |

TP：真正，被模型分类正确的正样本 【预测为1 实际为1】 

FN：假负，被模型分类错误的正样本 【预测为0 实际为1】

FP：假正，被模型分类错误的负样本 【预测为1 实际为0】

TN：真负，被模型分类正确的负样本 【预测为0 实际为0】

## 计算评价指标

### **Precision查准率、精确率**：

分类**正确的正样本个数占分类器分成的所有正样本个数的比例**

​      ![img](data:image/gif;base64,R0lGODlhdgAnALMAAP///wAAAHZ2diIiImZmZkRERMzMzFRUVLq6utzc3JiYmO7u7oiIiBAQEDIyMqqqqiH5BAEAAAAALAAAAAB2ACcAAAT+EMhJq704663aEULTgMdICUMACkRhbHAszzR8LJMpIUpVHBREIFErGo+1xGOSCCAmhicl0JssAgykdssFLCeKQBA3ERKhzq56XTw4NCHfm02vZxrZjAM4YTjIdoF2TS8ZAQUMDCBSgo10DGIZQoWOlXUFBXANlpx0WBoDfJ2jWgYBlBZXX6SsNAwEKQUCZycOh6uOKCogLaitv0Y/QUPAxUVUFFd5xswbZhSmjM3TFnEUBXPU2hR7FH6A29uHiYsZBAfo6eoC4Y6TjQHx8vP09fb3+Pn6+xPW7f8TQs04p64gO4B1VCH8J8BWAVwLI0qc2KXDhxAjBJQ4KEHXil7SahbACkCgZItP/VJ8dNHqRg6OPHyIAvBMTTdvEIWVITZKCZM0EqJUQGYFJQYF0jYEgIjAF1EJykjhCjNmGC0A0TQglSGETCEE4GoGBfrLjSY5HJJmaNgPgz8J2Izh0XBTwre0MgY40Gi0Ql0Ad4ER0jBOkQC1F7bCuFJIga8JhcsZg6ThHQyC6BzsScexwoNINDFYpobpbBHFGwhkW7tJ3LILAk8jtjCgM4bY1Ew9LgpRBuoMTWbzpvYq1qxqDnvD+H2h4SHbtW5R3MJ8ujFwlSIAADs=)

### **Recall查全率、召回率**：

分类正确的正样本个数占正样本个数的比例

​      ![img](data:image/gif;base64,R0lGODlheAAnALMAAP///wAAAHZ2diIiImZmZlRUVO7u7kRERKqqqszMzLq6uhAQEDIyMpiYmIiIiNzc3CH5BAEAAAAALAAAAAB4ACcAAAT+EMhJq7046w3aKoKwLGFBUsIQhAJxJFwsz3SNFcZ0SkpTHQWKIvCwGY/I2QMxeQQUkwSUEvBNDAFHcsvlMieNgDA3GRajz656fSwwNKLfm02vbxbaDCM4cTDIdoGCTjAZAQcODiFTgo12DmIZQ4WOlYEHB3ALlpx2WRoDfJ2jXAkBlBZYX6SsNg4EKgcCZygMh6uVKQEELQUEgK3BSQcEKAPCyEgBuAinyc8zQxUm0NUcAnMSBAuo1t4UewgNbnnf5lSUBFYXBAXu7/AC55XSYJFrAfn6+/z9/v8AAwocmG8CNgoE7s37NkDeBAbHMLSDR9HhQjtYGAFgkOmBxov+wrAFKLBKAYMG5UCqXMlyiwcQIkgIMGERgC4WLrohMQBrF4GfBz4ZVEHJQIFbrHDosNjjhygAZtbsqeAAFwAHoSoUY7WkSRoJUipUoYAlpYUGHzUsq6CgG8oAgAxY5YQrzBghRCiYSksBrYwhZAopAKbgQdkJCIAFc6NJzga/MQ5KqEkhz7Y+0PBomNrnz2O+FwYwmCnUQp69Esy2IqQWkSIBoCtA3oClUAOdAAys4wjA4zNIGibFmOiOwZ53lBHfA11Y+YN1yDA1tjFbA4FsGaADIKG6VenQT2dUz9CQg3YR3UmZwi1BlZHxF5zEBqs9I7JXsWZZEHlgrvn5NtkSIksG2wyg3VYtUQdggq0oZkkEADs=)

### **F1值**：

​     F1度量的一般形式是：![img](data:image/gif;base64,R0lGODlhtgAtALMAAP///wAAAO7u7mZmZiIiIkRERIiIiDIyMlRUVKqqqrq6uszMzHZ2dpiYmNzc3BAQECH5BAEAAAAALAAAAAC2AC0AAAT+EMhJq7046817TQYzeGRpnmiqrqRgSAjDznRtW0Jzk8ohNb6fYEcsGimj48YhEU0ECKV0mjIsKoKCgphLNF4UAnOiANsYhABjPShca+jAQIQYDKmcoMTBaAS2Nw46Em4TCIAUejYFUWQBYzQFSU0EeBsNgxV/HQ2IG2YABG9WAJASBgk7AZkAAgGgLAGpEwkBb5YXBaYTmxydHQq3DUkJCQIOsAuNFsgWCp7OjxQLvRTNFc8aCgEVCA+4Gd8X1Rq/HC8FBwUypQHuahbiFwygCcsYDPKEihT0FPYbGCga8OAWuAr6KJDLYO5SojsbuGXwBwDghgP3DByAaIGiRQ3+GL0gOADr4AQHlcZBw9AwgwNPFDcQ4NgR1b0MAQoYCMFgZc2PGmxNGMDK5MmUFhYORcB0JMam7CoUHVDywswNBZBqE2oiKzCJP8AafZKQl08LLVlWIHAW4QZ7azjkOwE36sSBYsdKKCtB6YW086Zp1cCXVqO4Ggjc5GARcQYCdg8M1qtLJae2EhZEzbILg7IMQB3jkFUitF0s5NSVwowHEwUBDAoEOMDA4F/WxjCBIIEKg4PFPecdyDmLw++OPgUGQFC8x5fXA9IMmN7mlZIsKwBLTcFP7w6MFXorqa2C5oXiJRQU9X6DNIVgUyYRUWCegwD57G9su/OG/pQcRtT+54uA+bEg0ASnFajggicQQBsC1jEo4YT2cdWAbTk0kCAAAzTlIVQUhqhELWSg1YRtKryj4oostujiizDGKOOMNNZo4ztDdUfLHQWI6GN+kIGWwAEHoMjhhx9u+OOSJzjglwSzOIBfiCA4waQUyq1zgQNjCKAjLgBu4AIMShbRAIE+NjclOGs6EwQQlkBxZQVRVTYWKTg04AUYY1gJRxpriFCIBGXsEMccHdqxAwgaGomLjoJMMKgEYuzAyHvSSPDlCpL0M9kKAjh6kGsWgCKKIayhsMprEQIgXjmp8oJeLaKeECsedr4nzCSkdFZKSdlsZQo1iHx2ya3bdFNYCujpeccXOupEVcwxVQEQU0WLdZRQOm4d68GBQxU0Zwl8sbJRO++U6VG2FoA3gUY05XWbByE1MFK14/o22Uv94IuPTR7ktNMaPl0Fa8C3EJUvCSiZiIS/GHj1Va2UmtdhU0R6qGSyE/ixsAcClFUUWybU1cFcHCzbcazgSjCAvB9jUJZdC3yqQWNlTqBYByr/EGuQE0gWcwe5AqDZE0Uzdo9oozV7gbEHi5laj/wOzRIruekJMTPAnZWl0+GBLRVryjFHxgHPWY0Bdh1TuCkF2qlNA3lQTqheB2jKzUIS/jF4n97eAZi3UWdOGAEAOw==)

> ​        ![img](data:image/gif;base64,R0lGODlhLAAQALMAAP///wAAAO7u7piYmKqqqoiIiNzc3ERERMzMzCIiIrq6umZmZjIyMnZ2dlRUVBAQECH5BAEAAAAALAAAAAAsABAAAASlEEgpBhll6s27JAWmcMYwHYinesoxMamWTUmsIc64bgcxDQ6NwjZYeASF3m4SsCkeMsCBcWgsCYeZyhAwTBABwU/DEC8VjoaZA/ZK2hKDTtLQLhGLxfrb5btNGgt2Ozh6JE0TCmESgDRzKy1qKgFzBFB0NwlLA1lLB40FQQAIVhQHbh1ISksALS9uBBUWgxs4j6wgRXONrL24vsA7CnvBxRPExsYRADs=) 度量了查全率对查准率的相对重要性
>
> ​       ![img](data:image/gif;base64,R0lGODlhKwAQALMAAP///wAAAO7u7piYmKqqqoiIiNzc3ERERMzMzCIiIrq6umZmZjIyMnZ2dlRUVAAAACH5BAEAAAAALAAAAAArABAAAAR5EEgpBhll6s37FIfCGcN0IF7qGc0QiFs2Jahqc++m1MCw3EBNTpM5MA6NoBIwnJQmDMES2AQYYJKGbLNweL/gZKr6nCy20zG2t0ms052qWIJIeLrg/DyORcxBBnA2TQQVFmiCGwINBwEMDShliWkEk3AKUpZTmZpLEQA7) 时退化为标准的F1
>
> ​       ![img](data:image/gif;base64,R0lGODlhKwAQALMAAP///wAAAO7u7piYmKqqqoiIiNzc3ERERMzMzCIiIrq6umZmZjIyMnZ2dlRUVAAAACH5BAEAAAAALAAAAAArABAAAASPEEgpBhll6s37FIfCGcN0IF7qGc0QiFs2JeiGOLDavZtSA4OFR1A4EHQcniZzYBwaSMJBhgQoJ6UJQ1BVOBpc3RVgyAEaVCRisQh7xtnJIq27tVXwTcKcUjzdKWNQEwgJSANTVRJXCIMAIAYpREaKE1cEFRZ0NjiVjw0HAQwNKHGepxxHqKsTCoCsp6+wnhEAOw==) 时查全率有更大影响
>
> ​       ![img](data:image/gif;base64,R0lGODlhKwAQALMAAP///wAAAO7u7piYmKqqqoiIiNzc3ERERMzMzCIiIrq6umZmZjIyMnZ2dlRUVAAAACH5BAEAAAAALAAAAAArABAAAASQEEgpBhll6s37FIfCGcN0IF7qGc0QiFs2JaiqKY6hvptSA4OFbTI4lGw8TebAODSGgsKBMJQkiRqGIGVYOGBVwBVgAAMaso3BsdCFJ+PjZJHWrNtvuFkuSZjVXj9VY08TCAlQDVNhVwiFACBuVUV8HlcEFRZ1bwQHkhoCigEMDSiVeagaVKmsEwpbra2wsawRADs=) 时查准率有更大影响

​    F1是基于查全率和查准率的调和平均定义的如下：

​      ![img](data:image/gif;base64,R0lGODlhkQAmALMAAP///wAAAO7u7kRERLq6utzc3HZ2dpiYmDIyMszMzGZmZiIiIoiIiFRUVKqqqhAQECH5BAEAAAAALAAAAACRACYAAAT+EMhJxSA06827tJgneuBojuWpFcYRhGq8tq9ssi5sn3i9U77fLyjsEIuj406JNDGbkic0I1VVp5prUTvlirxYANg23uqG57AxjS0L3Vm2Oj6fwJfy2N1cj+bVe3Z/fYFfg22HSYmAixwCB2t9dBsHAiICBgMBCAYJkhWZm50/ChuYmpyefaeiqhUNn7EjDK6yRQQMtroZCLtNvb66B5BqB40xDA7BtgMFc8Z1CbDLsQ910HXW1J/axcd6fQoN4+TlBk0JC7USBQvX3yoLAgH09fb3+PVt+QEbCQ9p2lHhh0+DOHIIEJjbQPCeBnnbVnULg23OxDAHyy2UdHFKRTX+HSNiafYMnglpIvsMK1knWco6FljOATZBgIIFARToVKAp14cLkg4w4GkSwCgNBnAaWMpznSMZSXMaEKfAEgUCxHhNm+CyR9ERldgFULaj1IYBWwEQCOBszgCzEpJWgEuF7AQCtQp1UDehgbsdjxhmBTDPJyC7ABwEcBVWw1qrqghY9dPnQVYD/bCsbSshgV7HmfuGRErzHMOvMfyGMXBxAM3VrxU8cLphAacGAQxTQX1iHuJfaRkgmBxGoYMDDRDoJrFYwgHaYnibGLC8SQDqDJZKN+JKwWAPiu9G+sSgOpK10NWspeDihILX4/s4MI+EtS4DsUOLWGDakCSsXGFCsUBan/BHAQJ/iVAAF5/5owABEDpAVxG+2TKPDggMAEABeeB3XX8asJKKRfcQ+IOHA/w2h4cN2EUAAkK9JOOMG0QAADs=)    ***\*or\****  ![img](data:image/gif;base64,R0lGODlheQAnALMAAP///wAAAO7u7mZmZiIiIkRERIiIiDIyMlRUVKqqqrq6uszMzHZ2dpiYmNzc3BAQECH5BAEAAAAALAAAAAB5ACcAAAT+EMhJq704680lIwEjDsXSVV8wMAMyCGcsz7RVIJQSOHQxUJ+acEikBBoUQcBACyQoiYCpSK1udLzJIqCY6SqIh3VMBoophQONoZ4MHtOynHrATQwH2KyeaCAOTHOCRQEFBgYiXRcLBHEWUm5Ig5M0Oo4ZCw+KFl8TDQGUoTEMZ0VsFAOgoqsaBHZFBAwUBwSsthZKT0VKmwAHBQAOvbehbIW6QsYIyAoHDYG4AyAD1CRLEwIFw1TPJNvEdK8SBk8ODJ/fQw16Dk7gZO45jlxkjRMIte9VOnpTCnoS6I15IAkAA1X6TLUxiEHgHHwJqRA4wADBtQsOy+SKSETJlAb8lyZkrNACgcmTJ2VpMLTIHscNUSakAzDSyiEMmb4F2Mmzp8+fQIMK9elmoYaaVBJAeykkVgekEkqiRKmSU8GlTE+0m2mEK40FAxSITfAj66gDhariYlAgAMWQQh74FGe2rt2771KIYFFiTIoVLV7gHXIjx44xPoDkG9ykIAAlWCs08GoEGYAocBlrwEJhC9fJJzrdK6X5BKkKaTaA7nDKDZzSMupQwAMQw2oOfPwAgj2j0KFEHW5vgBTVMe8rxDdI/VMnJQbRAD4dN006hvAMraMinN6K7onrGJxOoMV9w0Ya4HFl/BWMcmljBSx/96qMmbPI5bm5z8+q9pAIADs=)

**TPR:真正例率**

​     ![img](data:image/gif;base64,R0lGODlhkwAnALMAAP///wAAAJiYmBAQEFRUVHZ2du7u7rq6utzc3KqqqszMzDIyMoiIiERERGZmZiIiIiH5BAEAAAAALAAAAACTACcAAAT+EMhJq7046837FANRFMMwEiZVPMFYOI3izXRt3/hEGFMqHYJKg0A5BBC5pHLJrCASE0TgMFFQKYHgxBBgNL/gsAb6CRR5EyOyOhW732/CQkMSzuH4vHLgzSyIEwwLaHqFhhxSMhkBDQwMI1eHkpMVDGYZRoqUm5wADQ10A52jlF0aD4CkqnkKAZoWXGSrs2EMDiwNBWsqC4yycCAiJCYFKAUqLC4wrwArAQ4vBA6EtNUSOz3HP1oTQ0VHQg7I1uRPUW0SVhVZFFx9EwG/Ca7k1b8Clz/UahStkQBGKqCoRxCAnFB2KhS4I8HBAGYFZ/HR8IeCIGoA/iQQIOddxGr+iTQwcgTpAr2G3Cw4IMCypUttH91Y0pBpQ8AyYQLo3Mmzp8+fQIMKHUpUJ4VPCDcspOAgX8xZpjKg4vAAZsYHGVa63Gr1aZNWELfE28Dl3wJQCP55nWQLly4LC31liEtA1oEFAjyu3cu3b8xgI0qc8CHBmbIYTQzceuagcYOohVloMkBArl+DhAgDEZIKILgmFS3+AsBgKtPLEsylRgdAHZaU7jQIUCty9AFmeQMQMjCa7z2nB/Z9Tsf6wmwPRtAoCl4EQWwJCTCiNsjwQp2j1Y3TxrC0MIY+DgNNtzDRT+eLG453eLCgGORKxK/onR5yUaNHBbZbUE/2pICwBnBscxYAaY1nkVMW1MSBViwt8EdLXU0wTxoYHLDLPAikZCBSc4liA38aOJCdcRWYMN9071lgWg0gShXhfgqVZ2BrJ10Qyw0tXiCFflVoWNaMbTHylkK9NNAbBzkSKWRWAzygoTgz6pFklHtJp0cEADs=)

**FPR:假正例率**

​     ![img](data:image/gif;base64,R0lGODlhlAAnALMAAP///wAAAO7u7mZmZiIiIkRERIiIiDIyMlRUVKqqqrq6uszMzHZ2dpiYmNzc3BAQECH5BAEAAAAALAAAAACUACcAAAT+EMhJq7046827FQMRDORQBAbFiExrLl4sz3RtXwdSGUlV6BNFwHErGo9ITaBHUcAogQZFgEpar1iPUCB5KrjBIWURUGTP6DPjMGFgGA8fO02vFwkHBqKKA0oMB2B2g4QbVE8NTxYBBQYGLWaFkpMVCQFBGEKKlJycA3MacJ2jnARuGwR+pKt1DmWGS6yyaWuMpxe1BUx1ICIlJnwSKwEtDC8VwwPGCAOCs88YOTu7Ej9NYhQFAyoE0N4XsU2bAFFTwRLhEpbj389bXRJf10QTZJHxlxQIce3ta21v+E0oAEpYwQEP2PWThUfPOQrSJgByBiBHggYIDqRY+O2QhET+GBg5ggRO0QApGAYgWMmy5S2OWSxhuqBpgxAKDfJZCcCzp8+fQIMKHUq0qFGeEz5xELXhX1KdMGeZ4pCK6ssD3VK23IrgZdQkru5hoELtAhWxBwoAcCD2K6VcXlUcYFRWbgAEuxQcaLDRrd+/gAN/fNAVzoMW+14OUyRgjy4kvUaUONEXwLBixwRPQCDocBCUEqtS2HYlosSy1sLQE+xgV9h6bfkGECSgroUGbZWUdVKh3AQqlQFTy9nEmQIHwCckoHgBt5bZ8ADIUz3mleYKGTVsRCixg/MOTi0HlHPdwoPgO7pYR988N4aGe9hXVDWxPAVXCiUIAJ12rfvb/1Vz4BEAIIHTyCMMBCiYAVBZcJxyQ4BmgUosHZADS3Gpo9N/NdmXQQFqZSAhAIfJB2AHSjUlkIfgmEggMud58B0qGVYgGosWkJEfAAuMeJaMCkrwmgZk4bhDCLasltQDBIxImndBwpVBLrYZSceMVgrGHB0RAAA7)

### **ROC曲线**：

​    根据学习器的预测结果对样例进行排序，按此顺序逐个把正例进行预测，每次计算出真正例率、假正例率，并以真正例率为纵轴，假正例率为横轴，即得到ROC曲线

  

### **总体分类精度（Overall Accuracy）**:

**分类正确的样本个数占所有样本个数的比例**     

​      ![img](data:image/gif;base64,R0lGODlh6wAnALMAAP///wAAANzc3IiIiHZ2dpiYmGZmZu7u7hAQEMzMzERERCIiIjIyMrq6uqqqqlRUVCH5BAEAAAAALAAAAADrACcAAAT+EMhJq7046827BwXyEASCkM9JEUtAEoaSfFk4lieREms7S4dHQOGgGY/IpHLJlDwOE5WkUagoHpRGQMB8RnnTKmWwwFIMzbR6zW4LihJBoDFJ0CkBMTAw0BTuGG8TcoB2FQUDAVBAcG2Oj5CREo0gAVmLU1sUCXN+gBeUBZYTDZgADQIHfBMOppKvsLFHDwwaJVa1nh+0Gn0ABggTvrLExcYVCMMXDGYSAwyuFn8fyb0SnHfKx9vcj3I/GEMDAySfNeYZ3xkHegwKAALo3fP0SYkaWuAaBg/9tMz+wFy4lwEVqy166ilc+EHBuwy3jkzr4NAPMgLaGGrciCcjhTL+SCZyWFWjQgmPHFPW46SvgipKH0RqYJkhQUIAquSp3HlsgIEWCghwMclgCMwOMgf+HCLUArAFN9HwHESuQAEDQzMY0Dl1YdKujmBgOoCgZZY8YFVGS7smqAUyNUiynUv3iIFcFbSsrbJAat2/gDFokecggNkEXCoGXsxYQbALBDRVgPOTsWXAQzAwWGCh0YDHF/j5G91P4OXTxFSZHoSWggADLxSMWhOgtu3buHPr3s27t+/fwIMLH068eG8ALi4owMvqbFbU0FUqHrPAFUyaoUmTXh29uyOyn3S4ouIywFHv6OcdMDAAEYGWAoRAIyUEavr7+PPrx2+DhAkUUkj+wIILJMRgVhLrtWDAgjHIBcCA4ARhVBoJBsBgg8oM+IKBTfSHA4CmQTiBhESo5IUEAZJHwRXOrcFMBQPAREYzEvilxotjwMQiKZIpcSIAKd40YwU2biRIHJ1cY05re6AkwVcZmJdXS4goMuJ5h3BlgZRZtMQkTg5Ko+WR8CQJgCEUVDkWlgqFMtsppmjxHDa6dKDXNVPEmYpcrXAA5QV3nplni3WYecGfk6T5ZilZ7DlMn2DxAhFoEiy3AaJETcCdMzWC5mSaWq6A16YPUgqApXV2ICkGvgAjDFvVZICjM/OlusECDOgQ5qtnJvnpBJh+lKsQKM0KwDNrgUrNp77o0HlsWupEqcA45SAV6h4/FHAgOxO4A8+1IICrSrYHIjctOQSAGyxr5eLUzjvxpEWQYIZ1IFo/DABUGgaFkSJYVoUJcNMZo+U7Gqn9TkFvuwT7Y3BAGcwLKMAIsTUdZKZ6sO4vzGFw0wm/AgvuXRxEZMTGpz7k8UUhM7TrRzTGBO4CpKZpUqx+zlzzBCBJBC5yn950UlrYXfBSSNcSsoFN5f28rtIaHO1zB0VbwDQFOXXlE1BNETXhyaESUJRbGDwVlcwbiM0URGOzec4GWzP1XKf2nbHfpT/DgrIke9/dRrLbAF6M4I9EAAA7)    或  ![img](data:image/gif;base64,R0lGODlhfAAlALMAAP///wAAANzc3IiIiHZ2dpiYmGZmZu7u7hAQEMzMzERERCIiIjIyMrq6uqqqqlRUVCH5BAEAAAAALAAAAAB8ACUAAAT+EMhJq70463pU22AojmRpTgJRBN/pvnBMsnJt3y6N73w/6b6g8AUcGo+bInLJBCibUOEzSt1Nq1jZNcslHQiKAIOQcH0NhUEBKRioCgbBxtDCLgKEfN5wHwAcTgcOgEYEBgcTBwhlGA0Ba1mOBBYDBhIJCksKkxUDCxkrflwET5YAA6JGBgwXjogWawumXAsIrxOQD4xDjnUUDgG7EwlyCpldAgEPGKwuDgUEDc8PchgKCBik1b8SfF0SAwGENg1yAgiAixkBxxYMnxXjA9gYBg/3+PmcOwy+MoQN6P3i5IDAATwXkkFCYUBPGCsBIkqcSDFAhQYLdxBYxiGRBIT+FhQ04zbB0TYsCVJNSFUQhIB2FBhkzGCM0oJbEsZdClYvn88H+2p8uTALJwZo8CgcrNZiAEenEhT52mgUIwdxWYZyeDBLhICkAQGsuCQn1gGzUQ2ocSMMgIAHYm41gLtgZhORxvIyiCgKqsukCQw4SFDQQR0BzRB/M2L2gL2f1b6KUCGB8uILBO7sOgBXgU6XIzVIDnHTrN3LpxZwnNB18mkLo0EQGCBn9knUEtIEuCXIRGnOP6lJiI0bRrmDLI2GsL2hgYEAtIvDEGVAoErpxUUl0HEd++WzExhkEuDP++JyE4AJeG2ey0wEs9sXn0kAQXf5WRLMPFgef5XqdVUF0BpuEQAAOw==)

   

### **Kappa系数**：

​    ![img](data:image/gif;base64,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)

\- - - - - - - - - - - - - - - - - - - - - - -- - - -- - - - - - - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

 **（二） 例：** 单位/像素（pixel）

| class    | 水体  | 林地  | 耕地  | 未利用地 | 居民地 | 总计       |
| -------- | ----- | ----- | ----- | -------- | ------ | ---------- |
| 水体     | 25792 | 0     | 0     | 2        | 44     | 25838      |
| 林地     | 80    | 16825 | 297   | 684      | 1324   | 19210      |
| 耕地     | 519   | 60    | 27424 | 38       | 11542  | 39583      |
| 未利用地 | 31    | 0     | 0     | 9638     | 487    | 10156      |
| 居民地   | 323   | 0     | 49    | 133      | 30551  | 31056      |
| 总计     | 26745 | 16885 | 27770 | 10495    | 43948  | **125843** |

| 林地                | Predicted as Positive | Predicted as Negative |
| ------------------- | --------------------- | --------------------- |
| Labeled as Positive | 16825(TP)             | 2385(FN)              |
| Labeled as Negative | 60(FP)                | 93405(TN)             |

林地-精确率P：

P = （16825）/（16825+60）= 0.9964

林地-召回率R：

R = （16825）/（16825+2385）= 0.8758

林地-F1值：

![img](data:image/gif;base64,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)

总体精度： 正确分类的像元总和除以总像元数。被正确分类的像元数目沿着混淆矩阵的对角线（红色字体）分布，总像元数等于所有真实参考源的像元总数 （蓝色字体）   

OA = （110230/125843）=87.5933%

Kappa系数：通过把**所有真实参考的像元总数（N）乘以混淆矩阵对角线（Xii）的和，再减去各类中真实参考像元数与该类中被分类像元总数之积之后，再除以像元总数的平方减去各类中真实参考像元总数与该类中被分类像元总数之积对所有类别求和的结果。**

K = 0.8396

\- - - - - - - - - - - - - - - - - - - - - - -- - - -- - - - - - - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# 方法积累

## 网格搜索法(SVM_grid)

```python
clf = sklearn.svm.SVC(class_weight=class_weight)
clf = sklearn.model_selection.GridSearchCV(clf, SVM_GRID_PARAMS, verbose=5, n_jobs=4)  # Grid search function
clf.fit(X_train, y_train)
```



## nn.conv3d()

```java
class torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
```



```cpp
in_channels(int) – 输入信号的通道，就是输入中每帧图像的通道数
out_channels(int) – 卷积产生的通道，就是输出中每帧图像的通道数
kernel_size(int or tuple) - 过滤器的尺寸，假设为(a,b,c)，表示的是过滤器每次处理 a 帧图像，该图像的大小是b x c。
stride(int or tuple, optional) - 卷积步长，形状是三维的，假设为(x,y,z)，表示的是三维上的步长是x，在行方向上步长是y，在列方向上步长是z。
padding(int or tuple, optional) - 输入的每一条边补充0的层数，形状是三维的，假设是(l,m,n)，表示的是在输入的三维方向前后分别padding l 个全零二维矩阵，在输入的行方向上下分别padding m 个全零行向量，在输入的列方向左右分别padding n 个全零列向量。
dilation(int or tuple, optional) – 卷积核元素之间的间距，这个看看空洞卷积就okay了
groups(int, optional) – 从输入通道到输出通道的阻塞连接数；没用到，没细看
bias(bool, optional) - 如果bias=True，添加偏置；没用到，没细看
```



## 正则化方法

### `nn.LocalResponseNorm()-Local Response Normalization`

在由几个输入平面组成的输入信号上应用本地响应归一化，其中通道占据第二维。跨通道应用标准化。
$$
b_{c} = a_{c}\left(k + \frac{\alpha}{n}
        \sum_{c'=\max(0, c-n/2)}^{\min(N-1,c+n/2)}a_{c'}^2\right)^{-\beta}
$$








