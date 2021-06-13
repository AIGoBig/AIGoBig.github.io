# 按书上内容

## labuladong高频题

### 🚩(1h)[204. 计数质数](https://leetcode-cn.com/problems/count-primes/)

难度简单615

统计所有小于非负整数 *`n`* 的质数的数量。

**示例 1：**

```
输入：n = 10
输出：4
解释：小于 10 的质数一共有 4 个, 它们是 2, 3, 5, 7 。
```

**示例 2：**

```
输入：n = 0
输出：0
```

**示例 3：**

```
输入：n = 1
输出：0
```

 

**提示：**

- `0 <= n <= 5 * 106`

```python
class Solution:
    def countPrimes(self, n: int) -> int:
        # if n <=1 :return 0
        # not_prime = [False for i in range(n)]
        # for i in range(2,n//2+1):
        #     for j in range(2,n//i+1):
        #         if i*j >= n:
        #             break
        #         not_prime[i*j] = True

        # # print(not_prime)
        # return n-sum(not_prime)-2

        if n <=1 :return 0
        is_prime = [True for i in range(n)]
        for i in range(2,int(n**(1/2)+1)):
            # for j in range(2,n+1//i):
            if is_prime[i]:
                for j in range(i, (n-1)//i+1): # 注意i的取值，10个的话，取到9
                    is_prime[i*j] = False

        print(is_prime)
        return sum(is_prime)-2
```



![Sieve_of_Eratosthenes_animation.gif](/img/in-post/20_07/1606932458-HgVOnW-Sieve_of_Eratosthenes_animation.gif)

注意：

1. 注意每次找当前**素数 x** 的倍数时，是从 x^2 开始的。

   > 如果 x > 2，那么 2 * x 肯定被素数 2 给过滤了，以此类推，最小未被过滤的肯定是 x^2.

2. 由于（1），标记到 $\sqrt{n}$ 时停止即可

3. 细节问题：

   1. (n+1)//i 而不是 n+1//i
   2. 注意取值：`for j in range(i, (n-1)//i+1):` （10个的话，取到9）

####  2️⃣     

### 🚩(1h) [372. 超级次方](https://leetcode-cn.com/problems/super-pow/)

难度中等115

你的任务是计算 `ab` 对 `1337` 取模，`a` 是一个正整数，`b` 是一个非常大的正整数且会以数组形式给出。

```python
class Solution:
    def superPow(self, a: int, b: List[int]) -> int:

        if not b:
            return 1
        k = b.pop()
        part1 = self.mypow(a,k)
        part2 = self.mypow(self.superPow(a,b), 10)
        return part1*part2 % 1337

    def mypow(self, a, k):
        if k == 0:
            return 1
        if k%2 == 1:
            return (a * self.mypow(a,k-1)) % 1337
        if k%2 == 0:
            return (self.mypow(a,k//2)**2) % 1337
```

 注意：

1. 数组拆分乘
2. 分步取余
3. 快速幂

### 🚩(50m) [875. 爱吃香蕉的珂珂](https://leetcode-cn.com/problems/koko-eating-bananas/)

难度中等176

珂珂喜欢吃香蕉。这里有 `N` 堆香蕉，第 `i` 堆中有 `piles[i]` 根香蕉。警卫已经离开了，将在 `H` 小时后回来。

珂珂可以决定她吃香蕉的速度 `K` （单位：根/小时）。每个小时，她将会选择一堆香蕉，从中吃掉 `K` 根。如果这堆香蕉少于 `K` 根，她将吃掉这堆的所有香蕉，然后这一小时内不会再吃更多的香蕉。 

珂珂喜欢慢慢吃，但仍然想在警卫回来前吃掉所有的香蕉。

返回她可以在 `H` 小时内吃掉所有香蕉的最小速度 `K`（`K` 为整数）。



**示例 1：**

```
输入: piles = [3,6,7,11], H = 8
输出: 4
```

**示例 2：**

```
输入: piles = [30,11,23,4,20], H = 5
输出: 30
```

**示例 3：**

```
输入: piles = [30,11,23,4,20], H = 6
输出: 23
```

 

**提示：**

- `1 <= piles.length <= 10^4`
- `piles.length <= H <= 10^9`
- `1 <= piles[i] <= 10^9`

### [292. Nim 游戏](https://leetcode-cn.com/problems/nim-game/)

难度简单434

你和你的朋友，两个人一起玩 [Nim 游戏](https://baike.baidu.com/item/Nim游戏/6737105)：

- 桌子上有一堆石头。
- 你们轮流进行自己的回合，你作为先手。
- 每一回合，轮到的人拿掉 1 - 3 块石头。
- 拿掉最后一块石头的人就是获胜者。

假设你们每一步都是最优解。请编写一个函数，来判断你是否可以在给定石头数量为 `n` 的情况下赢得游戏。如果可以赢，返回 `true`；否则，返回 `false` 。

