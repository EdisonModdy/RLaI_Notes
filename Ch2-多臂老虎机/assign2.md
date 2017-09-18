##### 复习与回顾

1. **行动价值**：选择行动$a$后的期望激励$q_*(a)$：

$$
q_*(a) \dot= \mathbb E[R_t\mid A_t=a]
$$

2. **行动-价值方法**：记行动$a$在时间$t$的估计价值为$Q_t(a)$：
   $$
   Q_t(a) \dot= \frac{\text{迄时间t采取a获得激励的总和}}{\text{迄时间t采取a的次数}} = \frac{\sum_{i=1}^{t-1} R_i \bullet\mathbf 1_{A_i=a}}{\sum_{i=1}^{t=1} \mathbf 1_{A_i=a}}
   $$
   $\mathbf 1_{predicate}$为预测为真时1否则0的随机变量；若分母为0，则设$Q_t(a)$为一默认值，如$Q_1(a) = 0$。


3. **10臂试验台**：行动价值$q_*(a), a=1,\dots,10$由均值为0方差为1的正态分布产生；在时间$t$选择行动$A_t$后，激励值$R_t$由均值为$q_*(A_t)$方差为1的正态分布产生。

4. **贪婪与$\varepsilon$贪婪**：若某个行动激励的方差变大，适于$\varepsilon$-贪婪；若方差为0，则贪婪表现最好，但若任务非平稳，也需要探索。

5. **增量实现**：令$R_i$表示所评估的行动在第$i$次选中后收到的激励，$Q_n$表示被选中$n-1$次后价值的估计，即$Q_n \dot= \frac{R_1 + R_2 + \cdots + R_{n-1}}{n-1}$，可以改为增量形式来提高效率：
   $$
   Q_{n+1} = \frac{1}{n} \sum_{i=1}^n R_i  = Q_n + \frac{1}{n} \left[ R_n - Q_n \right]
   $$

6. 使用增量计算样本均值和$\varepsilon$贪婪行为选择的完整老虎机算法：
   $$
   \bbox[25px,border:2px solid]
   {\begin{aligned}
   &\underline{\mathbf{A\space simple\space bandit\space algorithm}}\\
   \\
   &\text{Initialize, for }a=1\text{ to }k\text{:}\\
   &\qquad Q(a) \leftarrow 0\\
   &\qquad N(a) \leftarrow 0\\
   \\
   &\text{Repeat forever:}\\
   &\qquad A \leftarrow 
   \begin{cases}
   \arg\max_a Q(a)&\qquad\text{with probability }1-\varepsilon\qquad(\text{breaking ties randomly})\\
   \text{a random action}&\qquad\text{with probability }\varepsilon
   \end{cases}\\
   &\qquad R \leftarrow bandit(A)\\
   &\qquad N(A) \leftarrow N(A) + 1\\
   &\qquad Q(A) \leftarrow Q(A) + \frac{1}{N(A)}[R-Q(A)]
   \end{aligned}}
   $$

7. 用**常量步长**处理**非平稳**问题：非平稳环境中行动的真值随时间变化，可以给近期的激励赋予更多权重，即加权平均，最普遍的一种做法是使用固定步长参数，也称指数新近加权均值：
   $$
   Q_{n+1} \dot= Q_n + \alpha \left[ R_n - Q_n \right]=(1-\alpha)^nQ_1 + \sum_{i=1}^n \alpha(1-\alpha)^{n-i}R_i
   $$
   步长参数$\alpha \in (0,1]^1$为常量，权重之和$(1-\alpha)^n + \sum_{i=1}^n \alpha(1-\alpha)^{n-i}=1$。

8. **步长参数随步数改变**：令$\alpha_n(a)$表示$n$次选择行动$a$后用于计算激励的步长参数，在随机逼近理论中，保证以概率1收敛到真实行动价值的条件是：
   $$
   \sum_{n=1}^\infty \alpha_n(a) = \infty\ \ \ \ \ \ \ \ \text{and}\ \ \ \ \ \ \ \ \sum_{n=1}^\infty \alpha_n^2(a) < \infty
   $$
   第一个条件保证了步长足够大到能逐渐克服任何初始条件或随机波动，这二个条件保证了步长逐渐变得足够小到能收敛。采样-平均的步长参数$\alpha_n(a) = \frac{1}{n}$两个收敛条件都满足；常量步长$\alpha_n(a)=\alpha$不满足第二个条件。





##### 作业与练习

**练习 2.1**  在$\varepsilon$-贪婪行动选择中，有两种行动并且$\varepsilon=0.5$，求贪婪行动被选中的概率。

> 设选中行动为贪婪为事件$G$，选择行动时发生探索为事件$R$，
> $$
> \begin{eqnarray*}
> P(G) &=&P(R)\times P(G\mid R) + P(\Omega-R)\times P(G\mid \Omega-R)\tag{全概率公式}\\
> &=& 0.5\times 0.5 + 0.5\times 1\\
> &=& 0.75
> \end{eqnarray*}
> $$
>



**练习 2.2 老虎机示例**：考虑行为个数$k=4$（记为1，2，3，4）的多臂老虎机问题，应用一个行动选择为$\varepsilon$-贪婪、行动价值估计为抽样-平均的老虎机算法，且初始估计$Q_1(a)=0, \forall a$。若行动和激励的初始序列为$A_1=1, R_1=1, A_2=2, R_2=1, A_3=2, R_3=2, A_4=2, R_4=2, A_5=3, R_5=0$。其中某些时间$\varepsilon$情况可能发生了，使其选择的行为是随机的。求$\varepsilon$必然和可能分别发生在哪一步？

> 列出每一步每个行动的价值估计
>
> | 时间$t$ | 选择行为$A_t$ | $G_t(1)$ | $G_t(2)$ | $G_t(3)$ | $G_t(4)$ |
> | :---: | :-------: | :------: | :------: | :------: | -------- |
> |   0   |     -     |    0     |    0     |    0     | 0        |
> |   1   |     1     |    1     |    0     |    0     | 0        |
> |   2   |     2     |    1     |    1     |    0     | 0        |
> |   3   |     2     |    1     |   1.5    |    0     | 0        |
> |   4   |     2     |    1     |   1.67   |    0     | 0        |
> |   5   |     3     |    1     |   1.67   |    0     | 0        |
>
> 可以看出第2、5步必然发生了$\varepsilon$，第1、3步可能发生了$\varepsilon$。



**练习 2.3**  在图2.2的比较中，就激励累积和选择最优行为概率累积而言，哪种方法的长期表现会最好？有多好？定量地表示答案。

> $\varepsilon=0.01$的方法会最好。
>
> 因为在极限情况下，贪婪算法选择最优行为的概率和次数都很低，因此其累积的激励和选择最优行为概率的累积都是最差的。
>
> 而在极限情况下，$\varepsilon$-贪婪算法选择最优行为的概率为超过$1-\varepsilon$，因此$\varepsilon$越小，选择最优行为的概率越大，选择最优行为的累积次数越多，所获的的累积激励越多。



**练习2.4**  若步长参数$\alpha_n$不是常量，则$Q_n$估计的值是前面收到激励权值权重异于(2.6)的加权平均。求一般情况中类似(2.6)关于步长参数序列的每一个先前激励的权重。

>令$\alpha_n$表示$n$次选择评估行动后用于计算激励的步长参数，则
>$$
>\begin{eqnarray*}
>Q_{n+1}
>&=& Q_n + \alpha_n[R_n-Q_n]\\
>&=& \alpha_nR_n + (1-\alpha_n)Q_n\\
>&=& \alpha_nR_n + (1-\alpha_n)(\alpha_{n-1}R_{n-1}+(1-\alpha_{n-1})Q_{n-1})\\
>&=& \alpha_nR_n + \alpha_{n-1}(1-\alpha_n)R_{n-1}+(1-\alpha_n)(1-\alpha_{n-1})Q_{n-1}\\
>&=& \alpha_nR_n + \alpha_{n-1}(1-\alpha_n)R_{n-1}+(1-\alpha_n)(1-\alpha_{n-1})(\alpha_{n-2}R_{n-2}+(1-\alpha_{n-2})Q_{n-2})\\
>&=& \alpha_nR_n + \alpha_{n-1}(1-\alpha_n)R_{n-1}+\alpha_{n-2}(1-\alpha_{n-1})(1-\alpha_n)R_{n-2} + (1-\alpha_{n-2})(1-\alpha_{n-1})(1-\alpha_n)Q_{n-2}\\
>&=& \alpha_nR_n + \alpha_{n-1}(1-\alpha_n)R_{n-1}+\cdots+\alpha_{n-k}(1-\alpha_{n-k+1})\cdots(1-\alpha_n)R_{n-k}+\cdots+(1-\alpha_1)\cdots(1-\alpha_n)Q_1\\
>&=&R_n\alpha_n + \cdots+R_{n-k}\alpha_{n-k}\prod_{j=0}^{k-1}(1-\alpha_{n-j}) + \cdots+R_1\prod_{j=1}^{n}(1-\alpha_j)\\
>&=& \sum_{i=1}^n R_i\prod_{j=1}^{i}(1-\alpha_{n-j})
>\end{eqnarray*}
>$$
>



**练习2.5（程序）**设计并实施一个实验展示采样-平均方法在非平稳问题上的困难。对10臂实验台稍作修改，所有的$q_*(a)$平等出发，然后独立地随机游走（即在每一步给所有$q_*(a)$增加一个均值为0方差为0.01的正态分布量）。使用增量采样平均和常数步长$\alpha=0.1$两种价值方法画出类似2.2的图。使用$\varepsilon=0.1$和10000步的运行。

>这里给出算法的核心代码，具体见bandit.py文件
>
>```python
>epsilon = 0.1
>ARM_NUM = 10
>RUN_NUM = 2000
>EPOCH_NUM = 100000
>
># Store the average rewards in each step
>avg_rewards = [0 for i in range(EPOCH_NUM)]
>avg_optimals = [0 for i in range(EPOCH_NUM)]
>for run in range(RUN_NUM):
>    qstars = [rd.gauss(0,1) for i in range(ARM_NUM)]
>    qestims = [0 for i in range(ARM_NUM)]
>    nums = [0 for i in range(ARM_NUM)]
>    for epoch in range(EPOCH_NUM):
>        # epsilon greedy
>        if rd.random() < epsilon:
>            action = rd.choice(qcurs)
>        else:
>            action = np.argmax(qcurs)
>        # Incrementally compute the estimates
>        reward = rd.gauss(qstars[action], 1)
>        nums[action] += 1
>        qcurs[action] += 1.0 / nums[action] * (reward - qcurs[action])
>        # Add the statistic variables
>        avg_rewards[epoch] += reward
>        if action == np.argmax(qstars):
>            avg_optimals[epoch] += 1
>        # Add the increment
>        qstars = [qstar+rd.gauss(0,0.01) for qstar in qstars]
>```
>





**练习2.6 神秘尖峰**：图2.3展示的结果应该是相当可靠的，因其是超过2000次单独、随机从10臂老虎机任务中选择的。那么，在乐观方法早期部分的曲线是否存在振荡和尖峰，为什么？或者说，有可能是什么使得这种方法在平均上、在早期特定步骤表现得特别地好或差？

**练习2.7**  证明在两个行为的情况下，softmax分布和在统计学和人工神经网络中经常用到的logistic或sigmoid函数给定的分布相同。
