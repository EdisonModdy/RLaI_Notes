##### 总结回顾

**策略评估**：
$$
v_{k+1}(s) = \sum_{a} \pi(a\mid s) \sum_{s',r} p(s',r\mid s,a)\left[r+\gamma v_k(s')\right]
$$

$$
\bbox[25px,border:2px solid]
{
\begin{aligned}
&\text{Input }\pi,\text{ the policy to be evaluated}\\
&\text{Initialize an array }V(s)=0,\text{ for all }s\in\mathcal S^+\\
&\text{Repeat}\\
&\qquad\Delta \leftarrow 0\\
&\qquad\text{For each }s\in\mathcal S:\\
&\qquad\qquad v \leftarrow V(s)\\
&\qquad\qquad V(s)\leftarrow\sum_a\pi(a\mid s)\sum_{s',r}p(s',r\mid s,a)\left[ r+\gamma V(s') \right]\\
&\qquad\qquad\Delta\leftarrow\max(\Delta,\left\vert v-V(s) \right\vert)\\
&\text{until }\Delta < \theta\text{ (a small positive number) }\\
&\text{Output }V\approx v_\pi
\end{aligned}
}
$$

**策略改善**：
$$
\pi'(s) = \arg\max_a\sum_{s',r} p(s',r\mid s,a)\left[ r+\gamma v_\pi(s') \right]
$$
**策略迭代**：
$$
\bbox[25px,border:2px solid]
{
\begin{aligned}
1.&\text{Initialization}\\
&V(s) \in \mathbb R\text{ and }\pi(s) \in \mathcal A(s)\text{ arbitrary for all }s\in \mathcal S\\
\\
2.&\text{Policy Evaluation}\\
&\text{Repeat}\\
&\qquad\Delta \leftarrow 0\\
&\qquad\text{For each }s \in \mathcal S:\\
&\qquad\qquad v\leftarrow V(s)\\
&\qquad\qquad V(s) \leftarrow \sum_{s',r}p(s',r\mid s,\pi(s))\left[ r+\gamma V(s') \right]\\
&\qquad\qquad \Delta\leftarrow \max(\Delta,\left\vert v-V(s) \right\vert)\\
&\text{until }\Delta<\theta\text{ (a small positive number) }\\
\\
3.&\text{Policy Improvement}\\
&policy\text-stable \leftarrow true\\
&\text{For each }s\in\mathcal S:\\
&\qquad old\text-action \leftarrow \pi(s)\\
&\qquad \pi(s) \leftarrow \arg\max_{a} \sum_{s',r} p(s',r\mid s,a)\left[ r+\gamma V(s') \right]\\
&\qquad \text{If } old\text-action\neq\pi(s),\text{ then }policy\text-stable \leftarrow false\\
&\text{If }policy\text-stable,\text{ then stop and return }V\approx v_*\text{ and }\pi\approx\pi_*;\text{else go to 2}
\end{aligned}
}
$$
**价值迭代**：



##### 作业与练习

**练习4.1**：示例4.1中，若$\pi$是等概率随机策略，则$q_\pi(11,\mathtt{down}), q_\pi(7,\mathtt{down})$分别是什么？

**练习4.2**：示例4.1中，若在状态13的正下方加入状态15，其行为$\mathtt{left, up, right, down}$分别转移到12，13，14和15。假定从原状态的转移不变，则等概率随机策略的$v_\pi(15)$是什么？若状态13的动态也发生变化，其行为$\mathtt{down}$将其带到新状态15，则这种情况下等概率随机策略的$v_\pi(15)$是什么？

**练习4.3**：行为价值函数$q_\pi$类似于(4.3)、(4.4)和(4.5)的等式，和其函数序列$q_0,q_1,q_2,\dots$的连续近似分别是什么？

**练习4.4（编程）**：写一个策略迭代的程序并按照以下变化重新解决Jack的汽车租赁问题。Jack第一个地点的一个员工需要每晚乘公交回第二个地点附近的家，因此很乐意免费将车开到第二个地点。此方向额外的车和另一个方向所有车的移动的花销依然是\$2。另外，Jack在每个地点的车位有限，若一个地点一晚停有超过10辆车（在车的任何移动后），则需额外的\$4来使用第二个停车场（无论在那停几辆车）。这些非线性和任意动态经常在实际问题中出现，无法用动态规划以外的最优化方法来解决。为检查程序，先复现原始问题给出的结果。

**练习4.5**：如何定义行为价值的策略迭代？给出完整的与$v_*$类似的计算$q_*$的算法。要特别重视这个练习，因其包含的思想在本书其余部分都会用到。

**练习4.6**：假定仅考虑$\epsilon\text-soft$的策略，即在每个状态$s$选择每个行为的概率至少是$\epsilon/\left\vert\mathcal A(s)\right\vert$。以3、2、1的顺序，量化地描述$v_*$的策略迭代算法每一步相应的变化。

**练习4.7**：为何赌徒问题的最优策略的形式如此奇怪？尤其是资本为50时堵上所有资本，但51时就不。为何这是好的策略？

**练习4.8（编程）**：实现赌徒问题的价值迭代并在$p_h=0.25$和$p_h=0.55$时解决此问题。编程中引入两个对应于资本为0和100的终止的伪状态可能会比较方便，分别赋予其价值0和1。似图4.3那样图形化展示结果。在$\theta \to 0$时结果是否稳定。

**练习4.9**：与(4.10)类似的行为价值$q_{k+1}(s,a)$价值迭代的备份是什么。