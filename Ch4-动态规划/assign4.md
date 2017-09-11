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