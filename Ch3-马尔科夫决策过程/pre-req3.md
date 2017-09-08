##### 1.基础知识

**加法公式**：若事件序列$A_1,A_2,\cdots$两两互不相容，则：
$$
P\left( \bigcup_{i=1}^\infty A_i \right) = \sum_{i=1}^\infty P(A_i) \tag{1}
$$
**乘法公式**：若事件组A_1,A_2,\cdots,A_n$满足$P(A_1A_2 \cdots A_{n-1})\neq 0$，则：
$$
P(A_1\cdots A_n) = P(A_1)P(A_2\mid A_1)\cdots P(A_n\mid A_1\cdots A_{n-1}) \tag{2}
$$
**全概率公式**：若事件组$B_1,\cdots,B_n$满足(1)两两互不相容；(2)$\bigcup_{i=1}^nA_i$是必然事件，则：
$$
P(A) = \sum_{i=1}^n P(B_i)P(A\mid B_i) \tag{3}
$$

**逆概率公式**：与(3)相同的条件，则：
$$
P(B_k\mid A) = \frac{P(B_k)P(A\mid B_k)}{\sum_{i=1}^n P(B_i)P(A\mid B_i)} \tag{4}
$$

##### 2.数字特征