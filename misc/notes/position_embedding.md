# Position Embedding

To enable a transformer[^1] to use positional information regarding the order of sequence, position embeddings are used. There are different types of position embeddings available: absolute position embedding[^1] and relative position embedding[^2].

## Absolute Position Embedding

## Relative Position Embedding

## Modern Alternatives

### Rotary Position Embedding

Rotary Position Embedding (RoPE)[^3] encodes relative position information as rotations under 2D subspaces.

Let us denote the query vector at $m^{\text{th}}$ and key vector at $n^{\text{th}}$ position obtained during attention as follows:

$$
\begin{align}
\textbf{q}_m = f_q(x_m, m) \in \mathbb{R}^{d_\text{head}} \\
\textbf{k}_n = f_k(x_n, n) \in \mathbb{R}^{d_\text{head}}
\end{align}
$$

where, $d_\text{head}$ is the dimension to which the embeddings are projected by the weight matrices for query and key.

Then, we want to determine some function $g(x_m, x_n, n - m)$ such that the dot product between $\textbf{q}$ and $\textbf{k}$ effectively captures relative position information,

$$
\textbf{q}_m^T \cdot \textbf{k}_n = \langle f_q(x_m, m), f_k(x_n, n) \rangle = g(x_m, x_n, n - m)
$$

## Exercises

1. Prove that the following is a solution to $\langle f_q(x_m, m), f_k(x_n, n) \rangle = g(x_m, x_n, n - m)$:

$$
\begin{align}
f_q(x_m, m) = (\textbf{W}_q \textbf{x}_m)e^{i m \theta} \\
f_k(\textbf{x}_n, n) = (\textbf{W}_k \textbf{x}_n)e^{i n \theta} \\
g(\textbf{x}_m, \textbf{x}_n, n - m) = Re[(\textbf{W}_q \textbf{x}_m)(\textbf{W}_k \textbf{x}_m)^* e^{i(n - m)\theta}].
\end{align}
$$

    


[^1]: Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems 30 (2017).
[^2]: Shaw, Peter, Jakob Uszkoreit, and Ashish Vaswani. "Self-attention with relative position representations." Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers). 2018.
[^3]: Su, Jianlin, et al. "Roformer: Enhanced transformer with rotary position embedding." Neurocomputing 568 (2024): 127063.
