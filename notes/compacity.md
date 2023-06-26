# Can the finite dimension condition be removed?

1. In the proof, it is used that $S^N$ is compact so that in $K\times S_N$ the map $(k,x)\mapsto T'(k)x$
is bounded below. This, combined with the fundamental theorem of calculus, shows the
inequality
$$\|x-y\| \leqslant C\|Fx-Fy\|$$
for $x,y\in K$.
2. In the proof that it can be uniformly bounded they are using the behaviour of a compact set
under a strongly convergent sequence of operators. This is a fairly easy lemma to prove,
$$\sup_{y\in K_Y}\|T_ny\|=\|T_ny_n\| \leqslant \|T_ny_n -T_ny\| + \|T_ny\|$$
where the first equality is by compactness. The uniform boundedness follows from Banach-Steinhaus.
This is used to show that
$$\sup_{\zeta\in K}\|(\text{id}-Q_n)F'(\zeta)\| \to 0$$
since $\text{ran} (F'(\zeta))\subseteq K_Y$ and $T_n = \text{id} -Q_n$ and $Q_n\to 0$ strongly.
3. In point 2, $K_Y$ was compact under the assumption that it is the image of $\hat K = K\times S_W$ under the map $(\zeta,x)\to F'(\zeta)x$.
Here $K$ is contained in a finite dimensional subspace $W$ and $S_W$ is the unit ball in $W$.
4. On the other hand we have the following lemma:
Let $K\subseteq X$ compact and $T_n \to T$ uniformly on $K$. Then,
$$T(K)\cup \bigcup_n T_n(K)$$
is compact.
5. Take $$
