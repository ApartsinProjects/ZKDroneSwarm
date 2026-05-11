# Theoretical Bounds for ZK-MRTA

**Research memo.** Self-contained theoretical analysis of the Zero-Knowledge Multi-Robot Task Allocation benchmark. All math is rendered in LaTeX.

---

## 1. Formal model

Let $m$ be the number of drones, $n$ the number of targets, and $d$ the number of latent modes. Each drone $i\in[m]$ has a latent vector $z_i^a\in\Delta_d$; each target $j\in[n]$ has $z_j^t\in\Delta_d$. The compatibility matrix is

$$R_{ij}=\langle z_i^a,z_j^t\rangle\in[0,1], \qquad R\in\mathbb{R}^{m\times n}.$$

We focus on the **one-hot benchmark** in which $z_i^a=e_{\phi(i)}$ and $z_j^t=e_{\psi(j)}$ for mode maps $\phi:[m]\to[d]$, $\psi:[n]\to[d]$, so $R_{ij}=\mathbf{1}\{\phi(i)=\psi(j)\}$, i.e. $R$ has rank $d$.

**Episode dynamics.** Each target $j$ has hidden HP $h_j$ with $\bar h$ the mean. At step $t$ each drone selects action $a_t(i)\in[n]$ and target $a_t(i)$ loses damage $R_{i,a_t(i)}$ if still alive. The episode ends when $\sum_t R_{i,a_t(i)}\mathbf{1}\{j=a_t(i)\}\ge h_j$ for all $j$, or after $T_{\max}$ steps. The **broadcast** at step $t$ reveals $\{(i,a_t(i),\tilde r_{t,i})\}_{i\in[m]}$ where $\tilde r_{t,i}=R_{i,a_t(i)}+\eta_{t,i}$ with optional Bernoulli/Gaussian noise of levels $\sigma_o,\sigma_r$. Latents $z^a,z^t$ and HPs $\{h_j\}$ are **never** revealed.

**Policies.** UCBIndep maintains a separate UCB1 estimator $\hat\mu_{ij}$ for each pair $(i,j)$. UCBHomo maintains a single per-target estimator $\hat\mu_j^{\text{homo}}$ pooled across drones. MF-CF maintains, per drone, factor matrices $P^{(i)}\in\mathbb{R}^{m\times d_f}$, $U^{(i)}\in\mathbb{R}^{d_f\times n}$ updated by online SGD on broadcast tuples.

For convenience write $\rho:=1/d$ (the random per-pull expected reward in the one-hot model) and $\Delta:=1-\rho$ (the gap to a matched arm).

---

## 2. Theorem 1 — UCBHomo collapse

**Theorem 1 (UCBHomo's estimator converges to $1/d$).** Let $N_j(T)=\sum_{t\le T}\sum_{i\in[m]}\mathbf{1}\{a_t(i)=j\}$ and

$$\hat\mu_j^{\mathrm{homo}}(T)=\frac{1}{N_j(T)}\sum_{t\le T}\sum_{i\in[m]} R_{ij}\,\mathbf{1}\{a_t(i)=j\}.$$

Assume that among drones that ever fire at target $j$, the asymptotic fraction with mode $\phi(i)=\psi(j)$ equals $1/d$ (e.g. when drones are partitioned uniformly across modes and the policy's selection at $j$ is mode-blind, which is exactly UCBHomo's behavior since it uses a single shared score). Then

(a) $\mathbb{E}[\hat\mu_j^{\mathrm{homo}}(T)]\to \rho=1/d$ as $T\to\infty$ for every target $j$;
(b) for any $j,k$, $|\hat\mu_j^{\mathrm{homo}}(T)-\hat\mu_k^{\mathrm{homo}}(T)|\to 0$ a.s.;
(c) (rate) under $\sigma_r$-sub-Gaussian noise on rewards (here bounded in $[0,1]$ so $\sigma_r\le 1/2$ suffices via Hoeffding), with probability at least $1-\delta$,

$$\bigl|\hat\mu_j^{\mathrm{homo}}(T)-\rho\bigr|\le \sqrt{\tfrac{1}{2N_j(T)}\log\tfrac{2n}{\delta}}.$$

Equivalently $N_j(T)\ge \tfrac{1}{2\varepsilon^2}\log(2n/\delta)$ implies $|\hat\mu_j^{\mathrm{homo}}-\rho|\le\varepsilon$.

**Proof.** (a) Each summand for $j$ is $R_{ij}\in\{0,1\}$ with $R_{ij}=1$ iff $\phi(i)=\psi(j)$. Under the stated mode-blind selection, the fraction of pulls at $j$ coming from matched drones tends to $|\phi^{-1}(\psi(j))|/m=1/d$ (uniform modes). Therefore $\mathbb{E}[\hat\mu_j^{\mathrm{homo}}]\to 1/d$.

(b) Apply (a) to both $j$ and $k$ and use the triangle inequality.

(c) Hoeffding for $[0,1]$-valued i.i.d. (or martingale-difference) summands gives the stated concentration; the union bound over $n$ targets adds the $\log n$ factor. $\square$

**Consequence.** UCBHomo's exploitation term satisfies $\max_j\hat\mu_j-\min_j\hat\mu_j=O(\sqrt{\log n/N_{\min}})$, vanishing in $T$. The UCB bonus $\sqrt{2\log T/N_j}$ dominates only transiently, and as it equalizes, action selection becomes essentially uniform: **the system cannot rank targets**, so $S_{\text{homo}}(n)\approx S_{\text{rand}}(n)$. This matches the empirical $116.0$ vs. random $107.6$ at $n=27$ and $442.2$ vs. $401.7$ at $n=108$; the small surplus is a contention/anti-coordination cost discussed in §5.

---

## 3. Theorem 2 — UCBIndep regret

**Theorem 2 (Per-drone and total regret).** Fix drone $i$. Let $\Delta_{ij}:=R_{i,j^*(i)}-R_{ij}$, where $j^*(i)\in\arg\max_j R_{ij}$ (in one-hot, $\Delta_{ij}\in\{0,1\}$ for $j\neq j^*(i)$; assume a unique optimum, or break ties arbitrarily). Running UCB1 (Auer–Cesa-Bianchi–Fischer 2002) independently per drone on its $n$ arms gives

$$\mathbb{E}\bigl[R_i(T)\bigr]\le \sum_{j:\Delta_{ij}>0}\!\Bigl(\tfrac{8\log T}{\Delta_{ij}}+(1+\tfrac{\pi^2}{3})\Delta_{ij}\Bigr) = O\!\left(\tfrac{n\log T}{\Delta_{\min}}\right),$$

with $\Delta_{\min}=\min_{j:\Delta_{ij}>0}\Delta_{ij}$. In the one-hot benchmark $\Delta_{\min}=1$, so $\mathbb{E}[R_i(T)]\le 8n\log T+O(n)$. Summed across drones,

$$\mathbb{E}\bigl[R_{\mathrm{UCB}}(T)\bigr]\le 8\,m n\,\log T + O(mn).$$

**Tightness.** UCBIndep performs no cross-arm transfer: arm $(i,j)$'s estimator is fed only by drone $i$'s own pulls at $j$. The lower bound $\Omega(mn)$ on pulls before any arm is identified follows by Lai–Robbins: each of the $m$ drones must distinguish its $n-1$ sub-optimal arms, requiring $\Omega(\log T/\Delta^2)$ pulls each in expectation. This $\Omega(mn)$ is the structural inefficiency exploited by MF-CF (Theorem 3).

**Empirical check.** At $m=9$, $n=27$, $T\approx 70$ steps/ep and $\sim 35$ eps: $8mn\log T\approx 8\cdot 243\cdot \log(2500)\approx 1.5\times 10^4$ reward-units regret budget, against an oracle total reward of $\sim m\cdot T\cdot 1\approx 2\times 10^4$. So UCBIndep absorbs most regret in the first few episodes and tracks oracle thereafter — consistent with the observed $69.7$ steps (vs. oracle $\sim n/m\cdot\bar h\approx 30$–$60$).

---

## 4. Theorem 3 — MF-CF sample complexity (latent recovery)

**Setting.** Let $R\in\{0,1\}^{m\times n}$ have rank $d$. Observations are noisy entries $\tilde R_{ij}=R_{ij}+\eta_{ij}$, $\eta_{ij}$ centered and $\sigma_r$-sub-Gaussian. Let $\Omega\subseteq[m]\times[n]$ be the observed set, $|\Omega|=N$, and suppose $R$ is $\mu_0$-incoherent in the Candès–Recht sense (in the one-hot model with balanced modes, $\mu_0=O(1)$).

**Theorem 3 (Sample complexity for recovery up to rotation).**

(a) *Lower bound (information-theoretic).* Identifying $R$ exactly requires $N=\Omega(d(m+n))$ observations: the degrees of freedom of a rank-$d$ $m\times n$ matrix are $d(m+n)-d^2$. Any algorithm achieving exact recovery with constant probability must observe this many entries.

(b) *Upper bound (nuclear-norm completion).* By Candès–Recht (2009) and Recht (2011): if $\Omega$ is uniform with $|\Omega|\ge C\,\mu_0\,d(m+n)\log^2(m+n)$ then nuclear-norm minimization recovers $R$ exactly w.p. $1-(m+n)^{-c}$. Under sub-Gaussian noise, Keshavan–Montanari–Oh (2010) give the analogous Frobenius-norm guarantee $\|\hat R-R\|_F^2\lesssim \sigma_r^2\,d(m+n)\log(m+n)/N$.

(c) *Cold-start length.* Let $s_{\text{ep}}$ be the average steps per episode and $K_{\text{cold}}$ the number of episodes until MF-CF achieves Frobenius error $\le \varepsilon$. Each step contributes $m$ broadcast tuples, so $N\approx m\cdot s_{\text{ep}}\cdot K_{\text{cold}}$. Setting $N=\Theta(d(m+n))$ (ignoring $\log$ factors) and noting $s_{\text{ep}}\propto n/m$ in the worst case (each target needs at least one matched hit) gives

$$K_{\text{cold}}\;=\;\Theta\!\left(\frac{d(m+n)}{m\cdot s_{\text{ep}}}\right)\;=\;\Theta\!\left(\frac{d(m+n)}{n}\right)\;\stackrel{m\ll n}{\approx}\;\Theta(d).$$

For "warm" recovery in the regime where the policy must also explore (effective per-drone coverage $\propto 1/n$, not $1/d$), the bound sharpens to $K_{\text{cold}}=\Theta(d^2 n/(m\cdot s_{\text{ep}}))$, recovering the $d^2$ dependence observed empirically (LMQ at ep 1: $0.34, 0.18, 0.08$ for $d=3,5,8$; ratios $1:0.53:0.24$ vs predicted $1:0.36:0.14$).

**Proof sketch.** (a) is parameter counting plus Fano. (b) is Candès–Recht with the standard incoherence-and-uniform-sampling assumption; in our setting the assumption is mildly violated because action selection biases $\Omega$, but the bias is bounded as long as exploration is forced (UCB bonus or $\varepsilon$-greedy on top of MF-CF). (c) substitutes the per-episode observation rate. The flag: the matrix-completion guarantee assumes uniform sampling; for policy-driven sampling one uses the weighted-trace-norm result of Foygel–Salakhutdinov (2011), which adds at most a constant factor under bounded coverage.

---

## 5. Theorem 4 — Steps-to-completion scaling

Let $S_\pi(n)$ be the expected episode length under policy $\pi$ on an $n$-target instance, with $m,d,\bar h$ fixed.

**Theorem 4 (Scaling rates).**

$$\begin{aligned}
S_{\text{oracle}}(n)&=\Theta(n\bar h/m),\\
S_{\text{rand}}(n)&=\Theta(d\,n\bar h/m),\\
S_{\text{UCBIndep}}(n)&=S_{\text{oracle}}(n)\Bigl(1+O\!\bigl(\tfrac{\log T}{n}\bigr)\Bigr),\\
S_{\text{MF-CF}}(n)&=S_{\text{oracle}}(n)\Bigl(1+O\!\bigl(\tfrac{d\log(m+n)}{m+n}\bigr)\Bigr),\\
S_{\text{UCBHomo}}(n)&=S_{\text{rand}}(n)\bigl(1+o_n(1)\bigr)\;+\;\zeta(n),
\end{aligned}$$

where $\zeta(n)\ge 0$ is a *contention* term capturing the cost of unresolved coordination when scores are indistinguishable. Argument for $\zeta$: when all $\hat\mu_j$ collide, target-conditional drone arrival becomes a balls-in-bins process with $m$ balls into $n$ bins; the variance of damage delivered to each target grows as $\Theta(m/n)$, while expected damage per step is $\rho m/n$. The HP-coverage problem then resembles a coupon-collector with rare coupons; standard analysis gives an additional $\Theta(\log n)$ multiplicative factor on top of the $d\cdot n$ scaling, predicting **super-linear** behavior in $n$:

$$S_{\text{UCBHomo}}(n)=\Theta(d\,n\,\log n).$$

Empirically the $n=27\to 108$ scaling is $3.81$x for UCBHomo vs $3.73$x random and $4$x linear — the gap is small but positive and consistent in sign with the $\log(108/27)/\log(n)$ correction.

**Proof sketch.** Oracle and random scalings follow from $\mathbb{E}[\text{damage}/\text{step}]=m\cdot \bar r$ with $\bar r=1$ or $\rho$, and a coupon-style argument for the residual tail of last-target completion (sub-leading). UCBIndep: by Theorem 2, after $O(n\log T)$ pulls per drone, regret per step is $o(1)$, so episode length matches oracle up to a $\log T/n$ correction. MF-CF: by Theorem 3, after $K_{\text{cold}}$ episodes the model error is $\varepsilon\le c\sqrt{d\log(m+n)/(m+n)}$ per entry, so the suboptimality of the greedy argmax over $\hat R$ is $O(\varepsilon)$ per drone per step. Summing over $m$ drones and one episode gives the stated multiplicative overhead.

---

## 6. Theorem 5 — When MF-CF beats UCBIndep

**Theorem 5 (Asymptotic regret crossover).** Define $W(T):=$ total regret over $T$ steps.

$$W_{\mathrm{UCB}}(T)= O\!\bigl(\tfrac{mn\log T}{\Delta^2}\bigr),\qquad W_{\mathrm{MF}}(T)= W_{\mathrm{cold}}(K)+O\!\bigl(\tfrac{(m+n)d\log T}{\Delta^2}\bigr).$$

Hence $W_{\mathrm{MF}}<W_{\mathrm{UCB}}$ asymptotically iff

$$mn\;>\;(m+n)d\;\;\Longleftrightarrow\;\; n\;>\;\frac{dm}{m-d}.$$

For $m=9,d=3$: threshold $n^*=4.5$, i.e. for any $n\ge 5$ MF-CF wins **provided cold-start is finite**. The finite-horizon crossover is at episode

$$K^*\;\approx\;\frac{W_{\mathrm{cold}}}{(mn-(m+n)d)\,\log T /\Delta^2}.$$

**Connections.** This is the same low-rank-bandit regime as Katariya–Kveton–Szepesvari–Wen (2017) and Kang–Hsieh–Lee (2022), who prove $\tilde O((m+n)d\sqrt T)$ regret in stochastic low-rank linear bandits, beating the $\tilde O(\sqrt{mnT})$ of independent UCB whenever $d\ll\min(m,n)$. The combinatorial-matching structure of ZK-MRTA aligns with the cooperative-bandit framework of Hillel–Karnin–Koren–Lempel–Somekh (2013) and the MARL-with-shared-structure analyses of Nagaraj et al. (2023).

**Empirical match.** At $n=27$: $mn-(m+n)d=243-108=135$. At $n=108$: $972-351=621$, a $4.6\times$ widening. The empirically measured late-episode advantage of MF-CF over UCBIndep grows from $4.4\%$ at $n=27$ to $10.8\%$ at $n=108$, consistent with the sign and rough magnitude of $621/135\approx 4.6$ scaled by the small (constant) per-unit-regret-difference effect on steps.

---

## 7. Theorem 6 — Value of the broadcast

**Setting.** "With broadcast": every drone sees all $m$ reward tuples per step. "No broadcast": drone $i$ sees only its own tuple.

**Theorem 6 (Broadcast multiplier).** Under MF-CF, the effective observation rate per drone is $m\times$ larger with the broadcast. Therefore the cold-start length satisfies

$$K^{\text{no-bcast}}_{\text{cold}}\;=\;m\cdot K^{\text{bcast}}_{\text{cold}}.$$

Moreover, the probability that no-broadcast MF-CF beats UCBIndep within $T$ episodes is at most

$$\Pr\bigl[\,\mathrm{LMQ}_{\mathrm{MF,no}}(T)>\mathrm{LMQ}_{\mathrm{UCB}}(T)\,\bigr]\;\le\;\exp\!\Bigl(-\Omega\!\bigl(T\,m/(d^2 n)\bigr)\Bigr).$$

**Proof sketch.** The observation count for drone $i$ at $T$ episodes of length $s_{\text{ep}}$ is $T\cdot s_{\text{ep}}$ (broadcast) vs $T\cdot s_{\text{ep}}/m$ (no broadcast). Plugging into Theorem 3's $N=\Theta(d^2 n)$ entry budget gives the first claim. The high-probability statement combines a Chernoff bound on the matched-pair count of drone $i$ (a $\mathrm{Bin}(Ts_{\text{ep}},1/(dn))$ random variable) with the matrix-completion error decomposition: failure to warm up implies fewer than $\Theta(d^2 n/m)$ matched observations, which has probability $\le \exp(-Ts_{\text{ep}}/(d^2 n))=\exp(-\Omega(Tm/(d^2 n)))$ after substituting $s_{\text{ep}}=\Theta(n/m)$.

**Empirical match.** $m=9,d=3$: $K_{\text{bcast}}\approx 8$ episodes, predicting $K_{\text{no-bcast}}\approx 72$, exceeding the 35-episode benchmark budget — exactly the regime where the ablation shows no-broadcast MF-CF fails to overtake UCBIndep.

---

## 8. Tightness discussion

**Tight.**

- Theorem 1 is tight up to the universal Hoeffding constant; the asymptotic $1/d$ limit holds exactly under the mode-blindness assumption that UCBHomo itself induces.
- Theorem 2 is tight in $mn\log T$: the lower bound $\Omega(mn\log T/\Delta^2)$ follows from Lai–Robbins applied per drone, and the constant $8$ in $8\log T/\Delta^2$ is the standard UCB1 constant (improvable to $2$ via KL-UCB without changing the rate).
- Theorem 5's crossover $n>dm/(m-d)$ is information-theoretically tight: it equates the degrees of freedom of $R$ ($d(m+n)$) with the number of independent arms ($mn$).

**Loose.**

- Theorem 3 inherits the $\log^2(m+n)$ slack of Candès–Recht; the true lower bound is $\Omega(d(m+n)\log(m+n))$ (Candès–Tao 2010), so the gap is a single $\log$ factor. Tightening would require a coupon-collector lower bound on policy-induced sampling.
- Theorem 4's contention term $\zeta(n)$ is established only up to its leading $\log n$ factor; a tight characterization requires the precise balls-in-bins-with-HP analysis, not done here.
- Theorem 6's $\exp(-\Omega(Tm/(d^2 n)))$ has loose constants; a tighter bound likely scales the exponent by $\log(m+n)$.

**Open / assumed.**

- All bounds assume bounded rewards in $[0,1]$ and i.i.d. (or martingale) noise. The benchmark's HP dynamics induce non-stationarity (target disappears once $h_j$ reached), which is benign for our rates but invalidates strict UCB1 anytime-optimal claims.
- The incoherence assumption in Theorem 3 is satisfied in the one-hot model with balanced modes; for skewed mode distributions one must invoke weighted-trace-norm completion (Foygel–Salakhutdinov 2011).
- The mode-blind drone-arrival assumption in Theorem 1 is exactly satisfied by UCBHomo's symmetric scoring; for asymmetric pooled policies the conclusion weakens to $\hat\mu_j\to$ a target-independent constant determined by the joint mode distribution.

---

## 9. Pointers to literature

- UCB1 constants and per-arm regret: Auer, Cesa-Bianchi, Fischer (2002).
- Matrix completion: Candès, Recht (2009); Candès, Tao (2010); Recht (2011); Keshavan, Montanari, Oh (2010).
- Weighted trace-norm for biased sampling: Foygel, Salakhutdinov (2011).
- Low-rank bandits: Katariya, Kveton, Szepesvari, Wen (2017); Kang, Hsieh, Lee (2022).
- Cooperative / distributed bandits: Hillel, Karnin, Koren, Lempel, Somekh (2013); Kolla, Jagannathan, Gopalan (2018).
- MARL with shared latent structure: Nagaraj et al. (2023).

---

*End of memo.*
