# Ablation study

Our ablation study is done on CartPole environment. 
## Matrix:
1. Federation Gap
Oracle regression Q value $\hat{Q}_i$ is calculated by solving regression $Q^*$ on $\phi(i)$. $Q^*$ is estimated by fully training single QHD agent with huge D and using it to generate MC returns on the randomly chosen test state sets. 

Federation error in FedQHD is calculated by the sup norm between fedQHD's output after 500 episodes and $\hat{Q}_i$. 
2. Policy Value
Estimate oracle policy by MC roll out.
Same for the compiled fedqhd policy.

Calculate the policy value gap.


\begin{itemize}
\item Embedding dimension: $D \in \{128, 256, 512, 1024, 2048, 4096\}$; fixed $D_\mathrm{fixed}=4096$
\item Anchor size: $m \in \{200, 500, 1000, 2000, 5000, 8000\}$; fixed $m_\mathrm{fixed}=2000$
\item Regularization: $\lambda \in \{10^{-6},10^{-4},10^{-2},0.1,1,10\}$ (A3 sweeps $\lambda$ directly; paper notation uses $\alpha=\lambda/m$); fixed $\lambda_\mathrm{fixed}=10^{-4}$
\end{itemize}

Each experiment is averaged over 3 random seeds.

### Implementation notes (2026-02-28)

**$Q^*$ estimation.**
Reference agent: $D_\mathrm{ref}=8192$, 1000 training episodes, $\mathrm{lr}=0.05$.
After training set $\epsilon\leftarrow0$ (greedy). For each anchor $s_j$ and action $a$: force first step $(s_j,a)$, follow greedy policy up to 500 steps, average 30 independent rollouts → $Q^*(s_j,a)$.
MC rollouts are **vectorized** via pure NumPy CartPole physics (no gym overhead): all $m\times|\mathcal{A}|\times n_\mathrm{mc}$ rollouts run as one batched loop over timesteps; RFF encoding chunked to ≤4096 rollouts at a time (~128 MB).
Reference agent auto-saved to `results/ablation/ref_agent.npz` and loaded on re-runs.

**Oracle regression $\hat{Q}_i$.**
$\hat{Q}_i = X_i(X_i^\top X_i + \varepsilon I)^{-1}X_i^\top Q^*$, $\varepsilon=10^{-10}$ (pseudoinverse projection — best-in-class approximation of $Q^*$ in $\mathrm{col}(X_i)$).

**Federation error.**
$\|\hat{Q}_i^\mathrm{fed} - \hat{Q}_i\|_\infty$ (element-wise sup-norm over all anchors and actions) after 500 FedQHD training episodes.

% -------------------------------------------------
\paragraph{Ablation 1: Geometry Floor (Vary $D$, Fix $m \gg D$)}

\paragraph{Protocol.}
Fix $m = 4D$ and small $\alpha$.
Vary $D$ only.

\paragraph{Expected behavior.}
Since sample and regularization terms are suppressed,
error should follow:

\[
\|\Delta_i\| \propto D^{-1/2}.
\]

\paragraph{Figure format.}

\begin{itemize}
\item X-axis: $D$
\item Y-axis: error
\item Log-log scale
\item Plot theoretical slope $-1/2$ as reference line
\end{itemize}

\paragraph{Result interpretation.}
If slope $\approx -0.5$, geometry floor is validated.

% -------------------------------------------------
\paragraph{Ablation 2: Sample Regime Transition (Vary $m/D$)}

\paragraph{Protocol.}
Fix $D=1048$, $\alpha$ small.
Vary $m$ across under/over-parameterized regimes.

\paragraph{Expected behavior.}

\[
\|\Delta_i\|
\propto
\begin{cases}
\sqrt{D/m}, & m < D \\
D^{-1/2}, & m \gg D
\end{cases}
\]

\paragraph{Figure format.}

\begin{itemize}
\item X-axis: $m/D$
\item Y-axis: Error
\item Vertical line at $m/D=1$
\end{itemize}

This plot should show:

\begin{itemize}
\item Sharp decay when $m < D$
\item Plateau when $m \ge D$
\end{itemize}

\paragraph{Key observation.}
Transition near $m \approx D$ confirms theorem separation.

% -------------------------------------------------
\paragraph{Ablation 3: Regularization–Sample Coupling (Vary $\alpha$)}

\paragraph{Protocol.}
Fix $D$ and $m$.
Vary $\alpha = \lambda/m$.

\paragraph{Expected U-shape behavior.}

Small $\alpha$:
variance dominates.

Large $\alpha$:
bias dominates.

Optimal $\alpha^*$ predicted from:

\[
\alpha^*
\sim
\sqrt{\frac{D^2}{m}}.
\]

\paragraph{Figure format.}

\begin{itemize}
\item X-axis: $\alpha$ (log scale)
\item Y-axis: Error
\item Mark predicted $\alpha^*$
\end{itemize}

\paragraph{Result analysis.}
Empirical minimum near predicted scaling validates decomposition.

