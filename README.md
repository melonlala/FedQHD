# Federated Q Learning with Hyperdimensional Approximation (FedQHD)

This repository contains the LaTeX source for the FedQHD research paper.

## Project Structure

```
FedQHD/
├── main.tex                 # Main LaTeX file (compile this)
├── main.bib                 # Bibliography file
├── rlj.sty                  # RLJ conference style file
├── rlj.bst                  # RLJ bibliography style file
├── sections/                # Paper sections
│   ├── intro.tex           # Introduction
│   ├── related_work.tex    # Related Work
│   ├── preliminaries.tex   # Background and Preliminaries
│   ├── methodology.tex     # Proposed Methodology
│   ├── experiments.tex     # Experimental Setup
│   ├── results.tex         # Experimental Results
│   ├── discussion.tex      # Discussion
│   └── conclusion.tex      # Conclusion
└── figures/                 # Directory for figures and images
```

## Compilation

### Using Overleaf
1. Upload all files to Overleaf
2. Set the main document to `main.tex`
3. Click "Recompile"

### Using Local LaTeX
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or use latexmk for automatic compilation:
```bash
latexmk -pdf main.tex
```

## Writing Guidelines

### Format Reminders (from main.tex)
- **Citations**: Use `\citet{}` for in-text citations, `\citep{}` for parenthetical
- **Figures**: Always reference figures before they appear; use `\ref{fig:label}`
- **Tables**: Caption goes BEFORE the table; use `\ref{tab:label}`
- **Equations**: Use align environment with `&` after `=` or inequality symbols
- **Sections**: Use `\section`, `\subsection`, and `\subsubsection` for structure

### Package Warnings
The following packages are already included in `rlj.sty` and should NOT be re-included:
1. fancyhdr, 2. natbib, 3. enumitem, 4. fontenc, 5. times, 6. ragged2e
7. tcolorbox, 8. hyperref, 9. xcolor, 10. amsmath, 11. etoolbox, 12. lineno

### Submission Options
- For submission: `\usepackage{rlj}`
- For camera-ready: `\usepackage[accepted]{rlj}`
- For preprint: `\usepackage[preprint]{rlj}`

## Paper Sections Overview

### Introduction (sections/intro.tex)
- Motivation and problem statement
- Gap in existing approaches
- Brief overview of proposed approach
- Main contributions
- Paper organization

### Related Work (sections/related_work.tex)
- Review of relevant literature
- Comparison with existing approaches

### Preliminaries (sections/preliminaries.tex)
- Background on federated learning
- Background on Q-learning
- Background on hyperdimensional computing

### Methodology (sections/methodology.tex)
- Problem formulation
- Hyperdimensional approximation approach
- Federated learning protocol
- Main algorithm

### Experiments (sections/experiments.tex)
- Experimental setup
- Environments and tasks
- Baseline methods
- Evaluation metrics
- Hyperparameters

### Results (sections/results.tex)
- Performance comparison
- Communication efficiency
- Scalability analysis
- Ablation study

### Discussion (sections/discussion.tex)
- Key findings
- Limitations
- Future directions

### Conclusion (sections/conclusion.tex)
- Summary of contributions
- Main findings
- Broader impact

## Page Limit
The paper has an 8-12 page limit (not including references and supplementary materials).

## References
Add references to `main.bib` in BibTeX format.

## Figures
Place all figures in the `figures/` directory. Reference them using:
```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.8\linewidth]{figures/your_figure.pdf}
    \caption{Your caption here.}
    \label{fig:your_label}
\end{figure}
```

## TODO
- [ ] Complete Introduction section
- [ ] Add related work literature review
- [ ] Formalize problem in Methodology
- [ ] Design and run experiments
- [ ] Analyze and present results
- [ ] Write discussion and conclusions
- [ ] Create figures and tables
- [ ] Proofread and refine
