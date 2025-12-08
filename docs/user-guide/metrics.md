# Metrics & Analysis

Scripts for performance and uncertainty metrics, plus statistical testing:
- `metrics_analysis.py`: aggregates metrics across runs/splits.
- `metrics_stats_significance.py`: normality diagnostics (Shapiro-Wilk), non-parametric tests (Friedman + Nemenyi), pairwise Wilcoxon, Cliff's Delta, bootstrap CIs.

When normality holds, parametric RM-ANOVA + Tukey HSD can be used; otherwise, non-parametric tests are recommended by default.

Outputs include:
- Boxplots and calibration plots
- Critical difference diagrams
- Multiple-comparisons heatmaps (MCS)
- CI forest plots for pairwise differences

Results location:
- Figures and summaries saved under `figures/{data}/{activity}/all/{project}/`.

See README for more details and example figures.