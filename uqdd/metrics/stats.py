"""
Statistical utilities for metrics analysis and significance testing.

This module includes helpers to compute descriptive statistics, confidence intervals,
bootstrap aggregates, correlation and significance tests, and summary tables to
support model evaluation and reporting.
"""

import os
import warnings
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Tuple

import pingouin as pg
import scikit_posthocs as sp
from scipy import stats
from scipy.stats import shapiro, spearmanr, wilcoxon, friedmanchisquare
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.libqsturng import psturng, qsturng

from .analysis import save_plot

INTERACTIVE_MODE = False


def calc_regression_metrics(df, cycle_col, val_col, pred_col, thresh):
    """
    Compute regression and thresholded classification metrics per cycle/method/split.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing true and predicted values.
    cycle_col : str
        Column name identifying cross-validation cycles.
    val_col : str
        Column with true target values.
    pred_col : str
        Column with predicted target values.
    thresh : float
        Threshold to derive binary classes for precision/recall.

    Returns
    -------
    pd.DataFrame
        Metrics per (cv_cycle, method, split) with columns ['mae', 'mse', 'r2', 'rho', 'prec', 'recall'].
    """
    df_in = df.copy()
    metric_ls = ["mae", "mse", "r2", "rho", "prec", "recall"]
    metric_list = []
    df_in["true_class"] = df_in[val_col] > thresh
    assert len(df_in.true_class.unique()) == 2, "Binary classification requires two classes"
    df_in["pred_class"] = df_in[pred_col] > thresh

    for k, v in df_in.groupby([cycle_col, "method", "split"]):
        cycle, method, split = k
        mae = mean_absolute_error(v[val_col], v[pred_col])
        mse = mean_squared_error(v[val_col], v[pred_col])
        r2 = r2_score(v[val_col], v[pred_col])
        recall = recall_score(v.true_class, v.pred_class)
        prec = precision_score(v.true_class, v.pred_class)
        rho, _ = spearmanr(v[val_col], v[pred_col])
        metric_list.append([cycle, method, split, mae, mse, r2, rho, prec, recall])
    metric_df = pd.DataFrame(metric_list, columns=["cv_cycle", "method", "split"] + metric_ls)
    return metric_df


def bootstrap_ci(data, func=np.mean, n_bootstrap=1000, ci=95, random_state=42):
    """
    Compute bootstrap confidence interval for a statistic.

    Parameters
    ----------
    data : array-like
        Sequence of numeric values.
    func : callable, optional
        Statistic function applied to bootstrap samples (e.g., numpy.mean). Default is numpy.mean.
    n_bootstrap : int, optional
        Number of bootstrap resamples. Default is 1000.
    ci : int or float, optional
        Confidence level percentage (e.g., 95). Default is 95.
    random_state : int, optional
        Seed for reproducibility. Default is 42.

    Returns
    -------
    tuple[float, float]
        Lower and upper bounds for the confidence interval.
    """
    np.random.seed(random_state)
    bootstrap_samples = []
    for _ in range(n_bootstrap):
        sample = resample(data, random_state=np.random.randint(0, 10000))
        bootstrap_samples.append(func(sample))
    alpha = (100 - ci) / 2
    lower = np.percentile(bootstrap_samples, alpha)
    upper = np.percentile(bootstrap_samples, 100 - alpha)
    return lower, upper


def rm_tukey_hsd(df, metric, group_col, alpha=0.05, sort=False, direction_dict=None):
    """
    Repeated-measures Tukey HSD approximation using RM-ANOVA and studentized range.

    Parameters
    ----------
    df : pd.DataFrame
        Long-form DataFrame with columns including the metric, group, and 'cv_cycle' subject.
    metric : str
        Metric column to compare.
    group_col : str
        Column indicating groups (e.g., method/model type).
    alpha : float, optional
        Family-wise error rate for intervals. Default is 0.05.
    sort : bool, optional
        If True, sort groups by mean value of the metric. Default is False.
    direction_dict : dict or None, optional
        Mapping of metric -> 'maximize'|'minimize' to set sort ascending/descending.

    Returns
    -------
    tuple
        (result_tab, df_means, df_means_diff, p_values_matrix) where:
        - result_tab: DataFrame of pairwise comparisons with mean differences and CIs.
        - df_means: mean per group.
        - df_means_diff: matrix of mean differences.
        - pc: matrix of adjusted p-values.
    """
    if sort and direction_dict and metric in direction_dict:
        ascending = direction_dict[metric] != "maximize"
        df_means = df.groupby(group_col).mean(numeric_only=True).sort_values(metric, ascending=ascending)
    else:
        df_means = df.groupby(group_col).mean(numeric_only=True)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in scalar divide")
        aov = pg.rm_anova(dv=metric, within=group_col, subject="cv_cycle", data=df, detailed=True)
    mse = aov.loc[1, "MS"]
    df_resid = aov.loc[1, "DF"]

    methods = df_means.index
    n_groups = len(methods)
    n_per_group = df[group_col].value_counts().mean()
    tukey_se = np.sqrt(2 * mse / (n_per_group))
    q = qsturng(1 - alpha, n_groups, df_resid)
    if isinstance(q, (tuple, list, np.ndarray)):
        q = q[0]

    num_comparisons = len(methods) * (len(methods) - 1) // 2
    result_tab = pd.DataFrame(index=range(num_comparisons), columns=["group1", "group2", "meandiff", "lower", "upper", "p-adj"])
    df_means_diff = pd.DataFrame(index=methods, columns=methods, data=0.0)
    pc = pd.DataFrame(index=methods, columns=methods, data=1.0)

    row_idx = 0
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i < j:
                group1 = df[df[group_col] == method1][metric]
                group2 = df[df[group_col] == method2][metric]
                mean_diff = group1.mean() - group2.mean()
                studentized_range = np.abs(mean_diff) / tukey_se
                adjusted_p = psturng(studentized_range * np.sqrt(2), n_groups, df_resid)
                if isinstance(adjusted_p, (tuple, list, np.ndarray)):
                    adjusted_p = adjusted_p[0]
                lower = mean_diff - (q / np.sqrt(2) * tukey_se)
                upper = mean_diff + (q / np.sqrt(2) * tukey_se)
                result_tab.loc[row_idx] = [method1, method2, mean_diff, lower, upper, adjusted_p]
                pc.loc[method1, method2] = adjusted_p
                pc.loc[method2, method1] = adjusted_p
                df_means_diff.loc[method1, method2] = mean_diff
                df_means_diff.loc[method2, method1] = -mean_diff
                row_idx += 1

    df_means_diff = df_means_diff.astype(float)
    result_tab["group1_mean"] = result_tab["group1"].map(df_means[metric])
    result_tab["group2_mean"] = result_tab["group2"].map(df_means[metric])
    result_tab.index = result_tab["group1"] + " - " + result_tab["group2"]
    return result_tab, df_means, df_means_diff, pc


def make_boxplots(df, metric_ls, save_dir=None, name_prefix="", model_order=None):
    """
    Plot boxplots for each metric grouped by method.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    metric_ls : list of str
        Metrics to visualize.
    save_dir : str or None, optional
        Directory to save the plot. Default is None.
    name_prefix : str, optional
        Prefix for the output filename. Default is empty.
    model_order : list of str or None, optional
        Explicit order of methods on the x-axis. Default derives from data.

    Returns
    -------
    None
    """
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    sns.set_theme(context="paper", font_scale=1.5)
    sns.set_style("whitegrid")
    figure, axes = plt.subplots(1, len(metric_ls), sharex=False, sharey=False, figsize=(28, 8))
    for i, stat in enumerate(metric_ls):
        ax = sns.boxplot(y=stat, x="method", hue="method", ax=axes[i], data=df, palette="Set2", legend=False, order=model_order, hue_order=model_order)
        title = stat.upper()
        ax.set_xlabel("")
        ax.set_ylabel(title)
        x_tick_labels = ax.get_xticklabels()
        label_text_list = [x.get_text() for x in x_tick_labels]
        new_xtick_labels = ["\n".join(x.split("_")) for x in label_text_list]
        ax.set_xticks(list(range(0, len(x_tick_labels))))
        ax.set_xticklabels(new_xtick_labels, rotation=45, ha="right")
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    save_plot(figure, save_dir, f"{name_prefix}_boxplot_{'_'.join(metric_ls)}")
    if INTERACTIVE_MODE:
        plt.show()
    plt.close()


def make_boxplots_parametric(df, metric_ls, save_dir=None, name_prefix="", model_order=None):
    """
    Plot boxplots with RM-ANOVA p-values annotated per metric.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    metric_ls : list of str
        Metrics to visualize.
    save_dir : str or None, optional
        Directory to save the plot. Default is None.
    name_prefix : str, optional
        Prefix for the output filename. Default is empty.
    model_order : list of str or None, optional
        Explicit order of methods on the x-axis. Default derives from data.

    Returns
    -------
    None
    """
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    sns.set_theme(context="paper", font_scale=1.5)
    sns.set_style("whitegrid")
    figure, axes = plt.subplots(1, len(metric_ls), sharex=False, sharey=False, figsize=(28, 8))
    for i, stat in enumerate(metric_ls):
        model = AnovaRM(data=df, depvar=stat, subject="cv_cycle", within=["method"]).fit()
        p_value = model.anova_table["Pr > F"].iloc[0]
        ax = sns.boxplot(y=stat, x="method", hue="method", ax=axes[i], data=df, palette="Set2", legend=False, order=model_order, hue_order=model_order)
        title = stat.upper()
        ax.set_title(f"p={p_value:.1e}")
        ax.set_xlabel("")
        ax.set_ylabel(title)
        x_tick_labels = ax.get_xticklabels()
        label_text_list = [x.get_text() for x in x_tick_labels]
        new_xtick_labels = ["\n".join(x.split("_")) for x in label_text_list]
        ax.set_xticks(list(range(0, len(x_tick_labels))))
        ax.set_xticklabels(new_xtick_labels, rotation=45, ha="right")
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    save_plot(figure, save_dir, f"{name_prefix}_boxplot_parametric_{'_'.join(metric_ls)}")
    if INTERACTIVE_MODE:
        plt.show()
    plt.close()


def make_boxplots_nonparametric(df, metric_ls, save_dir=None, name_prefix="", model_order=None):
    """
    Plot boxplots with Friedman p-values annotated per metric.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    metric_ls : list of str
        Metrics to visualize.
    save_dir : str or None, optional
        Directory to save the plot. Default is None.
    name_prefix : str, optional
        Prefix for the output filename. Default is empty.
    model_order : list of str or None, optional
        Explicit order of methods on the x-axis. Default derives from data.

    Returns
    -------
    None
    """
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    n_metrics = len(metric_ls)
    sns.set_theme(context="paper", font_scale=1.5)
    sns.set_style("whitegrid")
    figure, axes = plt.subplots(1, n_metrics, sharex=False, sharey=False, figsize=(28, 8))
    for i, stat in enumerate(metric_ls):
        friedman = pg.friedman(df, dv=stat, within="method", subject="cv_cycle")["p-unc"].values[0]
        ax = sns.boxplot(y=stat, x="method", hue="method", ax=axes[i], data=df, palette="Set2", legend=False, order=model_order, hue_order=model_order)
        title = stat.replace("_", " ").upper()
        ax.set_title(f"p={friedman:.1e}")
        ax.set_xlabel("")
        ax.set_ylabel(title)
        x_tick_labels = ax.get_xticklabels()
        label_text_list = [x.get_text() for x in x_tick_labels]
        new_xtick_labels = ["\n".join(x.split("_")) for x in label_text_list]
        ax.set_xticks(list(range(0, len(x_tick_labels))))
        ax.set_xticklabels(new_xtick_labels, rotation=45, ha="right")
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    save_plot(figure, save_dir, f"{name_prefix}_boxplot_nonparametric_{'_'.join(metric_ls)}")
    if INTERACTIVE_MODE:
        plt.show()
    plt.close()


def make_sign_plots_nonparametric(df, metric_ls, save_dir=None, name_prefix="", model_order=None):
    """
    Plot significance heatmaps (Conover post-hoc) for nonparametric comparisons.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    metric_ls : list of str
        Metrics to analyze.
    save_dir : str or None, optional
        Directory to save the plot. Default is None.
    name_prefix : str, optional
        Prefix for the output filename. Default is empty.
    model_order : list of str or None, optional
        Explicit order of methods on axes. Default derives from data.

    Returns
    -------
    None
    """
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    heatmap_args = {"linewidths": 0.25, "linecolor": "0.5", "clip_on": True, "square": True, "cbar_kws": {"pad": 0.05, "location": "right"}}
    n_metrics = len(metric_ls)
    sns.set_theme(context="paper", font_scale=1.5)
    figure, axes = plt.subplots(1, n_metrics, sharex=False, sharey=True, figsize=(26, 8))
    if n_metrics == 1:
        axes = [axes]
    for i, stat in enumerate(metric_ls):
        pc = sp.posthoc_conover_friedman(df, y_col=stat, group_col="method", block_col="cv_cycle", block_id_col="cv_cycle", p_adjust="holm", melted=True)
        if model_order is not None:
            pc = pc.reindex(index=model_order, columns=model_order)
        sub_ax, sub_c = sp.sign_plot(pc, **heatmap_args, ax=axes[i], xticklabels=True)
        sub_ax.set_title(stat.upper())
        if sub_c is not None and hasattr(sub_c, "ax"):
            figure.subplots_adjust(right=0.85)
            sub_c.ax.set_position([0.87, 0.5, 0.02, 0.2])
    save_plot(figure, save_dir, f"{name_prefix}_sign_plot_nonparametric_{'_'.join(metric_ls)}", tighten=False)
    if INTERACTIVE_MODE:
        plt.show()
    plt.close()


def make_critical_difference_diagrams(df, metric_ls, save_dir=None, name_prefix="", model_order=None):
    """
    Plot critical difference diagrams per metric using average ranks and post-hoc p-values.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    metric_ls : list of str
        Metrics to analyze.
    save_dir : str or None, optional
        Directory to save the plot. Default is None.
    name_prefix : str, optional
        Prefix for the output filename. Default is empty.
    model_order : list of str or None, optional
        Explicit order of models on diagrams. Default derives from data.

    Returns
    -------
    None
    """
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    n_metrics = len(metric_ls)
    figure, axes = plt.subplots(n_metrics, 1, sharex=True, sharey=False, figsize=(16, 10))
    for i, stat in enumerate(metric_ls):
        avg_rank = df.groupby("cv_cycle")[stat].rank(pct=True).groupby(df.method).mean()
        pc = sp.posthoc_conover_friedman(df, y_col=stat, group_col="method", block_col="cv_cycle", block_id_col="cv_cycle", p_adjust="holm", melted=True)
        if model_order is not None:
            avg_rank = avg_rank.reindex(model_order)
            pc = pc.reindex(index=model_order, columns=model_order)
        sp.critical_difference_diagram(avg_rank, pc, ax=axes[i])
        axes[i].set_title(stat.upper())
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    save_plot(figure, save_dir, f"{name_prefix}_critical_difference_diagram_{'_'.join(metric_ls)}")
    if INTERACTIVE_MODE:
        plt.show()
    plt.close()


def make_normality_diagnostic(df, metric_ls, save_dir=None, name_prefix=""):
    """
    Plot normality diagnostics (histogram/KDE and Q-Q) for residualized metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    metric_ls : list of str
        Metrics to diagnose.
    save_dir : str or None, optional
        Directory to save the plot. Default is None.
    name_prefix : str, optional
        Prefix for the output filename. Default is empty.

    Returns
    -------
    None
    """
    df_norm = df.copy()
    df_norm.replace([np.inf, -np.inf], np.nan, inplace=True)
    for metric in metric_ls:
        df_norm[metric] = df_norm[metric] - df_norm.groupby("method")[metric].transform("mean")
    df_norm = df_norm.melt(id_vars=["cv_cycle", "method", "split"], value_vars=metric_ls, var_name="metric", value_name="value")
    sns.set_theme(context="paper", font_scale=1.5)
    sns.set_style("whitegrid")
    metrics = df_norm["metric"].unique()
    n_metrics = len(metrics)
    fig, axes = plt.subplots(2, n_metrics, figsize=(20, 10))
    for i, metric in enumerate(metrics):
        ax = axes[0, i]
        sns.histplot(df_norm[df_norm["metric"] == metric]["value"], kde=True, ax=ax)
        ax.set_title(f"{metric}")
        ax.set_xlabel("")
        if i == 0:
            ax.set_ylabel("Count")
        else:
            ax.set_ylabel("")
    for i, metric in enumerate(metrics):
        ax = axes[1, i]
        metric_data = df_norm[df_norm["metric"] == metric]["value"]
        stats.probplot(metric_data, dist="norm", plot=ax)
        ax.set_title("")
        ax.set_xlabel("Theoretical Quantiles")
        if i == 0:
            ax.set_ylabel("Ordered Values")
        else:
            ax.set_ylabel("")
    plt.subplots_adjust(hspace=0.3, wspace=0.8)
    save_plot(fig, save_dir, f"{name_prefix}_normality_diagnostic_{'_'.join(metric_ls)}")
    if INTERACTIVE_MODE:
        plt.show()
    plt.close()


def mcs_plot(pc, effect_size, means, labels=True, cmap=None, cbar_ax_bbox=None, ax=None, show_diff=True, cell_text_size=10, axis_text_size=8, show_cbar=True, reverse_cmap=False, vlim=None, **kwargs):
    """
    Render a multiple-comparisons significance heatmap annotated with effect sizes and stars.

    Parameters
    ----------
    pc : pd.DataFrame
        Matrix of adjusted p-values.
    effect_size : pd.DataFrame
        Matrix of mean differences (effect sizes) aligned with `pc`.
    means : pd.Series
        Mean values per group for labeling.
    labels : bool, optional
        If True, add x/y tick labels from `means.index`. Default is True.
    cmap : str or None, optional
        Colormap name for effect sizes. Default is 'YlGnBu'.
    cbar_ax_bbox : tuple or None, optional
        Custom colorbar axes bbox; unused here but kept for API compatibility.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw into; if None, a new axes is created.
    show_diff : bool, optional
        If True, annotate cells with rounded effect sizes plus significance. Default is True.
    cell_text_size : int, optional
        Font size for annotations. Default is 10.
    axis_text_size : int, optional
        Font size for axis tick labels. Default is 8.
    show_cbar : bool, optional
        If True, show colorbar. Default is True.
    reverse_cmap : bool, optional
        If True, use reversed colormap. Default is False.
    vlim : float or None, optional
        Symmetric limit for color scaling around 0. Default is None.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the rendered heatmap.
    """
    for key in ["cbar", "vmin", "vmax", "center"]:
        if key in kwargs:
            del kwargs[key]
    if not cmap:
        cmap = "YlGnBu"
    if reverse_cmap:
        cmap = cmap + "_r"
    significance = pc.copy().astype(object)
    significance[(pc < 0.001) & (pc >= 0)] = "***"
    significance[(pc < 0.01) & (pc >= 0.001)] = "**"
    significance[(pc < 0.05) & (pc >= 0.01)] = "*"
    significance[(pc >= 0.05)] = ""
    np.fill_diagonal(significance.values, "")
    annotations = effect_size.round(2).astype(str) + significance if show_diff else significance
    hax = sns.heatmap(effect_size, cmap=cmap, annot=annotations, fmt="", cbar=show_cbar, ax=ax, annot_kws={"size": cell_text_size}, vmin=-2 * vlim if vlim else None, vmax=2 * vlim if vlim else None, square=True, **kwargs)
    if labels:
        label_list = list(means.index)
        x_label_list = label_list
        y_label_list = label_list
        xtick_positions = np.arange(len(label_list))
        hax.set_xticks(xtick_positions + 0.5)
        hax.set_xticklabels(x_label_list, size=axis_text_size, ha="center", va="center", rotation=90)
        hax.set_yticks(xtick_positions + 0.5)
        hax.set_yticklabels(y_label_list, size=axis_text_size, ha="center", va="center", rotation=0)
    hax.set_xlabel("")
    hax.set_ylabel("")
    return hax


def make_mcs_plot_grid(df, stats_list, group_col, alpha=0.05, figsize=(20, 10), direction_dict=None, effect_dict=None, show_diff=True, cell_text_size=16, axis_text_size=12, title_text_size=16, sort_axes=False, save_dir=None, name_prefix="", model_order=None):
    """
    Generate a grid of MCS plots for multiple metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    stats_list : list of str
        Metrics to include.
    group_col : str
        Column indicating groups (e.g., method).
    alpha : float, optional
        Significance level. Default is 0.05.
    figsize : tuple, optional
        Figure size. Default is (20, 10).
    direction_dict : dict or None, optional
        Mapping metric -> 'maximize'|'minimize' for colormap orientation.
    effect_dict : dict or None, optional
        Mapping metric -> effect size limit for color scaling.
    show_diff : bool, optional
        If True, annotate mean differences; else annotate significance only.
    cell_text_size : int, optional
        Annotation font size.
    axis_text_size : int, optional
        Axis label font size.
    title_text_size : int, optional
        Title font size.
    sort_axes : bool, optional
        If True, sort groups by mean values per metric.
    save_dir : str or None, optional
        Directory to save the plot. Default is None.
    name_prefix : str, optional
        Filename prefix. Default is empty.
    model_order : list of str or None, optional
        Explicit model order for rows/cols.

    Returns
    -------
    None
    """
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    nrow = math.ceil(len(stats_list) / 3)
    fig, ax = plt.subplots(nrow, 3, figsize=figsize)
    for key in ["r2", "rho", "prec", "recall", "mae", "mse"]:
        direction_dict.setdefault(key, "maximize" if key in ["r2", "rho", "prec", "recall"] else "minimize")
    for key in ["r2", "rho", "prec", "recall"]:
        effect_dict.setdefault(key, 0.1)
    for i, stat in enumerate(stats_list):
        row = i // 3
        col = i % 3
        if stat not in direction_dict:
            raise ValueError(f"Stat '{stat}' is missing in direction_dict. Please set its value.")
        if stat not in effect_dict:
            raise ValueError(f"Stat '{stat}' is missing in effect_dict. Please set its value.")
        reverse_cmap = direction_dict[stat] == "minimize"
        _, df_means, df_means_diff, pc = rm_tukey_hsd(df, stat, group_col, alpha, sort_axes, direction_dict)
        if model_order is not None:
            df_means = df_means.reindex(model_order)
            df_means_diff = df_means_diff.reindex(index=model_order, columns=model_order)
            pc = pc.reindex(index=model_order, columns=model_order)
        hax = mcs_plot(pc, effect_size=df_means_diff, means=df_means[stat], show_diff=show_diff, ax=ax[row, col], cbar=True, cell_text_size=cell_text_size, axis_text_size=axis_text_size, reverse_cmap=reverse_cmap, vlim=effect_dict[stat])
        hax.set_title(stat.upper(), fontsize=title_text_size)
    if (len(stats_list) % 3) != 0:
        for i in range(len(stats_list), nrow * 3):
            row = i // 3
            col = i % 3
            ax[row, col].set_visible(False)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="p < 0.001 (***): Highly Significant", markerfacecolor="black", markersize=10),
        Line2D([0], [0], marker="o", color="w", label="p < 0.01 (**): Very Significant", markerfacecolor="black", markersize=10),
        Line2D([0], [0], marker="o", color="w", label="p < 0.05 (*): Significant", markerfacecolor="black", markersize=10),
        Line2D([0], [0], marker="o", color="w", label="p >= 0.05: Not Significant", markerfacecolor="black", markersize=10),
    ]
    fig.legend(handles=legend_elements, loc="upper right", ncol=2, fontsize=12, frameon=False)
    plt.subplots_adjust(top=0.88)
    save_plot(fig, save_dir, f"{name_prefix}_mcs_plot_grid_{'_'.join(stats_list)}")
    if INTERACTIVE_MODE:
        plt.show()
    plt.close()


def make_scatterplot(df, val_col, pred_col, thresh, cycle_col="cv_cycle", group_col="method", save_dir=None):
    """
    Scatter plots of predicted vs true values per method, with threshold lines and summary stats.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    val_col : str
        True value column.
    pred_col : str
        Predicted value column.
    thresh : float
        Threshold for classification overlays.
    cycle_col : str, optional
        Cross-validation cycle column. Default is 'cv_cycle'.
    group_col : str, optional
        Method/model type column. Default is 'method'.
    save_dir : str or None, optional
        Directory to save the plot. Default is None.

    Returns
    -------
    None
    """
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_split_metrics = calc_regression_metrics(df, cycle_col=cycle_col, val_col=val_col, pred_col=pred_col, thresh=thresh)
    methods = df[group_col].unique()
    fig, axs = plt.subplots(nrows=1, ncols=len(methods), figsize=(25, 10))
    for ax, method in zip(axs, methods):
        df_method = df.query(f"{group_col} == @method")
        df_metrics = df_split_metrics.query(f"{group_col} == @method")
        ax.scatter(df_method[pred_col], df_method[val_col], alpha=0.3)
        ax.plot([df_method[val_col].min(), df_method[val_col].max()], [df_method[val_col].min(), df_method[val_col].max()], "k--", lw=1)
        ax.axhline(y=thresh, color="r", linestyle="--")
        ax.axvline(x=thresh, color="r", linestyle="--")
        ax.set_title(method)
        y_true = df_method[val_col] > thresh
        y_pred = df_method[pred_col] > thresh
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        metrics_text = f"MAE: {df_metrics['mae'].mean():.2f}\nMSE: {df_metrics['mse'].mean():.2f}\nR2: {df_metrics['r2'].mean():.2f}\nrho: {df_metrics['rho'].mean():.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}"
        ax.text(0.05, 0.5, metrics_text, transform=ax.transAxes, verticalalignment="top")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Measured")
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    save_plot(fig, save_dir, f"scatterplot_{val_col}_vs_{pred_col}")
    if INTERACTIVE_MODE:
        plt.show()
    plt.close()


def ci_plot(result_tab, ax_in, name):
    """
    Plot mean differences with confidence intervals for pairwise comparisons.

    Parameters
    ----------
    result_tab : pd.DataFrame
        Output of rm_tukey_hsd with columns ['meandiff', 'lower', 'upper'].
    ax_in : matplotlib.axes.Axes
        Axes to plot into.
    name : str
        Title for the plot.

    Returns
    -------
    None
    """
    result_err = np.array([result_tab["meandiff"] - result_tab["lower"], result_tab["upper"] - result_tab["meandiff"]])
    sns.set_theme(context="paper")
    sns.set_style("whitegrid")
    ax = sns.pointplot(x=result_tab.meandiff, y=result_tab.index, marker="o", linestyle="", ax=ax_in)
    ax.errorbar(y=result_tab.index, x=result_tab["meandiff"], xerr=result_err, fmt="o", capsize=5)
    ax.axvline(0, ls="--", lw=3)
    ax.set_xlabel("Mean Difference")
    ax.set_ylabel("")
    ax.set_title(name)
    ax.set_xlim(-0.2, 0.2)


def make_ci_plot_grid(df_in, metric_list, group_col="method", save_dir=None, name_prefix="", model_order=None):
    """
    Plot a grid of confidence-interval charts for multiple metrics.

    Parameters
    ----------
    df_in : pd.DataFrame
        Input DataFrame.
    metric_list : list of str
        Metrics to render.
    group_col : str, optional
        Group column (e.g., 'method'). Default is 'method'.
    save_dir : str or None, optional
        Directory to save the plot. Default is None.
    name_prefix : str, optional
        Filename prefix. Default is empty.
    model_order : list of str or None, optional
        Explicit row order for the CI plots.

    Returns
    -------
    None
    """
    df_in = df_in.copy()
    df_in.replace([np.inf, -np.inf], np.nan, inplace=True)
    figure, axes = plt.subplots(len(metric_list), 1, figsize=(8, 2 * len(metric_list)), sharex=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for i, metric in enumerate(metric_list):
        df_tukey, _, _, _ = rm_tukey_hsd(df_in, metric, group_col=group_col)
        if model_order is not None:
            df_tukey = df_tukey.reindex(index=model_order)
        ci_plot(df_tukey, ax_in=axes[i], name=metric)
    figure.suptitle("Multiple Comparison of Means\nTukey HSD, FWER=0.05")
    plt.subplots_adjust(hspace=0.9, wspace=0.3)
    save_plot(figure, save_dir, f"{name_prefix}_ci_plot_grid_{'_'.join(metric_list)}")
    if INTERACTIVE_MODE:
        plt.show()
    plt.close()


def recall_at_precision(y_true, y_score, precision_threshold=0.5, direction="greater"):
    """
    Find recall and threshold achieving at least a target precision.

    Parameters
    ----------
    y_true : array-like
        Binary ground-truth labels.
    y_score : array-like
        Continuous scores or probabilities.
    precision_threshold : float, optional
        Minimum precision to achieve. Default is 0.5.
    direction : {"greater", "lesser"}, optional
        If 'greater', thresholding uses >=; if 'lesser', uses <=. Default is 'greater'.

    Returns
    -------
    tuple[float, float or None]
        (recall, threshold) if achievable; otherwise (nan, None).

    Raises
    ------
    ValueError
        If `direction` is invalid.
    """
    if direction not in ["greater", "lesser"]:
        raise ValueError("Invalid direction. Expected one of: ['greater', 'lesser']")
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    thresholds = np.unique(y_score)
    thresholds = np.sort(thresholds)
    if direction == "lesser":
        thresholds = thresholds[::-1]
    for threshold in thresholds:
        y_pred = y_score >= threshold if direction == "greater" else y_score <= threshold
        precision = precision_score(y_true, y_pred)
        if precision >= precision_threshold:
            recall = recall_score(y_true, y_pred)
            return recall, threshold
    return np.nan, None


def calc_classification_metrics(df_in, cycle_col, val_col, prob_col, pred_col):
    """
    Compute classification metrics per cycle/method/split, including ROC-AUC, PR-AUC, MCC, recall, and TNR.

    Parameters
    ----------
    df_in : pd.DataFrame
        Input DataFrame.
    cycle_col : str
        Column name for cross-validation cycles.
    val_col : str
        True binary label column.
    prob_col : str
        Predicted probability/score column.
    pred_col : str
        Predicted binary label column.

    Returns
    -------
    pd.DataFrame
        Metrics per (cv_cycle, method, split) with columns ['roc_auc', 'pr_auc', 'mcc', 'recall', 'tnr'].
    """
    metric_list = []
    for k, v in df_in.groupby([cycle_col, "method", "split"]):
        cycle, method, split = k
        roc_auc = roc_auc_score(v[val_col], v[prob_col])
        pr_auc = average_precision_score(v[val_col], v[prob_col])
        mcc = matthews_corrcoef(v[val_col], v[pred_col])
        recall, _ = recall_at_precision(v[val_col].astype(bool), v[prob_col], precision_threshold=0.8, direction="greater")
        tnr, _ = recall_at_precision(~v[val_col].astype(bool), v[prob_col], precision_threshold=0.8, direction="lesser")
        metric_list.append([cycle, method, split, roc_auc, pr_auc, mcc, recall, tnr])
    metric_df = pd.DataFrame(metric_list, columns=["cv_cycle", "method", "split", "roc_auc", "pr_auc", "mcc", "recall", "tnr"])
    return metric_df


def make_curve_plots(df):
    """
    Plot ROC and PR curves for split/method selections with threshold markers.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing 'cv_cycle', 'split', and method columns plus true/probability fields.

    Returns
    -------
    None
    """
    df_plot = df.query("cv_cycle == 0 and split == 'scaffold'").copy()
    color_map = plt.get_cmap("tab10")
    le = LabelEncoder()
    df_plot["color"] = le.fit_transform(df_plot["method"])
    colors = color_map(df_plot["color"].unique())
    val_col = "Sol"
    prob_col = "Sol_prob"
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for (k, v), color in zip(df_plot.groupby("method"), colors):
        roc_auc = roc_auc_score(v[val_col], v[prob_col])
        pr_auc = average_precision_score(v[val_col], v[prob_col])
        fpr, recall_pos, thresholds_roc = roc_curve(v[val_col], v[prob_col])
        precision, recall, thresholds_pr = precision_recall_curve(v[val_col], v[prob_col])
        _, threshold_recall_pos = recall_at_precision(v[val_col].astype(bool), v[prob_col], precision_threshold=0.8, direction="greater")
        _, threshold_recall_neg = recall_at_precision(~v[val_col].astype(bool), v[prob_col], precision_threshold=0.8, direction="lesser")
        fpr_recall_pos = fpr[np.abs(thresholds_roc - threshold_recall_pos).argmin()]
        fpr_recall_neg = fpr[np.abs(thresholds_roc - threshold_recall_neg).argmin()]
        recall_recall_pos = recall[np.abs(thresholds_pr - threshold_recall_pos).argmin()]
        recall_recall_neg = recall[np.abs(thresholds_pr - threshold_recall_neg).argmin()]
        axes[0].plot(fpr, recall_pos, label=f"{k} (ROC AUC={roc_auc:.03f})", color=color, alpha=0.75)
        axes[1].plot(recall, precision, label=f"{k} (PR AUC={pr_auc:.03f})", color=color, alpha=0.75)
        axes[0].axvline(fpr_recall_pos, color=color, linestyle=":", alpha=0.75)
        axes[0].axvline(fpr_recall_neg, color=color, linestyle="--", alpha=0.75)
        axes[1].axvline(recall_recall_pos, color=color, linestyle=":", alpha=0.75)
        axes[1].axvline(recall_recall_neg, color=color, linestyle="--", alpha=0.75)
    axes[0].plot([0, 1], [0, 1], "--", color="black", lw=0.5)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()


def harmonize_columns(df):
    """
    Normalize common column names to ['method', 'split', 'cv_cycle'].

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with possibly varied column naming.

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized column names and assertion that required columns exist.
    """
    df = df.copy()
    rename_map = {
        "Model type": "method",
        "Split": "split",
        "Group_Number": "cv_cycle",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
    assert {"method", "split", "cv_cycle"}.issubset(df.columns)
    return df


def cliffs_delta(x, y):
    """
    Compute Cliff's delta effect size and qualitative interpretation.

    Parameters
    ----------
    x : array-like
        First sample of numeric values.
    y : array-like
        Second sample of numeric values.

    Returns
    -------
    tuple[float, str]
        (delta, interpretation) where interpretation is one of {'negligible','small','medium','large'}.
    """
    x, y = np.array(x), np.array(y)
    m, n = len(x), len(y)
    comparisons = 0
    for xi in x:
        for yi in y:
            if xi > yi:
                comparisons += 1
            elif xi < yi:
                comparisons -= 1
    delta = comparisons / (m * n)
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        interpretation = "negligible"
    elif abs_delta < 0.33:
        interpretation = "small"
    elif abs_delta < 0.474:
        interpretation = "medium"
    else:
        interpretation = "large"
    return delta, interpretation


def wilcoxon_pairwise_test(df, metric, model_a, model_b, task=None, split=None, seed_col=None):
    """
    Perform paired Wilcoxon signed-rank test between two models on a metric.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    metric : str
        Metric column to compare.
    model_a : str
        First model type name.
    model_b : str
        Second model type name.
    task : str or None, optional
        Task filter. Default is None.
    split : str or None, optional
        Split filter. Default is None.
    seed_col : str or None, optional
        Optional seed column identifier (unused here).

    Returns
    -------
    dict or None
        Test summary including statistic, p-value, Cliff's delta, CI on differences; None if insufficient data.
    """
    data = df.copy()
    if task is not None:
        data = data[data["Task"] == task]
    if split is not None:
        data = data[data["Split"] == split]
    values_a = data[data["Model type"] == model_a][metric].values
    values_b = data[data["Model type"] == model_b][metric].values
    if len(values_a) == 0 or len(values_b) == 0:
        return None
    min_len = min(len(values_a), len(values_b))
    values_a = values_a[:min_len]
    values_b = values_b[:min_len]
    statistic, p_value = wilcoxon(values_a, values_b, alternative="two-sided")
    delta, effect_size_interpretation = cliffs_delta(values_a, values_b)
    differences = values_a - values_b
    median_diff = np.median(differences)
    ci_lower, ci_upper = bootstrap_ci(differences, np.median, n_bootstrap=1000)
    if ci_lower <= 0 <= ci_upper:
        practical_significance = "difference is small (CI includes 0)"
    elif abs(median_diff) < 0.1 * np.std(np.concatenate([values_a, values_b])):
        practical_significance = "difference is small"
    else:
        practical_significance = "difference may be meaningful"
    return {
        "model_a": model_a,
        "model_b": model_b,
        "metric": metric,
        "task": task,
        "split": split,
        "n_pairs": min_len,
        "wilcoxon_statistic": statistic,
        "p_value": p_value,
        "cliffs_delta": delta,
        "effect_size_interpretation": effect_size_interpretation,
        "median_difference": median_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "practical_significance": practical_significance,
    }


def holm_bonferroni_correction(p_values):
    """
    Apply Holm–Bonferroni correction to an array of p-values.

    Parameters
    ----------
    p_values : array-like
        Raw p-values.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        (corrected_p_values, rejected_mask) where rejected indicates significance after correction.
    """
    p_values = np.array(p_values)
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p_values = p_values[sorted_indices]
    corrected_p_values = np.zeros(n)
    rejected = np.zeros(n, dtype=bool)
    for i in range(n):
        correction_factor = n - i
        corrected_p_values[sorted_indices[i]] = min(1.0, sorted_p_values[i] * correction_factor)
        if corrected_p_values[sorted_indices[i]] < 0.05:
            rejected[sorted_indices[i]] = True
        else:
            break
    return corrected_p_values, rejected


def pairwise_model_comparison(df, metrics, models=None, tasks=None, splits=None, alpha=0.05):
    """
    Run pairwise Wilcoxon tests across models/tasks/splits for multiple metrics and adjust p-values.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    metrics : list of str
        Metrics to compare.
    models : list of str or None, optional
        Models to include; default derives from data.
    tasks : list of str or None, optional
        Tasks to include; default derives from data.
    splits : list of str or None, optional
        Splits to include; default derives from data.
    alpha : float, optional
        Significance level. Default is 0.05.

    Returns
    -------
    pd.DataFrame
        Results table with corrected p-values and significance flags.
    """
    if models is None:
        models = df["Model type"].unique()
    if tasks is None:
        tasks = df["Task"].unique()
    if splits is None:
        splits = df["Split"].unique()
    results = []
    for metric in metrics:
        for task in tasks:
            for split in splits:
                for i, model_a in enumerate(models):
                    for j, model_b in enumerate(models):
                        if i < j:
                            result = wilcoxon_pairwise_test(df, metric, model_a, model_b, task, split)
                            if result is not None:
                                results.append(result)
    if not results:
        return pd.DataFrame()
    results_df = pd.DataFrame(results)
    p_values = results_df["p_value"].values
    corrected_p_values, rejected = holm_bonferroni_correction(p_values)
    results_df["corrected_p_value"] = corrected_p_values
    results_df["significant_after_correction"] = rejected
    return results_df


def friedman_nemenyi_test(df, metrics, models=None, alpha=0.05):
    """
    Run Friedman test across models with Nemenyi post-hoc where significant, per metric.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    metrics : list of str
        Metrics to test.
    models : list of str or None, optional
        Models to include; default derives from data.
    alpha : float, optional
        Significance level. Default is 0.05.

    Returns
    -------
    dict
        Mapping metric -> result dict containing stats, p-values, mean ranks, and optional post-hoc outputs.
    """
    if models is None:
        models = df["Model type"].unique()
    results = {}
    for metric in metrics:
        pivot_data = df.pivot_table(values=metric, index=["Task", "Split"], columns="Model type", aggfunc="mean")
        available_models = [m for m in models if m in pivot_data.columns]
        pivot_data = pivot_data[available_models]
        pivot_data = pivot_data.dropna()
        if pivot_data.shape[0] < 2 or pivot_data.shape[1] < 3:
            results[metric] = {"error": "Insufficient data for Friedman test", "data_shape": pivot_data.shape}
            continue
        try:
            friedman_stat, friedman_p = friedmanchisquare(*[pivot_data[col].values for col in pivot_data.columns])
            ranks = pivot_data.rank(axis=1, ascending=False)
            mean_ranks = ranks.mean()
            result = {
                "friedman_statistic": friedman_stat,
                "friedman_p_value": friedman_p,
                "mean_ranks": mean_ranks.to_dict(),
                "significant": friedman_p < alpha,
            }
            if friedman_p < alpha:
                try:
                    data_array = pivot_data.values
                    nemenyi_result = sp.posthoc_nemenyi_friedman(data_array.T)
                    nemenyi_result.index = available_models
                    nemenyi_result.columns = available_models
                    result["nemenyi_p_values"] = nemenyi_result.to_dict()
                    result["critical_difference"] = calculate_critical_difference(len(available_models), pivot_data.shape[0], alpha)
                except Exception as e:
                    result["nemenyi_error"] = str(e)
            results[metric] = result
        except Exception as e:
            results[metric] = {"error": str(e)}
    return results


def calculate_critical_difference(k, n, alpha=0.05):
    """
    Compute the critical difference for average ranks in Nemenyi post-hoc tests.

    Parameters
    ----------
    k : int
        Number of models.
    n : int
        Number of datasets/blocks.
    alpha : float, optional
        Significance level. Default is 0.05.

    Returns
    -------
    float
        Critical difference value.
    """
    from scipy.stats import studentized_range
    q_alpha = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))
    return cd


def bootstrap_auc_difference(auc_values_a, auc_values_b, n_bootstrap=1000, ci=95, random_state=42):
    """
    Bootstrap confidence interval for difference of mean AUCs between two models.

    Parameters
    ----------
    auc_values_a : array-like
        AUC values for model A.
    auc_values_b : array-like
        AUC values for model B.
    n_bootstrap : int, optional
        Number of bootstrap resamples. Default is 1000.
    ci : int or float, optional
        Confidence level in percent. Default is 95.
    random_state : int, optional
        Seed for reproducibility. Default is 42.

    Returns
    -------
    dict
        {'mean_difference', 'ci_lower', 'ci_upper', 'bootstrap_differences'}
    """
    np.random.seed(random_state)
    differences = []
    for _ in range(n_bootstrap):
        sample_a = resample(auc_values_a, random_state=np.random.randint(0, 10000))
        sample_b = resample(auc_values_b, random_state=np.random.randint(0, 10000))
        diff = np.mean(sample_a) - np.mean(sample_b)
        differences.append(diff)
    differences = np.array(differences)
    alpha = (100 - ci) / 2
    ci_lower = np.percentile(differences, alpha)
    ci_upper = np.percentile(differences, 100 - alpha)
    original_diff = np.mean(auc_values_a) - np.mean(auc_values_b)
    return {"mean_difference": original_diff, "ci_lower": ci_lower, "ci_upper": ci_upper, "bootstrap_differences": differences}


def plot_critical_difference_diagram(friedman_results, metric, save_dir=None, alpha=0.05):
    """
    Plot a simple critical difference diagram using mean ranks and CD value.

    Parameters
    ----------
    friedman_results : dict
        Output dictionary from friedman_nemenyi_test.
    metric : str
        Metric to plot.
    save_dir : str or None, optional
        Directory to save the plot. Default is None.
    alpha : float, optional
        Significance level used to compute CD. Default is 0.05.

    Returns
    -------
    None
    """
    if metric not in friedman_results:
        print(f"Metric {metric} not found in Friedman results")
        return
    result = friedman_results[metric]
    if "error" in result:
        print(f"Error in Friedman test for {metric}: {result['error']}")
        return
    if not result["significant"]:
        print(f"Friedman test not significant for {metric}, skipping CD diagram")
        return
    mean_ranks = result["mean_ranks"]
    models = list(mean_ranks.keys())
    ranks = [mean_ranks[model] for model in models]
    sorted_indices = np.argsort(ranks)
    sorted_models = [models[i] for i in sorted_indices]
    sorted_ranks = [ranks[i] for i in sorted_indices]
    fig, ax = plt.subplots(figsize=(12, 6))
    y_pos = 0
    ax.scatter(sorted_ranks, [y_pos] * len(sorted_ranks), s=100, c="blue")
    for i, (model, rank) in enumerate(zip(sorted_models, sorted_ranks)):
        ax.annotate(model, (rank, y_pos), xytext=(0, 20), textcoords="offset points", ha="center", rotation=45)
    if "critical_difference" in result:
        cd = result["critical_difference"]
        groups = []
        for i, model_a in enumerate(sorted_models):
            group = [model_a]
            rank_a = sorted_ranks[i]
            for j, model_b in enumerate(sorted_models):
                if i != j:
                    rank_b = sorted_ranks[j]
                    if abs(rank_a - rank_b) <= cd:
                        if model_b not in [m for g in groups for m in g]:
                            group.append(model_b)
            if len(group) > 1:
                groups.append(group)
        colors = plt.cm.Set3(np.linspace(0, 1, len(groups)))
        for group, color in zip(groups, colors):
            if len(group) > 1:
                group_ranks = [sorted_ranks[sorted_models.index(m)] for m in group]
                min_rank, max_rank = min(group_ranks), max(group_ranks)
                ax.plot([min_rank, max_rank], [y_pos - 0.05, y_pos - 0.05], color=color, linewidth=3, alpha=0.7)
    ax.set_xlim(min(sorted_ranks) - 0.5, max(sorted_ranks) + 0.5)
    ax.set_ylim(-0.3, 0.5)
    ax.set_xlabel("Average Rank")
    ax.set_title(f"Critical Difference Diagram - {metric}")
    ax.grid(True, alpha=0.3)
    ax.set_yticks([])
    if save_dir:
        plot_name = f"critical_difference_{metric.replace(' ', '_')}"
        save_plot(fig, save_dir, plot_name)
    if INTERACTIVE_MODE:
        plt.show()
    plt.close()


def analyze_significance(df_raw, metrics, direction_dict, effect_dict, save_dir=None, model_order=None, activity=None):
    """
    End-to-end significance analysis and plotting across splits for multiple metrics.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw results DataFrame.
    metrics : list of str
        Metric names to analyze.
    direction_dict : dict
        Mapping metric -> 'maximize'|'minimize'.
    effect_dict : dict
        Mapping metric -> effect size threshold for visualization.
    save_dir : str or None, optional
        Directory to save plots and outputs. Default is None.
    model_order : list of str or None, optional
        Explicit ordering of models. Default derives from data.
    activity : str or None, optional
        Activity name for prefixes. Default is None.

    Returns
    -------
    None
    """
    df = harmonize_columns(df_raw)
    for metric in metrics:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for split in df["split"].unique():
        df_s = df[df["split"] == split].copy()
        print(f"\n=== Split: {split} ===")
        name_prefix = f"06_{activity}_{split}" if activity else f"{split}"
        make_normality_diagnostic(df_s, metrics, save_dir=save_dir, name_prefix=name_prefix)
        for metric in metrics:
            print(f"\n-- Metric: {metric}")
            wide = df_s.pivot(index="cv_cycle", columns="method", values=metric)
            resid = (wide.T - wide.mean(axis=1)).T
            vals = resid.values.flatten()
            vals = vals[~np.isnan(vals)]
            W, p_norm = shapiro(vals) if len(vals) >= 3 else (None, 0.0)
            if p_norm is None:
                print("Not enough data for Shapiro-Wilk test (need at least 3 non-NaN values), assuming non-normality")
            elif p_norm < 0.05:
                print(f"Shapiro-Wilk test for {metric} indicates non-normality (W={W:.3f}, p={p_norm:.3f})")
            else:
                print(f"Shapiro-Wilk test for {metric} indicates normality (W={W:.3f}, p={p_norm:.3f})")
        make_boxplots(df_s, metrics, save_dir=save_dir, name_prefix=name_prefix, model_order=model_order)
        make_boxplots_parametric(df_s, metrics, save_dir=save_dir, name_prefix=name_prefix, model_order=model_order)
        make_boxplots_nonparametric(df_s, metrics, save_dir=save_dir, name_prefix=name_prefix, model_order=model_order)
        make_sign_plots_nonparametric(df_s, metrics, save_dir=save_dir, name_prefix=name_prefix, model_order=model_order)
        make_critical_difference_diagrams(df_s, metrics, save_dir=save_dir, name_prefix=name_prefix, model_order=model_order)
        make_mcs_plot_grid(df=df_s, stats_list=metrics, group_col="method", alpha=0.05, figsize=(30, 15), direction_dict=direction_dict, effect_dict=effect_dict, show_diff=True, sort_axes=True, save_dir=save_dir, name_prefix=name_prefix + "_diff", model_order=model_order)
        make_mcs_plot_grid(df=df_s, stats_list=metrics, group_col="method", alpha=0.05, figsize=(30, 15), direction_dict=direction_dict, effect_dict=effect_dict, show_diff=False, sort_axes=True, save_dir=save_dir, name_prefix=name_prefix, model_order=model_order)
        make_ci_plot_grid(df_s, metrics, group_col="method", save_dir=save_dir, name_prefix=name_prefix, model_order=model_order)


def comprehensive_statistical_analysis(df, metrics, models=None, tasks=None, splits=None, save_dir=None, alpha=0.05):
    """
    Run a comprehensive suite of statistical tests and export results.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    metrics : list of str
        Metrics to analyze.
    models : list of str or None, optional
        Models to include. Default derives from data.
    tasks : list of str or None, optional
        Tasks to include. Default derives from data.
    splits : list of str or None, optional
        Splits to include. Default derives from data.
    save_dir : str or None, optional
        Directory to save tables and JSON outputs. Default is None.
    alpha : float, optional
        Significance level. Default is 0.05.

    Returns
    -------
    dict
        Results dict including pairwise tests, Friedman/Nemenyi outputs, and optional AUC bootstrap comparisons.
    """
    print("Performing comprehensive statistical analysis...")
    results = {}
    print("1. Running pairwise Wilcoxon signed-rank tests...")
    pairwise_results = pairwise_model_comparison(df, metrics, models, tasks, splits, alpha)
    results["pairwise_tests"] = pairwise_results
    print("2. Running Friedman tests with Nemenyi post-hoc...")
    friedman_results = friedman_nemenyi_test(df, metrics, models, alpha)
    results["friedman_nemenyi"] = friedman_results
    auc_columns = [col for col in df.columns if "AUC" in col or "auc" in col]
    if auc_columns:
        print("3. Running bootstrap comparisons for AUC metrics...")
        auc_bootstrap_results = {}
        for auc_col in auc_columns:
            auc_bootstrap_results[auc_col] = {}
            available_models = df["Model type"].unique() if models is None else models
            for i, model_a in enumerate(available_models):
                for j, model_b in enumerate(available_models):
                    if i < j:
                        auc_a = df[df["Model type"] == model_a][auc_col].dropna().values
                        auc_b = df[df["Model type"] == model_b][auc_col].dropna().values
                        if len(auc_a) > 0 and len(auc_b) > 0:
                            bootstrap_result = bootstrap_auc_difference(auc_a, auc_b)
                            auc_bootstrap_results[auc_col][f"{model_a}_vs_{model_b}"] = bootstrap_result
        results["auc_bootstrap"] = auc_bootstrap_results
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        if not pairwise_results.empty:
            pairwise_results.to_csv(os.path.join(save_dir, "pairwise_statistical_tests.csv"), index=False)
        import json
        with open(os.path.join(save_dir, "friedman_nemenyi_results.json"), "w") as f:
            json_compatible_results = {}
            for metric, result in friedman_results.items():
                json_compatible_results[metric] = {}
                for key, value in result.items():
                    if isinstance(value, (np.ndarray, np.generic)):
                        json_compatible_results[metric][key] = value.tolist()
                    elif isinstance(value, dict):
                        json_compatible_results[metric][key] = {str(k): (float(v) if isinstance(v, (np.ndarray, np.generic)) else v) for k, v in value.items()}
                    else:
                        json_compatible_results[metric][key] = (float(value) if isinstance(value, (np.ndarray, np.generic)) else value)
            json.dump(json_compatible_results, f, indent=2)
        if auc_columns:
            with open(os.path.join(save_dir, "auc_bootstrap_results.json"), "w") as f:
                json_compatible_auc = {}
                for auc_col, comparisons in results["auc_bootstrap"].items():
                    json_compatible_auc[auc_col] = {}
                    for comparison, result in comparisons.items():
                        json_compatible_auc[auc_col][comparison] = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in result.items()}
                json.dump(json_compatible_auc, f, indent=2)
    return results


def generate_statistical_report(results, save_dir=None, df_raw=None, metrics=None, direction_dict=None, effect_dict=None):
    """
    Generate a human-readable text report from comprehensive statistical results and optionally run plots.

    Parameters
    ----------
    results : dict
        Output of comprehensive_statistical_analysis.
    save_dir : str or None, optional
        Directory to save the report text file. Default is None.
    df_raw : pd.DataFrame or None, optional
        Raw DataFrame to run plotting-based significance analysis. Default is None.
    metrics : list of str or None, optional
        Metrics to plot (when df_raw provided).
    direction_dict : dict or None, optional
        Direction mapping for metrics (required when df_raw provided).
    effect_dict : dict or None, optional
        Effect threshold mapping (required when df_raw provided).

    Returns
    -------
    str
        Report text.
    """
    report = []
    report.append("=" * 80)
    report.append("COMPREHENSIVE STATISTICAL ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")
    if "pairwise_tests" in results and not results["pairwise_tests"].empty:
        pairwise_df = results["pairwise_tests"]
        report.append("1. PAIRWISE MODEL COMPARISONS (Wilcoxon Signed-Rank Test)")
        report.append("-" * 60)
        significant = pairwise_df[pairwise_df["significant_after_correction"] == True]
        report.append(f"Total pairwise comparisons performed: {len(pairwise_df)}")
        report.append(f"Significant differences (after Holm-Bonferroni correction): {len(significant)}")
        report.append("")
        if len(significant) > 0:
            report.append("Significant differences found:")
            for _, row in significant.iterrows():
                effect_size = row["effect_size_interpretation"]
                report.append(f"  • {row['model_a']} vs {row['model_b']} ({row['metric']}, {row['split']}):")
                report.append(f"    - p-value: {row['p_value']:.4f} (corrected: {row['corrected_p_value']:.4f})")
                report.append(f"    - Cliff's Δ: {row['cliffs_delta']:.3f} ({effect_size} effect)")
                report.append(f"    - Median difference: {row['median_difference']:.4f} [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]")
                report.append(f"    - {row['practical_significance']}")
                report.append("")
        else:
            report.append("No significant differences found after multiple comparison correction.")
            report.append("")
    if "friedman_nemenyi" in results:
        friedman_results = results["friedman_nemenyi"]
        report.append("2. MULTIPLE MODEL COMPARISONS (Friedman + Nemenyi Tests)")
        report.append("-" * 60)
        for metric, result in friedman_results.items():
            if "error" in result:
                report.append(f"{metric}: {result['error']}")
                continue
            report.append(f"Metric: {metric}")
            report.append(f"  Friedman test p-value: {result['friedman_p_value']:.4f}")
            if result["significant"]:
                report.append("  Result: Significant difference between models detected")
                mean_ranks = result["mean_ranks"]
                sorted_ranks = sorted(mean_ranks.items(), key=lambda x: x[1])
                report.append("  Model rankings (lower rank = better performance):")
                for i, (model, rank) in enumerate(sorted_ranks, 1):
                    report.append(f"    {i}. {model}: {rank:.2f}")
                if "critical_difference" in result:
                    report.append(f"  Critical difference: {result['critical_difference']:.3f}")
            else:
                report.append("  Result: No significant difference between models")
            report.append("")
    if "auc_bootstrap" in results:
        auc_results = results["auc_bootstrap"]
        report.append("3. AUC BOOTSTRAP COMPARISONS")
        report.append("-" * 60)
        for auc_col, comparisons in auc_results.items():
            report.append(f"AUC Metric: {auc_col}")
            for comparison, result in comparisons.items():
                model_a, model_b = comparison.split("_vs_")
                mean_diff = result["mean_difference"]
                ci_lower = result["ci_lower"]
                ci_upper = result["ci_upper"]
                significance = "difference is small (CI includes 0)" if (ci_lower <= 0 <= ci_upper) else "difference may be meaningful"
                report.append(f"  {model_a} vs {model_b}:")
                report.append(f"    Mean difference: {mean_diff:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
                report.append(f"    {significance}")
            report.append("")
    report_text = "\n".join(report)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "statistical_analysis_report.txt"), "w") as f:
            f.write(report_text)
    print(report_text)
    if df_raw is not None and metrics is not None and direction_dict is not None and effect_dict is not None:
        analyze_significance(df_raw, metrics, direction_dict, effect_dict, save_dir=save_dir)
    return report_text

