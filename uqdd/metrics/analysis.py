# Core analysis and plotting functions extracted from metrics_analysis.py
"""
Analysis and plotting utilities for model metrics.

This module provides functions to aggregate experiment results, compute summary
statistics, and visualize metrics via pairplots, line plots, histograms, bar plots,
correlation matrices, calibration curves, and RMSE rejection curves.
"""
import os
import shutil
import sys
import warnings
from typing import List, Dict, Optional, Union, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from sklearn.metrics import mean_squared_error, auc

from .constants import group_order

INTERACTIVE_MODE = hasattr(sys, "ps1") or sys.flags.interactive


def aggregate_results_csv(
    df: pd.DataFrame,
    group_cols: List[str],
    numeric_cols: List[str],
    string_cols: List[str],
    order_by: Optional[Union[str, List[str]]] = None,
    output_file_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Aggregate metrics by groups and export a compact CSV summary.

    Parameters
    ----------
    df : pd.DataFrame
        Input results DataFrame.
    group_cols : list of str
        Column names to group by.
    numeric_cols : list of str
        Numeric metric columns to aggregate with mean and std.
    string_cols : list of str
        String columns to aggregate as lists.
    order_by : str or list of str or None, optional
        Column(s) to sort the final aggregated DataFrame by. Default is None.
    output_file_path : str or None, optional
        Path to write the aggregated CSV. If None, no file is written.

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame with combined mean(std) strings plus string/list aggregates.

    Notes
    -----
    - A helper column `project_model` is constructed and included in the aggregates.
    - When `output_file_path` is provided, the function ensures the directory exists.
    """
    grouped = df.groupby(group_cols)
    aggregated = grouped[numeric_cols].agg(["mean", "std"])
    for col in numeric_cols:
        aggregated[(col, "combined")] = (
            aggregated[(col, "mean")].round(3).astype(str)
            + "("
            + aggregated[(col, "std")].round(3).astype(str)
            + ")"
        )
    aggregated = aggregated[[col for col in aggregated.columns if col[1] == "combined"]]
    aggregated.columns = [col[0] for col in aggregated.columns]

    string_aggregated = grouped[string_cols].agg(lambda x: list(x))

    df["project_model"] = (
        "papyrus"
        + "/"
        + df["Activity"]
        + "/"
        + "all"
        + "/"
        + df["wandb project"]
        + "/"
        + df["model name"]
        + "/"
    )
    project_model_aggregated = grouped["project_model"].agg(lambda x: list(x))

    final_aggregated = pd.concat(
        [aggregated, string_aggregated, project_model_aggregated], axis=1
    ).reset_index()

    if order_by:
        final_aggregated = final_aggregated.sort_values(by=order_by)

    if output_file_path:
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        final_aggregated.to_csv(output_file_path, index=False)

    return final_aggregated


def save_plot(
    fig: plt.Figure,
    save_dir: Optional[str],
    plot_name: str,
    tighten: bool = True,
    show_legend: bool = False,
) -> None:
    """
    Save a matplotlib figure to PNG, SVG, and PDF with optional tight layout.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    save_dir : str or None
        Directory to save the figure files. If None, no files are written.
    plot_name : str
        Base filename (without extension).
    tighten : bool, optional
        If True, apply tight_layout and bbox_inches="tight". Default is True.
    show_legend : bool, optional
        If False, remove legend before saving. Default is False.

    Returns
    -------
    None
    """
    ax = fig.gca()
    if not show_legend:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    if tighten:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="This figure includes Axes that are not compatible with tight_layout",
                )
                fig.tight_layout()
        except (ValueError, RuntimeError):
            fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    if save_dir and tighten:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f"{plot_name}.png"), dpi=300, bbox_inches="tight")
        fig.savefig(os.path.join(save_dir, f"{plot_name}.svg"), bbox_inches="tight")
        fig.savefig(os.path.join(save_dir, f"{plot_name}.pdf"), dpi=300, bbox_inches="tight")
    elif save_dir and not tighten:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f"{plot_name}.png"), dpi=300)
        fig.savefig(os.path.join(save_dir, f"{plot_name}.svg"))
        fig.savefig(os.path.join(save_dir, f"{plot_name}.pdf"), dpi=300)


def handle_inf_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace +/- infinity values in a DataFrame with NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with infinite values replaced by NaN.
    """
    return df.replace([float("inf"), -float("inf")], float("nan"))


def plot_pairplot(
    df: pd.DataFrame,
    title: str,
    metrics: List[str],
    save_dir: Optional[str] = None,
    cmap: str = "viridis",
    group_order: Optional[List[str]] = group_order,
    show_legend: bool = False,
) -> None:
    """
    Plot a seaborn pairplot for a set of metrics colored by Group.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the metrics and a 'Group' column.
    title : str
        Plot title.
    metrics : list of str
        Metric column names to include in the pairplot.
    save_dir : str or None, optional
        Directory to save plot images. Default is None.
    cmap : str, optional
        Seaborn/matplotlib palette name. Default is "viridis".
    group_order : list of str or None, optional
        Order of class labels in the legend. Default is from constants.
    show_legend : bool, optional
        If True, keep the legend; otherwise it will be removed before saving.

    Returns
    -------
    None
    """
    df = handle_inf_values(df)
    sns.pairplot(
        df,
        hue="Group",
        hue_order=group_order,
        vars=metrics,
        palette=cmap,
        plot_kws={"alpha": 0.7},
    )
    plt.suptitle(title, y=1.02)
    plot_name = f"pairplot_{title.replace(' ', '_')}"
    save_plot(plt.gcf(), save_dir, plot_name, tighten=False, show_legend=show_legend)
    if INTERACTIVE_MODE:
        plt.show()
    plt.close()


def plot_line_metrics(
    df: pd.DataFrame,
    title: str,
    metrics: List[str],
    save_dir: Optional[str] = None,
    group_order: Optional[List[str]] = group_order,
    show_legend: bool = False,
) -> None:
    """
    Plot line charts of metrics over runs, colored by Group.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with 'wandb run', metrics, and 'Group'.
    title : str
        Plot title.
    metrics : list of str
        Metric column names to plot.
    save_dir : str or None, optional
        Directory to save plot images. Default is None.
    group_order : list of str or None, optional
        Order of class labels in the legend. Default is from constants.
    show_legend : bool, optional
        If True, keep the legend; otherwise it will be removed before saving.

    Returns
    -------
    None
    """
    df = handle_inf_values(df)
    for metric in metrics:
        plt.figure(figsize=(14, 7))
        sns.lineplot(
            data=df,
            x="wandb run",
            y=metric,
            hue="Group",
            marker="o",
            palette="Set2",
            hue_order=group_order,
            label=metric,
        )
        plt.title(f"{title} - {metric}")
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        if INTERACTIVE_MODE:
            plt.show()
        plot_name = f"line_{title.replace(' ', '_')}_{metric}"
        save_plot(plt.gcf(), save_dir, plot_name, tighten=False, show_legend=show_legend)
        plt.close()


def plot_histogram_metrics(
    df: pd.DataFrame,
    title: str,
    metrics: List[str],
    save_dir: Optional[str] = None,
    group_order: Optional[List[str]] = group_order,
    cmap: str = "crest",
    show_legend: bool = False,
) -> None:
    """
    Plot histograms with KDE for metrics, split by Group.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with metrics and 'Group'.
    title : str
        Plot title.
    metrics : list of str
        Metric column names to plot.
    save_dir : str or None, optional
        Directory to save plot images. Default is None.
    group_order : list of str or None, optional
        Order of class labels in the legend. Default is from constants.
    cmap : str, optional
        Seaborn/matplotlib palette name. Default is "crest".
    show_legend : bool, optional
        If True, keep the legend; otherwise it will be removed before saving.

    Returns
    -------
    None
    """
    df = handle_inf_values(df)
    for metric in metrics:
        plt.figure(figsize=(14, 7))
        sns.histplot(
            data=df,
            x=metric,
            hue="Group",
            kde=True,
            palette=cmap,
            element="step",
            hue_order=group_order,
            fill=True,
            alpha=0.7,
        )
        plt.title(f"{title} - {metric}")
        if INTERACTIVE_MODE:
            plt.show()
        plot_name = f"histogram_{title.replace(' ', '_')}_{metric}"
        save_plot(plt.gcf(), save_dir, plot_name, show_legend=show_legend)
        plt.close()


def plot_pairwise_scatter_metrics(
    df: pd.DataFrame,
    title: str,
    metrics: List[str],
    save_dir: Optional[str] = None,
    group_order: Optional[List[str]] = group_order,
    cmap: str = "tab10_r",
    show_legend: bool = False,
) -> None:
    """
    Plot pairwise scatterplots for all metric combinations, colored by Group.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with metrics and 'Group'.
    title : str
        Plot title.
    metrics : list of str
        Metric column names to plot pairwise.
    save_dir : str or None, optional
        Directory to save plot images. Default is None.
    group_order : list of str or None, optional
        Order of class labels in the legend. Default is from constants.
    cmap : str, optional
        Matplotlib palette name. Default is "tab10_r".
    show_legend : bool, optional
        If True, keep the legend; otherwise it will be removed before saving.

    Returns
    -------
    None
    """
    df = handle_inf_values(df)
    num_metrics = len(metrics)
    fig, axes = plt.subplots(num_metrics, num_metrics, figsize=(15, 15))

    for i in range(num_metrics):
        for j in range(num_metrics):
            if i != j:
                ax = sns.scatterplot(
                    data=df,
                    x=metrics[j],
                    y=metrics[i],
                    hue="Group",
                    palette=cmap,
                    hue_order=group_order,
                    ax=axes[i, j],
                    legend=False if not (i == 1 and j == 0) else "brief",
                )
                if i == 1 and j == 0:
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend().remove()
            else:
                axes[i, j].set_visible(False)

            axes[i, j].set_ylabel(metrics[i] if j == 0 and i > 0 else "")
            axes[i, j].set_xlabel(metrics[j] if i == num_metrics - 1 else "")

    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.15, 1))
    fig.suptitle(title, y=1.02)
    fig.subplots_adjust(top=0.95, wspace=0.4, hspace=0.4)
    plot_name = f"pairwise_scatter_{title.replace(' ', '_')}"
    save_plot(fig, save_dir, plot_name, tighten=True, show_legend=show_legend)
    if INTERACTIVE_MODE:
        plt.show()
    plt.close()


def plot_metrics(
    df: pd.DataFrame,
    metrics: List[str],
    cmap: str = "tab10_r",
    save_dir: Optional[str] = None,
    hatches_dict: Optional[Dict[str, str]] = None,
    group_order: Optional[List[str]] = None,
    show: bool = True,
    fig_width: Optional[float] = None,
    fig_height: Optional[float] = None,
    show_legend: bool = False,
) -> Dict[str, str]:
    """
    Plot grouped bar charts showing mean and std for metrics across splits and model types.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with columns ['Split', 'Model type'] and metrics.
    metrics : list of str
        Metric column names to plot.
    cmap : str, optional
        Matplotlib colormap name used to derive distinct colors per model type. Default is "tab10_r".
    save_dir : str or None, optional
        Directory to save plot images. Default is None.
    hatches_dict : dict[str, str] or None, optional
        Mapping from Split to hatch pattern. Default is None.
    group_order : list of str or None, optional
        Order of grouped labels (Split_Model type). Default derives from data.
    show : bool, optional
        If True, display plot in interactive mode. Default is True.
    fig_width : float or None, optional
        Width of the plot area (excluding legend). Default scales with number of metrics.
    fig_height : float or None, optional
        Height of the plot area (excluding legend). Default is 6.
    show_legend : bool, optional
        If True, include a legend of split/model combinations. Default is False.

    Returns
    -------
    dict[str, str]
        Color mapping from 'Model type' to RGBA string used in the plot.
    """
    plot_width = fig_width if fig_width else max(10, len(metrics) * 2)
    plot_height = fig_height if fig_height else 6
    total_width = plot_width + 5
    total_height = plot_height + 2

    fig = plt.figure(figsize=(total_width, total_height))
    gs = gridspec.GridSpec(1, 1, figure=fig, left=0.1, right=0.75, top=0.9, bottom=0.2)
    ax = fig.add_subplot(gs[0])
    ax.set_position([0.1, 0.15, plot_width / total_width, plot_height / total_height])

    stats_dfs = []
    for metric in metrics:
        mean_df = df.groupby(["Split", "Model type"])[metric].mean().rename(f"{metric}_mean")
        std_df = df.groupby(["Split", "Model type"])[metric].std().rename(f"{metric}_std")
        stats_df = pd.merge(mean_df, std_df, left_index=True, right_index=True).reset_index()
        stats_df["Group"] = stats_df.apply(lambda row: f"{row['Split']}_{row['Model type']}", axis=1)
        stats_df["Metric"] = metric
        stats_dfs.append(stats_df)

    combined_stats_df = pd.concat(stats_dfs)
    if group_order:
        combined_stats_df["Group"] = pd.Categorical(
            combined_stats_df["Group"], categories=group_order, ordered=True
        )
    else:
        group_order = combined_stats_df["Group"].unique().tolist()

    scalar_mappable = ScalarMappable(cmap=cmap)
    model_types = combined_stats_df["Model type"].unique()
    color_dict = {
        m: c
        for m, c in zip(
            model_types,
            scalar_mappable.to_rgba(range(len(model_types)), alpha=1).tolist(),
        )
    }

    bar_width = 0.12
    group_spacing = 0.4
    num_bars = len(model_types) * len(hatches_dict)
    positions = []
    tick_positions = []
    tick_labels = []

    for i, metric in enumerate(metrics):
        metric_data = combined_stats_df[combined_stats_df["Metric"] == metric]
        metric_data.loc[:, "Group"] = pd.Categorical(
            metric_data["Group"], categories=group_order, ordered=True
        )
        metric_data = metric_data.sort_values("Group").reset_index(drop=True)
        for j, (_, row) in enumerate(metric_data.iterrows()):
            position = i * (num_bars * bar_width + group_spacing) + (j % num_bars) * bar_width
            positions.append(position)
            ax.bar(
                position,
                height=row[f"{metric}_mean"],
                color=color_dict[row["Model type"]],
                hatch=hatches_dict[row["Split"]],
                width=bar_width,
            )
        center_position = i * (num_bars * bar_width + group_spacing) + (num_bars * bar_width) / 2
        tick_positions.append(center_position)
        tick_labels.append(metric.replace(" ", "\n") if " " in metric else metric)

    def create_stats_legend(df, color_mapping, hatches_dict, group_order):
        patches_dict = {}
        for _, row in df.iterrows():
            label = f"{row['Split']} {row['Model type']}"
            group_label = f"{row['Split']}_{row['Model type']}"
            if group_label not in patches_dict:
                patches_dict[group_label] = mpatches.Patch(
                    facecolor=color_mapping[row["Model type"]],
                    hatch=hatches_dict[row["Split"]],
                    label=label,
                )
        return [patches_dict[group] for group in group_order if group in patches_dict]

    if show_legend:
        legend_elements = create_stats_legend(combined_stats_df, color_dict, hatches_dict, group_order)
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0, frameon=False)

    for (_, row), bar in zip(combined_stats_df.iterrows(), ax.patches):
        x_bar = bar.get_x() + bar.get_width() / 2
        y_bar = bar.get_height()
        ax.errorbar(
            x_bar,
            y_bar,
            yerr=row[f"{row['Metric']}_std"],
            color="black",
            fmt="none",
            elinewidth=1,
            capsize=3,
            alpha=0.5,
        )

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", rotation_mode="anchor", fontsize=9)
    ax.set_ylim(bottom=0.0)

    if save_dir:
        metrics_names = "_".join(metrics)
        plot_name = f"barplot_{cmap}_{metrics_names}"
        save_plot(fig, save_dir, plot_name, show_legend=show_legend)

    if INTERACTIVE_MODE:
        plt.show()
    plt.close()

    return color_dict


def find_highly_correlated_metrics(
    df: pd.DataFrame,
    metrics: List[str],
    threshold: float = 0.8,
    save_dir: Optional[str] = None,
    cmap: str = "coolwarm",
    show_legend: bool = False,
) -> List[Tuple[str, str, float]]:
    """
    Identify pairs of metrics with correlation above a threshold and plot the matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the metric columns.
    metrics : list of str
        Metric column names to include in the correlation analysis.
    threshold : float, optional
        Absolute correlation threshold for reporting pairs. Default is 0.8.
    save_dir : str or None, optional
        Directory to save the heatmap plot. Default is None.
    cmap : str, optional
        Matplotlib colormap name. Default is "coolwarm".
    show_legend : bool, optional
        If True, keep the legend; otherwise it will be removed before saving.

    Returns
    -------
    list of tuple[str, str, float]
        List of metric pairs and their absolute correlation values.
    """
    corr_matrix = df[metrics].corr().abs()
    pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] > threshold:
                pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

    print(f"Highly correlated metrics (correlation coefficient > {threshold}):")
    for a, b, v in pairs:
        print(f"{a} and {b}: {v:.2f}")

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap=cmap)
    plt.title("Correlation Matrix")
    plot_name = f"correlation_matrix_{threshold}_{'_'.join(metrics)}"
    save_plot(plt.gcf(), save_dir, plot_name, show_legend=show_legend)
    if INTERACTIVE_MODE:
        plt.show()

    return pairs


def plot_comparison_metrics(
    df: pd.DataFrame,
    metrics: List[str],
    cmap: str = "tab10_r",
    color_dict: Optional[Dict[str, str]] = None,
    save_dir: Optional[str] = None,
    fig_width: Optional[float] = None,
    fig_height: Optional[float] = None,
    show_legend: bool = False,
    models_order: Optional[List[str]] = None,
) -> None:
    """
    Plot comparison bar charts across splits, model types, and calibration states.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with columns ['Split', 'Model type', 'Calibration'] and metrics.
    metrics : list of str
        Metric column names to plot.
    cmap : str, optional
        Matplotlib colormap name used to derive distinct colors per model type. Default is "tab10_r".
    color_dict : dict[str, str] or None, optional
        Precomputed color mapping from model type to color. If None, one is generated.
    save_dir : str or None, optional
        Directory to save plot images. Default is None.
    fig_width : float or None, optional
        Width of the plot area (excluding legend). Default scales with the number of metrics.
    fig_height : float or None, optional
        Height of the plot area (excluding legend). Default is 6.
    show_legend : bool, optional
        If True, include a legend. Default is False.
    models_order : list of str or None, optional
        Explicit order of model types for coloring and grouping. Default derives from data.

    Returns
    -------
    None
    """
    plot_width = fig_width if fig_width else max(7, len(metrics) * 3)
    plot_height = fig_height if fig_height else 6
    total_width = plot_width + 5
    total_height = plot_height + 2

    fig = plt.figure(figsize=(total_width, total_height))
    gs = gridspec.GridSpec(1, 1, figure=fig, left=0.1, right=0.75, top=0.9, bottom=0.15)
    ax = fig.add_subplot(gs[0])
    ax.set_position([0.1, 0.15, plot_width / total_width, plot_height / total_height])

    stats_dfs = []
    for metric in metrics:
        mean_df = df.groupby(["Split", "Model type", "Calibration"])[metric].mean().rename(f"{metric}_mean")
        std_df = df.groupby(["Split", "Model type", "Calibration"])[metric].std().rename(f"{metric}_std")
        stats_df = pd.merge(mean_df, std_df, left_index=True, right_index=True).reset_index()
        stats_df["Group"] = stats_df.apply(
            lambda row: f"{row['Split']}_{row['Model type']}_{row['Calibration']}", axis=1
        )
        stats_df["Metric"] = metric
        stats_dfs.append(stats_df)

    combined_stats_df = pd.concat(stats_dfs)
    if models_order is None:
        models_order = combined_stats_df["Model type"].unique().tolist()

    if color_dict is None:
        scalar_mappable = ScalarMappable(cmap=cmap)
        color_dict = {
            m: c
            for m, c in zip(
                models_order,
                scalar_mappable.to_rgba(range(len(models_order)), alpha=1).tolist(),
            )
        }
    color_dict = {k: color_dict[k] for k in models_order}

    hatches_dict = {
        "Before Calibration": "\\\\",
        "After Calibration": "",
    }

    bar_width = 0.1
    group_spacing = 0.2
    split_spacing = 0.6
    num_bars = len(models_order) * 2
    positions = []
    tick_positions = []
    tick_labels = []

    for i, metric in enumerate(metrics):
        metric_data = combined_stats_df[combined_stats_df["Metric"] == metric]
        split_types = metric_data["Split"].unique()
        for j, split in enumerate(split_types):
            split_data = metric_data[metric_data["Split"] == split]
            split_data = split_data[split_data["Model type"].isin(models_order)]

            for k, model_type in enumerate(models_order):
                for l, calibration in enumerate(["Before Calibration", "After Calibration"]):
                    position = (
                        i * (split_spacing + len(split_types) * (num_bars * bar_width + group_spacing))
                        + j * (num_bars * bar_width + group_spacing)
                        + k * 2 * bar_width
                        + l * bar_width
                    )
                    positions.append(position)
                    height = split_data[
                        (split_data["Model type"] == model_type)
                        & (split_data["Calibration"] == calibration)
                    ][f"{metric}_mean"].values[0]
                    ax.bar(
                        position,
                        height=height,
                        color=color_dict[model_type],
                        hatch=hatches_dict[calibration],
                        width=bar_width,
                    )

            center_position = (
                i * (split_spacing + len(split_types) * (num_bars * bar_width + group_spacing))
                + j * (num_bars * bar_width + group_spacing)
                + (num_bars * bar_width) / 2
            )
            tick_positions.append(center_position)
            tick_labels.append(f"{metric}\n{split}")

    if show_legend:
        legend_elements = [
            mpatches.Patch(facecolor=color_dict[model], edgecolor="black", label=model)
            for model in models_order
        ]
        legend_elements += [
            mpatches.Patch(facecolor="white", edgecolor="black", hatch=h, label=label)
            for label, h in hatches_dict.items()
        ]
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0, frameon=False)

    for (_, row), bar in zip(combined_stats_df.iterrows(), ax.patches):
        x_bar = bar.get_x() + bar.get_width() / 2
        y_bar = bar.get_height()
        yerr_lower = y_bar - max(0, y_bar - row[f"{row['Metric']}_std"])
        yerr_upper = row[f"{row['Metric']}_std"]
        ax.errorbar(
            x_bar,
            y_bar,
            yerr=[[yerr_lower], [yerr_upper]],
            color="black",
            fmt="none",
            elinewidth=1,
            capsize=3,
            alpha=0.5,
        )

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", rotation_mode="anchor", fontsize=9)
    ax.set_ylim(bottom=0.0)

    if save_dir:
        metrics_names = "_".join(metrics)
        plot_name = f"comparison_barplot_{cmap}_{metrics_names}"
        save_plot(fig, save_dir, plot_name, show_legend=show_legend)

    if INTERACTIVE_MODE:
        plt.show()
    plt.close()


def load_and_aggregate_calibration_data(base_path: str, paths: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load calibration curve data from multiple model paths and aggregate statistics.

    Parameters
    ----------
    base_path : str
        Base directory from which model subpaths are resolved.
    paths : list of str
        Relative paths to model directories containing 'calibration_plot_data.csv'.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray)
        Tuple of (expected_values, mean_observed, lower_bound, upper_bound), each of shape (n_bins,).
    """
    expected_values = []
    observed_values = []
    for path in paths:
        file_path = os.path.join(base_path, path, "calibration_plot_data.csv")
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)
            expected_values = data["Expected Proportion"]
            observed_values.append(data["Observed Proportion"])
        else:
            print(f"File not found: {file_path}")

    expected_values = np.array(expected_values)
    observed_values = np.array(observed_values)
    mean_observed = np.mean(observed_values, axis=0)
    lower_bound = np.min(observed_values, axis=0)
    upper_bound = np.max(observed_values, axis=0)
    return expected_values, mean_observed, lower_bound, upper_bound


def plot_calibration_data(
    df_aggregated: pd.DataFrame,
    base_path: str,
    save_dir: Optional[str] = None,
    title: str = "Calibration Plot",
    color_name: str = "tab10_r",
    color_dict: Optional[Dict[str, str]] = None,
    group_order: Optional[List[str]] = None,
    fig_width: Optional[float] = None,
    fig_height: Optional[float] = None,
    show_legend: bool = False,
) -> None:
    """
    Plot aggregated calibration curves for multiple groups against the perfect calibration line.

    Parameters
    ----------
    df_aggregated : pd.DataFrame
        Aggregated DataFrame containing 'Group' and 'project_model' lists for each group.
    base_path : str
        Base directory where model paths are located.
    save_dir : str or None, optional
        Directory to save plot images. Default is None.
    title : str, optional
        Plot title. Default is "Calibration Plot".
    color_name : str, optional
        Colormap name used to derive distinct colors per group. Default is "tab10_r".
    color_dict : dict[str, str] or None, optional
        Precomputed color mapping from group to color. If None, one is generated.
    group_order : list of str or None, optional
        Order of groups in the legend. Default derives from data.
    fig_width : float or None, optional
        Width of the plot area. Default is 6.
    fig_height : float or None, optional
        Height of the plot area. Default is 6.
    show_legend : bool, optional
        If True, include a legend. Default is False.

    Returns
    -------
    None
    """
    plot_width = fig_width if fig_width else 6
    plot_height = fig_height if fig_height else 6
    total_width = plot_width + 4
    total_height = plot_height + 2

    fig = plt.figure(figsize=(total_width, total_height))
    gs = gridspec.GridSpec(1, 1, figure=fig, left=0.15, right=0.75, top=0.9, bottom=0.15)
    ax = fig.add_subplot(gs[0])
    ax.set_position([0.15, 0.15, plot_width / total_width, plot_height / total_height])

    if group_order is None:
        group_order = list(df_aggregated["Group"].unique())

    if color_dict is None:
        scalar_mappable = ScalarMappable(cmap=color_name)
        colors = scalar_mappable.to_rgba(range(len(group_order)))
        color_dict = {group: color for group, color in zip(group_order, colors)}

    legend_handles = {}
    for idx, row in df_aggregated.iterrows():
        model_paths = row["project_model"]
        group_label = row["Group"]
        color = color_dict[group_label]
        expected, mean_observed, lower_bound, upper_bound = load_and_aggregate_calibration_data(base_path, model_paths)
        (line,) = ax.plot(expected, mean_observed, label=group_label, color=color)
        ax.fill_between(expected, lower_bound, upper_bound, alpha=0.2, color=color)
        if group_label not in legend_handles:
            legend_handles[group_label] = line

    (perfect_line,) = ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    legend_handles["Perfect Calibration"] = perfect_line

    ordered_legend_handles = [legend_handles[group] for group in group_order if group in legend_handles]
    ordered_legend_handles.append(legend_handles["Perfect Calibration"])
    if show_legend:
        ax.legend(handles=ordered_legend_handles, bbox_to_anchor=(1.05, 1), loc="upper left")

    ax.set_title(title)
    ax.set_xlabel("Expected Proportion")
    ax.set_ylabel("Observed Proportion")
    ax.grid(True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    if save_dir:
        plot_name = f"{title.replace(' ', '_')}"
        save_plot(fig, save_dir, plot_name, tighten=True, show_legend=show_legend)
    if INTERACTIVE_MODE:
        plt.show()
    plt.close()


def move_model_folders(
    df: pd.DataFrame,
    search_dirs: List[str],
    output_dir: str,
    overwrite: bool = False,
) -> None:
    """
    Move or merge model directories into a single output folder based on model names.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a 'model name' column.
    search_dirs : list of str
        Directories to search for model subfolders.
    output_dir : str
        Destination directory where model folders will be moved or merged.
    overwrite : bool, optional
        If True, existing folders are merged (copied) with source. Default is False.

    Returns
    -------
    None
    """
    model_names = df["model name"].unique()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory '{output_dir}'.")

    for model_name in model_names:
        found = False
        for search_dir in search_dirs:
            if not os.path.isdir(search_dir):
                print(f"Search directory '{search_dir}' does not exist. Skipping.")
                continue
            subdirs = [d for d in os.listdir(search_dir) if os.path.isdir(os.path.join(search_dir, d))]
            if model_name in subdirs:
                source_dir = os.path.join(search_dir, model_name)
                dest_dir = os.path.join(output_dir, model_name)
                if os.path.exists(dest_dir):
                    if overwrite:
                        shutil.copytree(source_dir, dest_dir, dirs_exist_ok=True)
                        print(f"Merged (Copied) '{source_dir}' to '{dest_dir}'.")
                else:
                    try:
                        shutil.move(source_dir, dest_dir)
                        print(f"Moved '{source_dir}' to '{dest_dir}'.")
                    except Exception as e:
                        print(f"Error moving '{source_dir}' to '{dest_dir}': {e}")
                found = True
                break
        if not found:
            print(f"Model folder '{model_name}' not found in any of the search directories.")


def load_predictions(model_path: str) -> pd.DataFrame:
    """
    Load pickled predictions from a model directory.

    Parameters
    ----------
    model_path : str
        Path to the model directory containing 'preds.pkl'.

    Returns
    -------
    pd.DataFrame
        DataFrame loaded from the pickle file.
    """
    preds_path = os.path.join(model_path, "preds.pkl")
    return pd.read_pickle(preds_path)


def calculate_rmse_rejection_curve(
    preds: pd.DataFrame,
    uncertainty_col: str = "y_alea",
    true_label_col: str = "y_true",
    pred_label_col: str = "y_pred",
    normalize_rmse: bool = False,
    random_rejection: bool = False,
    unc_type: Optional[str] = None,
    max_rejection_ratio: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute RMSE vs. rejection rate curve and its AUC by rejecting high-uncertainty predictions.

    Parameters
    ----------
    preds : pd.DataFrame
        DataFrame with columns for true labels, predicted labels, and uncertainty components.
    uncertainty_col : str, optional
        Column name for uncertainty to sort by if `unc_type` is None. Default is "y_alea".
    true_label_col : str, optional
        Column name for true labels. Default is "y_true".
    pred_label_col : str, optional
        Column name for predicted labels. Default is "y_pred".
    normalize_rmse : bool, optional
        If True, normalize RMSE by the initial RMSE before rejection. Default is False.
    random_rejection : bool, optional
        If True, randomly reject samples instead of sorting by uncertainty. Default is False.
    unc_type : {"aleatoric", "epistemic", "both"} or None, optional
        Which uncertainty to use. If "both", sums aleatoric and epistemic. If None, use `uncertainty_col`.
    max_rejection_ratio : float, optional
        Maximum fraction of samples to reject (exclusive of the tail). Default is 0.95.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray, float)
        Tuple of (rejection_rates, rmses, AUC of the RMSE–rejection curve).

    Raises
    ------
    ValueError
        If `unc_type` is invalid or `uncertainty_col` is not present when needed.
    """
    if unc_type == "aleatoric":
        uncertainty_col = "y_alea"
    elif unc_type == "epistemic":
        uncertainty_col = "y_eps"
    elif unc_type == "both":
        preds["y_unc"] = preds["y_alea"] + preds["y_eps"]
        uncertainty_col = "y_unc"
    elif unc_type is None and uncertainty_col in preds.columns:
        pass
    else:
        raise ValueError(
            "Either provide valid uncertainty type or provide the uncertainty column name in the DataFrame"
        )

    if random_rejection:
        preds = preds.sample(frac=max_rejection_ratio).reset_index(drop=True)
    else:
        preds = preds.sort_values(by=uncertainty_col, ascending=False)

    max_rejection_index = int(len(preds) * max_rejection_ratio)
    step = max(1, int(len(preds) * 0.01))
    rejection_steps = np.arange(0, max_rejection_index, step=step)
    rejection_rates = rejection_steps / len(preds)
    rmses = []

    initial_rmse = mean_squared_error(preds[true_label_col], preds[pred_label_col], squared=False)

    for i in rejection_steps:
        selected_preds = preds.iloc[i:]
        rmse = mean_squared_error(selected_preds[true_label_col], selected_preds[pred_label_col], squared=False)
        if normalize_rmse:
            rmse /= initial_rmse
        rmses.append(rmse)
    auc_arc = auc(rejection_rates, rmses)
    return rejection_rates, np.array(rmses), float(auc_arc)


def calculate_rejection_curve(
    df: pd.DataFrame,
    model_paths: List[str],
    unc_col: str,
    random_rejection: bool = False,
    normalize_rmse: bool = False,
    max_rejection_ratio: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Aggregate RMSE–rejection curves across models and compute mean/std and AUC statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Auxiliary DataFrame (not used directly, kept for API symmetry).
    model_paths : list of str
        Paths to model directories containing 'preds.pkl'.
    unc_col : str
        Uncertainty column name to use when computing curves (e.g., 'y_alea' or 'y_eps').
    random_rejection : bool, optional
        If True, randomly reject samples. Default is False.
    normalize_rmse : bool, optional
        If True, normalize RMSE by the initial RMSE. Default is False.
    max_rejection_ratio : float, optional
        Maximum fraction of samples to reject. Default is 0.95.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray, numpy.ndarray, float, float)
        Tuple of (rejection_rates, mean_rmses, std_rmses, mean_auc, std_auc).
    """
    aggregated_rmses = []
    auc_values = []
    rejection_rates = None

    for model_path in model_paths:
        preds = load_predictions(model_path)
        if preds.empty:
            print(f"Preds not loaded for model: {model_path}")
            continue
        rejection_rates, rmses, auc_arc = calculate_rmse_rejection_curve(
            preds,
            uncertainty_col=unc_col,
            random_rejection=random_rejection,
            normalize_rmse=normalize_rmse,
            max_rejection_ratio=max_rejection_ratio,
        )
        aggregated_rmses.append(rmses)
        auc_values.append(auc_arc)

    mean_rmses = np.mean(aggregated_rmses, axis=0)
    std_rmses = np.std(aggregated_rmses, axis=0)
    mean_auc = np.mean(auc_values)
    std_auc = np.std(auc_values)
    return rejection_rates, mean_rmses, std_rmses, float(mean_auc), float(std_auc)


def get_handles_labels(ax: plt.Axes, group_order: List[str]) -> Tuple[List, List[str]]:
    """
    Extract legend handles/labels ordered by group prefix.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object from which to retrieve legend entries.
    group_order : list of str
        Group prefixes to order legend entries by.

    Returns
    -------
    (list, list of str)
        Ordered handles and labels.
    """
    handles, labels = ax.get_legend_handles_labels()
    ordered_handles = []
    ordered_labels = []
    for group in group_order:
        for label, handle in zip(labels, handles):
            if label.startswith(group):
                ordered_handles.append(handle)
                ordered_labels.append(label)
    return ordered_handles, ordered_labels


def plot_rmse_rejection_curves(
    df: pd.DataFrame,
    base_dir: str,
    cmap: str = "tab10_r",
    color_dict: Optional[Dict[str, str]] = None,
    save_dir_plot: Optional[str] = None,
    add_to_title: str = "",
    normalize_rmse: bool = False,
    unc_type: str = "aleatoric",
    max_rejection_ratio: float = 0.95,
    group_order: Optional[List[str]] = None,
    fig_width: Optional[float] = None,
    fig_height: Optional[float] = None,
    show_legend: bool = False,
) -> pd.DataFrame:
    """
    Plot RMSE–rejection curves per group, including random rejection baselines, and summarize AUCs.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing columns 'Group', 'Split', and 'project_model'.
    base_dir : str
        Base directory where model paths are located.
    cmap : str, optional
        Colormap name used to derive distinct colors per group. Default is "tab10_r".
    color_dict : dict[str, str] or None, optional
        Precomputed color mapping from group to color. If None, one is generated.
    save_dir_plot : str or None, optional
        Directory to save the plot images. Default is None.
    add_to_title : str, optional
        Suffix for the plot filename and title. Default is empty string.
    normalize_rmse : bool, optional
        If True, normalize RMSE by initial RMSE. Default is False.
    unc_type : {"aleatoric", "epistemic", "both"}, optional
        Uncertainty component to use for rejection. Default is "aleatoric".
    max_rejection_ratio : float, optional
        Maximum fraction of samples to reject. Default is 0.95.
    group_order : list of str or None, optional
        Order of groups in the legend. Default derives from data.
    fig_width : float or None, optional
        Plot width. Default is 6.
    fig_height : float or None, optional
        Plot height. Default is 6.
    show_legend : bool, optional
        If True, include a legend. Default is False.

    Returns
    -------
    pd.DataFrame
        Summary DataFrame with columns ['Model type', 'Split', 'Group', 'AUC-RRC_mean', 'AUC-RRC_std'].
    """
    assert unc_type in ["aleatoric", "epistemic", "both"], "Invalid unc_type"
    unc_col = "y_alea" if unc_type == "aleatoric" else "y_eps"

    plot_width = fig_width if fig_width else 6
    plot_height = fig_height if fig_height else 6
    total_width = plot_width + 4
    total_height = plot_height + 2

    fig = plt.figure(figsize=(total_width, total_height))
    gs = gridspec.GridSpec(1, 1, figure=fig, left=0.15, right=0.75, top=0.9, bottom=0.15)
    ax = fig.add_subplot(gs[0])
    ax.set_position([0.15, 0.15, plot_width / total_width, plot_height / total_height])

    if group_order is None:
        group_order = list(df["Group"].unique())

    if color_dict is None:
        scalar_mappable = ScalarMappable(cmap=cmap)
        colors = scalar_mappable.to_rgba(range(len(group_order)))
        color_dict = {group: color for group, color in zip(group_order, colors)}

    color_dict["random reject"] = "black"

    df = df.copy()
    df.loc[:, "model_path"] = df["project_model"].apply(
        lambda x: (str(os.path.join(base_dir, x)) if not str(x).startswith(base_dir) else x)
    )

    stats_dfs = []
    included_groups = df["Group"].unique()
    legend_handles = []

    for group in included_groups:
        group_data = df[df["Group"] == group]
        model_paths = group_data["model_path"].unique()
        rejection_rates, mean_rmses, std_rmses, mean_auc, std_auc = calculate_rejection_curve(
            df, model_paths, unc_col, normalize_rmse=normalize_rmse, max_rejection_ratio=max_rejection_ratio
        )
        (line,) = ax.plot(
            rejection_rates,
            mean_rmses,
            label=f"{group} (AUC-RRC: {mean_auc:.3f} ± {std_auc:.3f})",
            color=color_dict[group],
        )
        ax.fill_between(rejection_rates, mean_rmses - std_rmses, mean_rmses + std_rmses, color=color_dict[group], alpha=0.2)
        legend_handles.append(line)
        stats_dfs.append({
            "Model type": group.rsplit("_", 1)[1],
            "Split": group.rsplit("_", 1)[0],
            "Group": group,
            "AUC-RRC_mean": mean_auc,
            "AUC-RRC_std": std_auc,
        })

    for split in df["Split"].unique():
        split_data = df[df["Split"] == split]
        model_paths = split_data["model_path"].unique()
        rejection_rates, mean_rmses, std_rmses, mean_auc, std_auc = calculate_rejection_curve(
            df, model_paths, unc_col, random_rejection=True, normalize_rmse=normalize_rmse, max_rejection_ratio=max_rejection_ratio
        )
        (line,) = ax.plot(
            rejection_rates,
            mean_rmses,
            label=f"random reject - {split} (AUC-RRC: {mean_auc:.3f} ± {std_auc:.3f})",
            color="black",
            linestyle="--",
        )
        ax.fill_between(rejection_rates, mean_rmses - std_rmses, mean_rmses + std_rmses, color="grey", alpha=0.2)
        legend_handles.append(line)
        stats_dfs.append({
            "Model type": "random reject",
            "Split": split,
            "Group": f"random reject - {split}",
            "AUC-RRC_mean": mean_auc,
            "AUC-RRC_std": std_auc,
        })

    ax.set_xlabel("Rejection Rate")
    ax.set_ylabel("RMSE" if not normalize_rmse else "Normalized RMSE")
    ax.set_xlim(0, max_rejection_ratio)
    ax.grid(True)

    if show_legend:
        ordered_handles, ordered_labels = get_handles_labels(ax, group_order)
        ordered_handles += [legend_handles[-1]]
        ordered_labels += [legend_handles[-1].get_label()]
        ax.legend(handles=ordered_handles, loc="lower left")

    plot_name = f"rmse_rejection_curve_{add_to_title}" if add_to_title else "rmse_rejection_curve"
    save_plot(fig, save_dir_plot, plot_name, tighten=True, show_legend=show_legend)
    if INTERACTIVE_MODE:
        plt.show()
    plt.close()

    return pd.DataFrame(stats_dfs)


def plot_auc_comparison(
    stats_df: pd.DataFrame,
    cmap: str = "tab10_r",
    color_dict: Optional[Dict[str, str]] = None,
    save_dir: Optional[str] = None,
    add_to_title: str = "",
    min_y_axis: float = 0.0,
    hatches_dict: Optional[Dict[str, str]] = None,
    group_order: Optional[List[str]] = None,
    fig_width: Optional[float] = None,
    fig_height: Optional[float] = None,
    show_legend: bool = False,
) -> None:
    """
    Plot bar charts comparing RRC-AUC across splits and model types, including random reject baselines.

    Parameters
    ----------
    stats_df : pd.DataFrame
        Summary DataFrame with columns ['Group', 'Split', 'Model type', 'AUC-RRC_mean', 'AUC-RRC_std'].
    cmap : str, optional
        Colormap name used to derive distinct colors per model type. Default is "tab10_r".
    color_dict : dict[str, str] or None, optional
        Precomputed color mapping from model type to color. If None, one is generated.
    save_dir : str or None, optional
        Directory to save plot images. Default is None.
    add_to_title : str, optional
        Title suffix for the plot. Default is empty string.
    min_y_axis : float, optional
        Minimum y-axis limit. Default is 0.0.
    hatches_dict : dict[str, str] or None, optional
        Hatch mapping for splits (e.g., {"stratified": "\\\\"}). Default uses sensible defaults.
    group_order : list of str or None, optional
        Order of groups in the legend and x-axis. Default derives from data.
    fig_width : float or None, optional
        Plot width. Default is 6.
    fig_height : float or None, optional
        Plot height. Default is 6.
    show_legend : bool, optional
        If True, include a legend. Default is False.

    Returns
    -------
    None
    """
    if hatches_dict is None:
        hatches_dict = {"stratified": "\\\\", "scaffold_cluster": "", "time": "/\\/\\/"}

    if group_order:
        all_groups = group_order + list(stats_df.loc[stats_df["Group"].str.startswith("random reject"), "Group"].unique())
        stats_df["Group"] = pd.Categorical(stats_df["Group"], categories=all_groups, ordered=True)
    else:
        all_groups = stats_df["Group"].unique().tolist()

    stats_df = stats_df.sort_values("Group").reset_index(drop=True)

    splits = list(hatches_dict.keys())
    stats_df.loc[:, "Split"] = pd.Categorical(stats_df["Split"], categories=splits, ordered=True)
    stats_df = stats_df.sort_values("Split").reset_index(drop=True)

    unique_model_types = stats_df.loc[stats_df["Model type"] != "random reject", "Model type"].unique()

    if color_dict is None:
        scalar_mappable = ScalarMappable(cmap=cmap)
        colors = scalar_mappable.to_rgba(range(len(unique_model_types)))
        color_dict = {model: color for model, color in zip(unique_model_types, colors)}
    color_dict["random reject"] = "black"

    unique_model_types = np.append(unique_model_types, "random reject")

    bar_width = 0.12
    group_spacing = 0.6

    plot_width = fig_width if fig_width else 6
    plot_height = fig_height if fig_height else 6
    total_width = plot_width + 4
    total_height = plot_height + 4

    fig = plt.figure(figsize=(total_width, total_height))
    gs = gridspec.GridSpec(1, 1, figure=fig, left=0.15, right=0.75, top=0.9, bottom=0.15)
    ax = fig.add_subplot(gs[0])
    ax.set_position([0.15, 0.15, plot_width / total_width, plot_height / total_height])

    tick_positions = []
    tick_labels = []

    for i, split in enumerate(splits):
        split_data = stats_df[stats_df["Split"] == split]
        split_data.loc[:, "Group"] = pd.Categorical(split_data["Group"], categories=all_groups, ordered=True)
        for j, (_, row) in enumerate(split_data.iterrows()):
            position = i * (len(unique_model_types) * bar_width + group_spacing) + j * bar_width
            ax.bar(
                position,
                height=row["AUC-RRC_mean"],
                yerr=row["AUC-RRC_std"],
                color=color_dict[row["Model type"]],
                edgecolor="white" if row["Model type"] == "random reject" else "black",
                hatch=hatches_dict[row["Split"]],
                width=bar_width,
            )
        center_position = i * (len(unique_model_types) * bar_width + group_spacing) + (len(unique_model_types) * bar_width) / 2
        tick_positions.append(center_position)
        tick_labels.append(split)

    def create_stats_legend(color_dict: Dict[str, str], hatches_dict: Dict[str, str], splits: List[str], model_types: Union[List[str], np.ndarray]):
        patches = []
        for split in splits:
            for model in model_types:
                label = f"{split} {model}"
                hatch_color = "white" if model == "random reject" else "black"
                patch = mpatches.Patch(facecolor=color_dict[model], hatch=hatches_dict[split], edgecolor=hatch_color, label=label)
                patches.append(patch)
        return patches

    if show_legend:
        legend_elements = create_stats_legend(color_dict, hatches_dict, splits, unique_model_types)
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0, frameon=False)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", rotation_mode="anchor", fontsize=9)
    ax.set_ylabel("RRC-AUC")
    ax.set_ylim(min_y_axis, 1.0)

    plot_name = f"auc_comparison_barplot_{cmap}" + (f"_{add_to_title}" if add_to_title else "")
    save_plot(fig, save_dir, plot_name, tighten=True, show_legend=show_legend)
    if INTERACTIVE_MODE:
        plt.show()
    plt.close()


def save_stats_df(stats_df: pd.DataFrame, save_dir: str, add_to_title: str = "") -> None:
    """
    Save a stats DataFrame to CSV in a given directory.

    Parameters
    ----------
    stats_df : pd.DataFrame
        DataFrame to save.
    save_dir : str
        Target directory to save the CSV.
    add_to_title : str, optional
        Suffix to append to the filename. Default is empty string.

    Returns
    -------
    None
    """
    os.makedirs(save_dir, exist_ok=True)
    stats_df.to_csv(os.path.join(save_dir, f"stats_df_{add_to_title}.csv"), index=False)


def load_stats_df(save_dir: str, add_to_title: str = "") -> pd.DataFrame:
    """
    Load a stats DataFrame from CSV in a given directory.

    Parameters
    ----------
    save_dir : str
        Directory containing the CSV.
    add_to_title : str, optional
        Suffix appended to the filename. Default is empty string.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.
    """
    return pd.read_csv(os.path.join(save_dir, f"stats_df_{add_to_title}.csv"))
