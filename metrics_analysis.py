import pandas as pd

import itertools
import os
import argparse
from typing import List, Dict, Optional, Union
import numpy as np

import shutil

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# from matplotlib import colormaps  # Use the new colormaps API
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec

from sklearn.metrics import mean_squared_error, auc

LEGEND_ON = False

# DESCRIPTORS
descriptor_protein = "ankh-large"
descriptor_chemical = "ecfp2048"
prot_input_dim = 1536
chem_input_dim = 2048

group_cols = [
    "Model type",
    "Task",
    "Activity",
    "Split",
    "desc_prot",
    "desc_chem",
    "dropout",
]

numeric_cols = [
    "RMSE",
    "R2",
    "MAE",
    "MDAE",
    "MARPD",
    "PCC",
    "RMS Calibration",
    "MA Calibration",
    "Miscalibration Area",
    "Sharpness",
    "NLL",
    "CRPS",
    "Check",
    "Interval",
    "rho_rank",
    "rho_rank_sim",
    "rho_rank_sim_std",
    "uq_mis_cal",
    "uq_NLL",
    "uq_NLL_sim",
    "uq_NLL_sim_std",
    "Z_var",
    "Z_var_CI_low",
    "Z_var_CI_high",
    "Z_mean",
    "Z_mean_CI_low",
    "Z_mean_CI_high",
    "rmv_rmse_slope",
    "rmv_rmse_r_sq",
    "rmv_rmse_intercept",
    "aleatoric_uct_mean",
    "epistemic_uct_mean",
    "total_uct_mean",
]

string_cols = ["wandb project", "wandb run", "model name"]

order_by = ["Split", "Model type"]

group_order = [
    "stratified_ensemble",
    "stratified_eoe",
    "stratified_evidential",
    "stratified_emc",
    "stratified_mcdropout",
    "scaffold_cluster_ensemble",
    "scaffold_cluster_eoe",
    "scaffold_cluster_evidential",
    "scaffold_cluster_emc",
    "scaffold_cluster_mcdropout",
    "time_ensemble",
    "time_eoe",
    "time_evidential",
    "time_emc",
    "time_mcdropout",
]

group_order_no_time = [
    "stratified_ensemble",
    "stratified_eoe",
    "stratified_evidential",
    "stratified_emc",
    "stratified_mcdropout",
    "scaffold_cluster_ensemble",
    "scaffold_cluster_eoe",
    "scaffold_cluster_evidential",
    "scaffold_cluster_emc",
    "scaffold_cluster_mcdropout",
]

hatches_dict = {
    "stratified": "\\\\",
    "scaffold_cluster": "",
    "time": "///",
}

hatches_dict_no_time = {
    "stratified": "\\\\",
    "scaffold_cluster": "",
}

accmetrics = ["RMSE", "R2", "MAE", "MDAE", "MARPD", "PCC"]

accmetrics2 = ["RMSE", "R2", "PCC"]

uctmetrics = [
    "RMS Calibration",
    "MA Calibration",
    "Miscalibration Area",
    "Sharpness",
    "CRPS",
    "Check",
    "NLL",
    "Interval",
]

uctmetrics2 = ["Miscalibration Area", "Sharpness", "CRPS", "NLL", "Interval"]

all_cmaps = [
    "Accent",
    "Accent_r",
    "Blues",
    "Blues_r",
    "BrBG",
    "BrBG_r",
    "BuGn",
    "BuGn_r",
    "BuPu",
    "BuPu_r",
    "CMRmap",
    "CMRmap_r",
    "Dark2",
    "Dark2_r",
    "GnBu",
    "GnBu_r",
    "Grays",
    "Greens",
    "Greens_r",
    "Greys",
    "Greys_r",
    "OrRd",
    "OrRd_r",
    "Oranges",
    "Oranges_r",
    "PRGn",
    "PRGn_r",
    "Paired",
    "Paired_r",
    "Pastel1",
    "Pastel1_r",
    "Pastel2",
    "Pastel2_r",
    "PiYG",
    "PiYG_r",
    "PuBu",
    "PuBuGn",
    "PuBuGn_r",
    "PuBu_r",
    "PuOr",
    "PuOr_r",
    "PuRd",
    "PuRd_r",
    "Purples",
    "Purples_r",
    "RdBu",
    "RdBu_r",
    "RdGy",
    "RdGy_r",
    "RdPu",
    "RdPu_r",
    "RdYlBu",
    "RdYlBu_r",
    "RdYlGn",
    "RdYlGn_r",
    "Reds",
    "Reds_r",
    "Set1",
    "Set1_r",
    "Set2",
    "Set2_r",
    "Set3",
    "Set3_r",
    "Spectral",
    "Spectral_r",
    "Wistia",
    "Wistia_r",
    "YlGn",
    "YlGnBu",
    "YlGnBu_r",
    "YlGn_r",
    "YlOrBr",
    "YlOrBr_r",
    "YlOrRd",
    "YlOrRd_r",
    "afmhot",
    "afmhot_r",
    "autumn",
    "autumn_r",
    "binary",
    "binary_r",
    "bone",
    "bone_r",
    "brg",
    "brg_r",
    "bwr",
    "bwr_r",
    "cividis",
    "cividis_r",
    "cool",
    "cool_r",
    "coolwarm",
    "coolwarm_r",
    "copper",
    "copper_r",
    "crest",
    "tab10_r",
    "cubehelix",
    "cubehelix_r",
    "flag",
    "flag_r",
    "flare",
    "flare_r",
    "gist_earth",
    "gist_earth_r",
    "gist_gray",
    "gist_gray_r",
    "gist_grey",
    "gist_heat",
    "gist_heat_r",
    "gist_ncar",
    "gist_ncar_r",
    "gist_rainbow",
    "gist_rainbow_r",
    "gist_stern",
    "gist_stern_r",
    "gist_yarg",
    "gist_yarg_r",
    "gist_yerg",
    "gnuplot",
    "gnuplot2",
    "gnuplot2_r",
    "gnuplot_r",
    "gray",
    "gray_r",
    "grey",
    "hot",
    "hot_r",
    "hsv",
    "hsv_r",
    "icefire",
    "icefire_r",
    "inferno",
    "inferno_r",
    "jet",
    "jet_r",
    "magma",
    "magma_r",
    "mako",
    "mako_r",
    "nipy_spectral",
    "nipy_spectral_r",
    "ocean",
    "ocean_r",
    "pink",
    "pink_r",
    "plasma",
    "plasma_r",
    "prism",
    "prism_r",
    "rainbow",
    "rainbow_r",
    "rocket",
    "rocket_r",
    "seismic",
    "seismic_r",
    "spring",
    "spring_r",
    "summer",
    "summer_r",
    "tab10",
    "tab10_r",
    "tab20",
    "tab20_r",
    "tab20b",
    "tab20b_r",
    "tab20c",
    "tab20c_r",
    "terrain",
    "terrain_r",
    "turbo",
    "turbo_r",
    "twilight",
    "twilight_r",
    "twilight_shifted",
    "twilight_shifted_r",
    "viridis",
    "viridis_r",
    "vlag",
    "vlag_r",
    "winter",
    "winter_r",
]
subset_cmaps = [
    "Accent",
    "Paired",
    "Set1",
    "Set1_r",
    "tab10",
    "tab10_r",
    "tab20b",
    "tab20b_r",
    "turbo",
    "turbo_r",
]


# RESULTS AGGREGATION TO CREATE THE FINAL RESULTS TABLE
def aggregate_results_csv(
    df, group_cols, numeric_cols, string_cols, order_by=None, output_file_path=None
):
    # Group the DataFrame by the specified columns
    grouped = df.groupby(group_cols)
    # Aggregate the numeric columns
    aggregated = grouped[numeric_cols].agg(["mean", "std"])
    # Combine mean and std into the required format
    for col in numeric_cols:
        aggregated[(col, "combined")] = (
            aggregated[(col, "mean")].round(3).astype(str)
            + "("
            + aggregated[(col, "std")].round(3).astype(str)
            + ")"
        )
    # Drop the separate mean and std columns, keeping only the combined column
    aggregated = aggregated[[col for col in aggregated.columns if col[1] == "combined"]]

    # Rename the columns to a simpler format
    aggregated.columns = [col[0] for col in aggregated.columns]

    # Step 6: Aggregate the string columns into lists
    string_aggregated = grouped[string_cols].agg(lambda x: list(x))

    # Step 7: Create the new column combining wandb project and model name
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

    # Combine the numeric and string aggregations
    final_aggregated = pd.concat(
        [aggregated, string_aggregated, project_model_aggregated], axis=1
    ).reset_index()

    if order_by:
        final_aggregated = final_aggregated.sort_values(by=order_by)

    if output_file_path:
        final_aggregated.to_csv(output_file_path, index=False)

    return final_aggregated


# # Group the DataFrame by the specified columns
# grouped = df.groupby(group_cols)
# # Aggregate the numeric columns
# aggregated = grouped[numeric_cols].agg(['mean', 'std'])
# # Combine mean and std into the required format
# for col in numeric_cols:
#     aggregated[(col, 'combined')] = aggregated[(col, 'mean')].round(3).astype(str) + ' (' + aggregated[(col, 'std')].round(3).astype(str) + ')'
# # Drop the separate mean and std columns, keeping only the combined column
# aggregated = aggregated[[col for col in aggregated.columns if col[1] == 'combined']]
#
# # Rename the columns to a simpler format
# aggregated.columns = [col[0] for col in aggregated.columns]
#
# # Step 6: Aggregate the string columns into lists
# string_aggregated = grouped[string_cols].agg(lambda x: list(x))
#
# # Combine the numeric and string aggregations
# final_aggregated = pd.concat([aggregated, string_aggregated], axis=1).reset_index()


# HELPER FUNCTIONS FOR PLOTTING #
def save_plot(fig, save_dir, plot_name, tighten=True, show_legend=LEGEND_ON):
    if not show_legend:
        plt.legend().remove()
    if tighten:
        plt.tight_layout()

    if save_dir and tighten:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(
            os.path.join(save_dir, f"{plot_name}.png"), dpi=300, bbox_inches="tight"
        )
        fig.savefig(os.path.join(save_dir, f"{plot_name}.svg"), bbox_inches="tight")
        fig.savefig(
            os.path.join(save_dir, f"{plot_name}.pdf"), dpi=300, bbox_inches="tight"
        )
        fig.savefig(
            os.path.join(save_dir, f"{plot_name}.eps"), dpi=300, bbox_inches="tight"
        )

    elif save_dir and not tighten:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f"{plot_name}.png"), dpi=300)
        fig.savefig(os.path.join(save_dir, f"{plot_name}.svg"))
        fig.savefig(os.path.join(save_dir, f"{plot_name}.pdf"), dpi=300)
        fig.savefig(os.path.join(save_dir, f"{plot_name}.eps"), dpi=300)


# Function to handle inf values
def handle_inf_values(df):
    df = df.replace([float("inf"), -float("inf")], float("nan"))
    return df


# Pair plot for visualizing relationships
def plot_pairplot(
    df, title, metrics, save_dir=None, cmap="viridis", group_order=group_order
):
    df = handle_inf_values(df)
    sns.pairplot(
        df,
        hue="Group",
        hue_order=group_order,
        # markers=['o', 's'],
        vars=metrics,
        palette=cmap,
        plot_kws={"alpha": 0.7},
    )
    plt.suptitle(title, y=1.02)
    plot_name = f"pairplot_{title.replace(' ', '_')}"
    save_plot(plt.gcf(), save_dir, plot_name, tighten=False)
    plt.show()


# Function to plot line metrics
def plot_line_metrics(df, title, metrics, save_dir=None, group_order=group_order):
    df = handle_inf_values(df)
    plt.figure(figsize=(14, 7))
    for metric in metrics:
        sns.lineplot(
            data=df,
            x="wandb run",
            y=metric,
            hue="Group",
            marker="o",
            palette="Set2",
            hue_order=group_order,
        )
        plt.title(f"{title} - {metric}")
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.show()

        plot_name = f"line_{title.replace(' ', '_')}_{metric}"
        save_plot(plt.gcf(), save_dir, plot_name)


# Function to plot histograms for metrics
def plot_histogram_metrics(
    df, title, metrics, save_dir=None, group_order=group_order, cmap="crest"
):
    df = handle_inf_values(df)
    plt.figure(figsize=(14, 7))
    for metric in metrics:
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
        plt.show()

        plot_name = f"histogram_{title.replace(' ', '_')}_{metric}"
        save_plot(plt.gcf(), save_dir, plot_name)


# Function to plot pairwise scatter plots for metrics
def plot_pairwise_scatter_metrics(
    df, title, metrics, save_dir=None, group_order=group_order, cmap="tab10_r"
):
    df = handle_inf_values(df)
    num_metrics = len(metrics)
    fig, axes = plt.subplots(num_metrics, num_metrics, figsize=(15, 15))

    for i, j in itertools.product(range(num_metrics), range(num_metrics)):
        if i != j:  # Only plot the lower triangle
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
            axes[i, j].set_visible(
                False
            )  # Hide the diagonal and upper triangle subplots

        if j == 0 and i > 0:
            axes[i, j].set_ylabel(metrics[i])
        else:
            axes[i, j].set_ylabel("")

        if i == num_metrics - 1:
            axes[i, j].set_xlabel(metrics[j])
        else:
            axes[i, j].set_xlabel("")

    # Add a single legend
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.15, 1))
    fig.suptitle(title, y=1.02)
    fig.subplots_adjust(top=0.95, wspace=0.4, hspace=0.4)
    plot_name = f"pairwise_scatter_{title.replace(' ', '_')}"
    save_plot(fig, save_dir, plot_name)
    plt.show()


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
) -> Dict[str, str]:
    """
    Plots bar charts for multiple metrics, ensuring that the plot box (axes area)
    has fixed dimensions while the overall figure adjusts to fit additional elements
    like legends, labels, and titles.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the metrics data.
    metrics : List[str]
        List of metric names to be plotted.
    cmap : str, optional
        Colormap for bars, by default "tab10_r".
    save_dir : Optional[str], optional
        Directory to save the plot, by default None.
    hatches_dict : Optional[Dict[str, str]], optional
        Dictionary mapping split types to hatching patterns, by default None.
    group_order : Optional[List[str]], optional
        Order of the groups in the plot, by default None.
    show : bool, optional
        Whether to display the plot, by default True.
    fig_width : Optional[float], optional
        Width of the **inner plot box** (not the full figure), by default None.
    fig_height : Optional[float], optional
        Height of the **inner plot box** (not the full figure), by default None.

    Returns
    -------
    Dict[str, str]
        Dictionary mapping model types to colors.
    """

    # Default plot area size if not provided
    plot_width = fig_width if fig_width else max(10, len(metrics) * 2)
    plot_height = fig_height if fig_height else 6

    # Compute total figure size, leaving space for legends and labels
    total_width = plot_width + 5  # Extra space for legend
    total_height = plot_height + 2  # Extra space for x-axis labels

    fig = plt.figure(figsize=(total_width, total_height))
    gs = gridspec.GridSpec(
        1, 1, figure=fig, left=0.1, right=0.75, top=0.9, bottom=0.15
    )  # Main plot area

    ax = fig.add_subplot(gs[0])  # Main plot area
    ax.set_position([0.1, 0.15, plot_width / total_width, plot_height / total_height])

    stats_dfs = []
    for metric in metrics:
        mean_df = (
            df.groupby(["Split", "Model type"])[metric].mean().rename(f"{metric}_mean")
        )
        std_df = (
            df.groupby(["Split", "Model type"])[metric].std().rename(f"{metric}_std")
        )
        stats_df = pd.merge(
            mean_df, std_df, left_index=True, right_index=True
        ).reset_index()
        stats_df["Group"] = stats_df.apply(
            lambda row: f"{row['Split']}_{row['Model type']}", axis=1
        )
        stats_df["Metric"] = metric
        stats_dfs.append(stats_df)

    combined_stats_df = pd.concat(stats_dfs)

    # Ensure categorical order if provided
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
        # Sort the data according to group_order
        metric_data["Group"] = pd.Categorical(
            metric_data["Group"], categories=group_order, ordered=True
        )
        metric_data = metric_data.sort_values("Group").reset_index(drop=True)
        for j, (_, row) in enumerate(metric_data.iterrows()):
            position = (
                i * (num_bars * bar_width + group_spacing) + (j % num_bars) * bar_width
            )
            positions.append(position)
            ax.bar(
                position,
                height=row[f"{metric}_mean"],
                color=color_dict[row["Model type"]],
                hatch=hatches_dict[row["Split"]],
                width=bar_width,
            )
        center_position = (
            i * (num_bars * bar_width + group_spacing) + (num_bars * bar_width) / 2
        )
        tick_positions.append(center_position)
        if " " in metric:
            metric = metric.replace(" ", "\n")
        tick_labels.append(metric)

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

    # if LEGEND_ON:
    legend_elements = create_stats_legend(
        combined_stats_df, color_dict, hatches_dict, group_order
    )

    ax.legend(
        handles=legend_elements,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0,
        frameon=False,
    )

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
    ax.set_xticklabels(
        tick_labels, rotation=45, ha="right", rotation_mode="anchor", fontsize=9
    )
    # ax.set_xlabel("Metrics")
    # ax.set_ylabel("Values")

    if save_dir:
        metrics_names = "_".join(metrics)
        plot_name = f"barplot_{cmap}_{metrics_names}"
        save_plot(fig, save_dir, plot_name)  #  tighten=True

    if show:
        plt.show()
        plt.close()

    return color_dict


def find_highly_correlated_metrics(
    df, metrics, threshold=0.8, save_dir=None, cmap="coolwarm"
):
    # Calculate the correlation matrix
    corr_matrix = df[metrics].corr().abs()

    # Find pairs of metrics with correlation above the threshold
    highly_correlated_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] > threshold:
                pair = (
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j],
                )
                highly_correlated_pairs.append(pair)

    # Print the highly correlated pairs
    print("Highly correlated metrics (correlation coefficient > {}):".format(threshold))
    for pair in highly_correlated_pairs:
        print(f"{pair[0]} and {pair[1]}: {pair[2]:.2f}")

    # Plot the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap=cmap)
    plt.title("Correlation Matrix")
    metrics_names = "_".join(metrics)
    plot_name = f"correlation_matrix_{threshold}_{metrics_names}"
    save_plot(plt.gcf(), save_dir, plot_name)
    plt.show()

    return highly_correlated_pairs


def plot_comparison_metrics(
    df: pd.DataFrame,
    metrics: List[str],
    cmap: str = "tab10_r",
    save_dir: Optional[str] = None,
    fig_width: Optional[float] = None,
    fig_height: Optional[float] = None,
):
    """
    Plots comparison bar charts for multiple metrics, ensuring that the **plot box** has fixed
    dimensions while the overall figure adapts for legends, labels, and titles.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the metrics data.
    metrics : List[str]
        List of metric names to be plotted.
    cmap : str, optional
        Colormap for bars, by default "tab10_r".
    save_dir : Optional[str], optional
        Directory to save the plot, by default None.
    fig_width : Optional[float], optional
        Width of the **inner plot box** (not the full figure), by default None.
    fig_height : Optional[float], optional
        Height of the **inner plot box** (not the full figure), by default None.
    """

    # Default plot area size if not provided
    plot_width = fig_width if fig_width else max(7, len(metrics) * 3)
    plot_height = fig_height if fig_height else 6

    # Compute total figure size to accommodate labels and legend
    total_width = plot_width + 5  # Extra space for legend
    total_height = plot_height + 2  # Extra space for x-axis labels

    fig = plt.figure(figsize=(total_width, total_height))
    gs = gridspec.GridSpec(
        1, 1, figure=fig, left=0.1, right=0.75, top=0.9, bottom=0.15
    )  # Main plot area

    ax = fig.add_subplot(gs[0])
    ax.set_position([0.1, 0.15, plot_width / total_width, plot_height / total_height])

    stats_dfs = []
    for metric in metrics:
        mean_df = (
            df.groupby(["Split", "Model type", "Calibration"])[metric]
            .mean()
            .rename(f"{metric}_mean")
        )
        std_df = (
            df.groupby(["Split", "Model type", "Calibration"])[metric]
            .std()
            .rename(f"{metric}_std")
        )
        stats_df = pd.merge(
            mean_df, std_df, left_index=True, right_index=True
        ).reset_index()
        stats_df["Group"] = stats_df.apply(
            lambda row: f"{row['Split']}_{row['Model type']}_{row['Calibration']}",
            axis=1,
        )
        stats_df["Metric"] = metric
        stats_dfs.append(stats_df)

    combined_stats_df = pd.concat(stats_dfs)

    scalar_mappable = ScalarMappable(cmap=cmap)
    model_types = combined_stats_df["Model type"].unique()
    color_dict = {
        m: c
        for m, c in zip(
            model_types,
            scalar_mappable.to_rgba(range(len(model_types)), alpha=1).tolist(),
        )
    }

    hatches_dict = {
        "Before Calibration": "\\\\",
        "After Calibration": "",
    }

    bar_width = 0.1
    group_spacing = 0.2  # Adjusted for closer split groups
    split_spacing = 0.6
    num_bars = len(model_types) * 2  # 2 calibration statuses (Before and After)
    positions = []
    tick_positions = []
    tick_labels = []

    for i, metric in enumerate(metrics):
        metric_data = combined_stats_df[combined_stats_df["Metric"] == metric]
        split_types = metric_data["Split"].unique()
        for j, split in enumerate(split_types):
            split_data = metric_data[metric_data["Split"] == split]
            for k, model_type in enumerate(model_types):
                for l, calibration in enumerate(
                    ["Before Calibration", "After Calibration"]
                ):
                    position = (
                        i
                        * (
                            split_spacing
                            + len(split_types) * (num_bars * bar_width + group_spacing)
                        )
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
                i
                * (
                    split_spacing
                    + len(split_types) * (num_bars * bar_width + group_spacing)
                )
                + j * (num_bars * bar_width + group_spacing)
                + (num_bars * bar_width) / 2
            )
            tick_positions.append(center_position)
            tick_labels.append(f"{metric}\n{split}")

    def create_stats_legend(color_dict, hatches_dict):
        patches = []
        for label, color in color_dict.items():
            patches.append(
                mpatches.Patch(facecolor=color, edgecolor="black", label=label)
            )
        for label, hatch in hatches_dict.items():
            patches.append(
                mpatches.Patch(
                    facecolor="white", edgecolor="black", hatch=hatch, label=label
                )
            )
        return patches

    if LEGEND_ON:
        legend_elements = create_stats_legend(color_dict, hatches_dict)

        ax.legend(
            handles=legend_elements,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0,
            frameon=False,
        )

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
    ax.set_xticklabels(
        tick_labels, rotation=45, ha="right", rotation_mode="anchor", fontsize=9
    )

    if save_dir:
        plot_name = f"comparison_barplot_{cmap}_{'_'.join(metrics)}"
        save_plot(fig, save_dir, plot_name, tighten=True)
        # plt.savefig(
        #     f"{save_dir}/comparison_barplot_{cmap}_{'_'.join(metrics)}.png",
        #     bbox_inches="tight",
        #     dpi=300,
        # )

    plt.show()
    plt.close()


# def save_plot(fig, save_dir, plot_name, tighten=True)
def load_and_aggregate_calibration_data(base_path, paths):
    """
    Loads calibration data for multiple paths, computes mean and bounds for observed proportions.
    """
    expected_values = []
    observed_values = []

    for path in paths:
        file_path = os.path.join(base_path, path, "calibration_plot_data.csv")
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)
            expected_values = data["Expected Proportion"]
            # expected_values.append(data['Expected Proportion'])
            observed_values.append(data["Observed Proportion"])
        else:
            print(f"File not found: {file_path}")

    # Convert lists to numpy arrays for aggregation
    expected_values = np.array(expected_values)
    observed_values = np.array(observed_values)

    # Aggregate mean, min, and max for shading
    # mean_expected = np.mean(expected_values,
    # mean_expected = np.mean(expected_values, axis=0)
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
    group_order: Optional[List[str]] = None,
    fig_width: Optional[float] = None,
    fig_height: Optional[float] = None,
):
    """
    Iterates over models in df_aggregated, loads and plots calibration data,
    ensuring the **inner plot box** has fixed width and height.

    Parameters
    ----------
    df_aggregated : pd.DataFrame
        Dataframe containing model paths and group labels.
    base_path : str
        Path to load the calibration data.
    save_dir : Optional[str], optional
        Directory to save the plot, by default None.
    title : str, optional
        Title of the plot, by default "Calibration Plot".
    color_name : str, optional
        Colormap name for the lines, by default "tab10_r".
    group_order : Optional[List[str]], optional
        Order of groups in the legend, by default None.
    fig_width : Optional[float], optional
        Width of the **inner plot box** (not the full figure), by default None.
    fig_height : Optional[float], optional
        Height of the **inner plot box** (not the full figure), by default None.
    """

    # Default plot area size if not provided
    plot_width = fig_width if fig_width else 6
    plot_height = fig_height if fig_height else 6

    # Compute total figure size to accommodate labels and legend
    total_width = plot_width + 4  # Extra space for legend
    total_height = plot_height + 2  # Extra space for x-axis labels

    fig = plt.figure(figsize=(total_width, total_height))
    gs = gridspec.GridSpec(
        1, 1, figure=fig, left=0.15, right=0.75, top=0.9, bottom=0.15
    )  # Main plot area

    ax = fig.add_subplot(gs[0])
    ax.set_position([0.15, 0.15, plot_width / total_width, plot_height / total_height])

    # Define colors based on `group_order`
    if group_order is None:
        group_order = list(df_aggregated["Group"].unique())

    scalar_mappable = ScalarMappable(cmap=color_name)
    colors = scalar_mappable.to_rgba(range(len(group_order)))
    color_dict = {group: color for group, color in zip(group_order, colors)}

    legend_handles = {}

    for idx, row in df_aggregated.iterrows():
        model_paths = row["project_model"]
        group_label = row["Group"]
        color = color_dict[group_label]  # Get the color based on group_order

        # Load and aggregate calibration data
        expected, mean_observed, lower_bound, upper_bound = (
            load_and_aggregate_calibration_data(base_path, model_paths)
        )

        # Plot the mean line
        (line,) = ax.plot(expected, mean_observed, label=group_label, color=color)

        # Fill the shaded area
        ax.fill_between(expected, lower_bound, upper_bound, alpha=0.2, color=color)

        # Store line handles for the legend in order
        if group_label not in legend_handles:
            legend_handles[group_label] = line

    # Perfect calibration line
    (perfect_line,) = ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    legend_handles["Perfect Calibration"] = perfect_line

    # Sort legend handles based on group_order
    ordered_legend_handles = [
        legend_handles[group] for group in group_order if group in legend_handles
    ]
    ordered_legend_handles.append(legend_handles["Perfect Calibration"])
    if LEGEND_ON:
        ax.legend(
            handles=ordered_legend_handles,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )

    ax.set_title(title)
    ax.set_xlabel("Expected Proportion")
    ax.set_ylabel("Observed Proportion")
    ax.grid(True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    if save_dir:
        plot_name = f"{title.replace(' ', '_')}"
        save_plot(fig, save_dir, plot_name, tighten=True)

    plt.show()
    plt.close()


def move_model_folders(df, search_dirs, output_dir, overwrite=False):
    """
    Moves folders matching the unique 'model name' entries from df to output_dir.

    Parameters:
    df (pd.DataFrame): DataFrame containing a 'model name' column.
    search_dirs (list): List of directories to search for model folders.
    output_dir (str): Destination directory to move the model folders to.
    """
    # Get unique list of model names
    model_names = df["model name"].unique()

    # Ensure output_dir exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory '{output_dir}'.")

    # Iterate over each model name
    for model_name in model_names:
        found = False
        for search_dir in search_dirs:
            # Check if the search directory exists
            if not os.path.isdir(search_dir):
                print(f"Search directory '{search_dir}' does not exist. Skipping.")
                continue

            # Get list of immediate subdirectories in search_dir
            subdirs = [
                d
                for d in os.listdir(search_dir)
                if os.path.isdir(os.path.join(search_dir, d))
            ]

            if model_name in subdirs:
                source_dir = os.path.join(search_dir, model_name)
                dest_dir = os.path.join(output_dir, model_name)

                # Check if destination folder already exists
                if os.path.exists(dest_dir):
                    if overwrite:
                        shutil.copytree(source_dir, dest_dir, dirs_exist_ok=True)
                        print(f"Merged (Copied) '{source_dir}' to '{dest_dir}'.")
                    # print(f"Destination folder '{dest_dir}' already exists. Skipping move for '{model_name}'.")
                else:
                    try:
                        shutil.move(source_dir, dest_dir)
                        print(f"Moved '{source_dir}' to '{dest_dir}'.")
                    except Exception as e:
                        print(f"Error moving '{source_dir}' to '{dest_dir}': {e}")
                found = True
                break  # No need to continue searching other directories
        if not found:
            print(
                f"Model folder '{model_name}' not found in any of the search directories."
            )


def load_predictions(model_path):
    preds_path = os.path.join(model_path, "preds.pkl")

    return pd.read_pickle(preds_path)


def calculate_rmse_rejection_curve(
    preds,
    uncertainty_col="y_alea",
    true_label_col="y_true",
    pred_label_col="y_pred",
    normalize_rmse=False,
    random_rejection=False,
    unc_type=None,
    max_rejection_ratio=0.95,
):
    # First we choose which type of uncertainty to use
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
            f"Either provide valid uncertainty type or provide the uncertainty column name in the DataFrame"
            f"unc_type: {unc_type}, uncertainty_col: {uncertainty_col}"
        )

    # Sort the DataFrame based on the uncertainty column or shuffle it randomly
    if random_rejection:
        preds = preds.sample(frac=max_rejection_ratio).reset_index(
            drop=True
        )  # Shuffle the DataFrame randomly
    else:
        preds = preds.sort_values(by=uncertainty_col, ascending=False)

    max_rejection_index = int(len(preds) * max_rejection_ratio)
    rejection_steps = np.arange(0, max_rejection_index, step=int(len(preds) * 0.01))
    # print(f"{len(preds)=}")
    rejection_rates = rejection_steps / len(preds)
    # print(len(rejection_rates))
    rmses = []

    initial_rmse = mean_squared_error(
        preds[true_label_col], preds[pred_label_col], squared=False
    )

    # RRC calculation
    for i in rejection_steps:
        selected_preds = preds.iloc[i:]
        rmse = mean_squared_error(
            selected_preds[true_label_col],
            selected_preds[pred_label_col],
            squared=False,
        )
        if normalize_rmse:
            rmse /= initial_rmse
        rmses.append(rmse)
    # AUC calculation
    auc_arc = auc(rejection_rates, rmses)

    return rejection_rates, rmses, auc_arc


def calculate_rejection_curve(
    df,
    model_paths,
    unc_col,
    random_rejection=False,
    normalize_rmse=False,
    max_rejection_ratio=0.95,
):
    """
    Calculate RMSE rejection curves for given model paths.
    """
    aggregated_rmses = []
    auc_values = []
    rejection_rates = None

    for model_path in model_paths:
        preds = load_predictions(model_path)
        # check if preds were correctly loaded or not
        if preds.empty:
            print(f"Preds not loaded for model: {model_path}")
            continue

            # break
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

    return rejection_rates, mean_rmses, std_rmses, mean_auc, std_auc


def get_handles_labels(ax, group_order):
    # Custom legend order
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
    save_dir_plot: Optional[str] = None,
    add_to_title: str = "",
    normalize_rmse: bool = False,
    unc_type: str = "aleatoric",
    max_rejection_ratio: float = 0.95,
    group_order: Optional[List[str]] = None,
    fig_width: Optional[float] = None,
    fig_height: Optional[float] = None,
) -> pd.DataFrame:
    """
    Plot RMSE rejection curves for different groups and splits, ensuring that the **inner plot box**
    has fixed width and height.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing model paths and group labels.
    base_dir : str
        Path to load the rejection data.
    cmap : str, optional
        Colormap name for the lines, by default "tab10_r".
    save_dir_plot : Optional[str], optional
        Directory to save the plot, by default None.
    add_to_title : str, optional
        Additional string to append to the plot title, by default "".
    normalize_rmse : bool, optional
        If True, normalizes RMSE, by default False.
    unc_type : str, optional
        Type of uncertainty: "aleatoric", "epistemic", or "both".
    max_rejection_ratio : float, optional
        Maximum rejection ratio to consider, by default 0.95.
    group_order : Optional[List[str]], optional
        Order of groups in the legend, by default None.
    fig_width : Optional[float], optional
        Width of the **inner plot box** (not the full figure), by default None.
    fig_height : Optional[float], optional
        Height of the **inner plot box** (not the full figure), by default None.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing AUC-RRC statistics for each model group.
    """

    assert unc_type in ["aleatoric", "epistemic", "both"], "Invalid unc_type"
    unc_col = "y_alea" if unc_type == "aleatoric" else "y_eps"

    # Set default dimensions for the **inner plot box**
    plot_width = fig_width if fig_width else 6
    plot_height = fig_height if fig_height else 6

    # Compute total figure size dynamically
    total_width = plot_width + 4  # Space for legend
    total_height = plot_height + 2  # Space for x-axis labels

    fig = plt.figure(figsize=(total_width, total_height))
    gs = gridspec.GridSpec(
        1, 1, figure=fig, left=0.15, right=0.75, top=0.9, bottom=0.15
    )  # Define space for the main plot

    ax = fig.add_subplot(gs[0])
    ax.set_position([0.15, 0.15, plot_width / total_width, plot_height / total_height])

    # Use group_order for consistent coloring
    if group_order is None:
        group_order = list(df["Group"].unique())

    scalar_mappable = ScalarMappable(cmap=cmap)
    colors = scalar_mappable.to_rgba(range(len(group_order)))
    color_dict = {group: color for group, color in zip(group_order, colors)}

    df.loc[:, "model_path"] = df["project_model"].apply(
        lambda x: (
            str(os.path.join(base_dir, x)) if not str(x).startswith(base_dir) else x
        )
    )

    stats_dfs = []
    included_groups = df["Group"].unique()
    legend_handles = []

    for group in included_groups:
        group_data = df[df["Group"] == group]
        model_paths = group_data["model_path"].unique()

        rejection_rates, mean_rmses, std_rmses, mean_auc, std_auc = (
            calculate_rejection_curve(
                df,
                model_paths,
                unc_col,
                normalize_rmse=normalize_rmse,
                max_rejection_ratio=max_rejection_ratio,
            )
        )

        # **Plot the curve**
        (line,) = ax.plot(
            rejection_rates,
            mean_rmses,
            label=f"{group} (AUC-RRC: {mean_auc:.3f} ± {std_auc:.3f})",
            color=color_dict[group],
        )
        ax.fill_between(
            rejection_rates,
            mean_rmses - std_rmses,
            mean_rmses + std_rmses,
            color=color_dict[group],
            alpha=0.2,
        )
        legend_handles.append(line)

        stats_dfs.append(
            {
                "Model type": group.rsplit("_", 1)[1],
                "Split": group.rsplit("_", 1)[0],
                "Group": group,
                "AUC-RRC_mean": mean_auc,
                "AUC-RRC_std": std_auc,
            }
        )

    # Add baseline random rejection curves
    for split in df["Split"].unique():
        split_data = df[df["Split"] == split]
        model_paths = split_data["model_path"].unique()

        rejection_rates, mean_rmses, std_rmses, mean_auc, std_auc = (
            calculate_rejection_curve(
                df,
                model_paths,
                unc_col,
                random_rejection=True,
                normalize_rmse=normalize_rmse,
                max_rejection_ratio=max_rejection_ratio,
            )
        )

        # **Plot the random reject curve**
        (line,) = ax.plot(
            rejection_rates,
            mean_rmses,
            label=f"random reject - {split} (AUC-RRC: {mean_auc:.3f} ± {std_auc:.3f})",
            color="black",
            linestyle="--",
        )
        ax.fill_between(
            rejection_rates,
            mean_rmses - std_rmses,
            mean_rmses + std_rmses,
            color="grey",
            alpha=0.2,
        )
        legend_handles.append(line)

        stats_dfs.append(
            {
                "Model type": "random reject",
                "Split": split,
                "Group": f"random reject - {split}",
                "AUC-RRC_mean": mean_auc,
                "AUC-RRC_std": std_auc,
            }
        )

    # Plot settings
    ax.set_xlabel("Rejection Rate")
    ax.set_ylabel("RMSE" if not normalize_rmse else "Normalized RMSE")
    # ax.set_title(
    #     "RMSE-Rejection Curves"
    #     if not normalize_rmse
    #     else "Normalized RMSE-Rejection Curves"
    # )

    ax.set_xlim(0, max_rejection_ratio)
    ax.grid(True)

    # Order legend by `group_order`
    if LEGEND_ON:
        ordered_handles, ordered_labels = get_handles_labels(ax, group_order)
        ordered_handles += [legend_handles[-1]]  # Append random reject at the end
        ordered_labels += [legend_handles[-1].get_label()]

        ax.legend(
            handles=ordered_handles,
            loc="lower left",
        )

    # fig.subplots_adjust(left=0.12, right=0.85, top=0.9, bottom=0.2)

    # Save the plot
    plot_name = (
        f"rmse_rejection_curve_{add_to_title}"
        if add_to_title
        else "rmse_rejection_curve"
    )
    save_plot(fig, save_dir_plot, plot_name, tighten=True)

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
):
    """
    Plots AUC-RRC comparison bar plot with colors by Model type and hatches by Split.

    Ensures that only the **inner plot box** has fixed width and height, while the total figure
    adjusts dynamically to accommodate the legend, labels, and title.

    Parameters
    ----------
    stats_df : pd.DataFrame
        DataFrame containing AUC-RRC statistics.
    cmap : str, optional
        Colormap for the bars, by default "tab10_r".
    save_dir : Optional[str], optional
        Directory to save the plot, by default None.
    add_to_title : str, optional
        Additional string for the plot name, by default "".
    min_y_axis : float, optional
        Minimum value for the y-axis, by default 0.0.
    hatches_dict : Optional[Dict[str, str]], optional
        Dictionary mapping splits to hatch patterns, by default None.
    group_order : Optional[List[str]], optional
        Order of groups for consistent color mapping, by default None.
    fig_width : Optional[float], optional
        Width of the **inner plot box**, by default None.
    fig_height : Optional[float], optional
        Height of the **inner plot box**, by default None.

    Returns
    -------
    None
    """
    if hatches_dict is None:
        hatches_dict = {"stratified": "\\\\", "scaffold_cluster": "", "time": "///"}

    if group_order:
        all_groups = group_order + list(
            stats_df.loc[
                stats_df["Group"].str.startswith("random reject"), "Group"
            ].unique()
        )
        stats_df["Group"] = pd.Categorical(
            stats_df["Group"], categories=all_groups, ordered=True
        )
    else:
        all_groups = stats_df["Group"].unique().tolist()

    stats_df = stats_df.sort_values("Group").reset_index(drop=True)

    # **Sort splits according to `hatches_dict` order**
    splits = list(hatches_dict.keys())
    stats_df.loc[:, "Split"] = pd.Categorical(
        stats_df["Split"], categories=splits, ordered=True
    )
    stats_df = stats_df.sort_values("Split").reset_index(drop=True)

    unique_model_types = stats_df.loc[
        stats_df["Model type"] != "random reject", "Model type"
    ].unique()

    if color_dict is None:
        scalar_mappable = ScalarMappable(cmap=cmap)
        # Exclude "random reject" from color mapping, then add it manually as black

        # Color mapping, ordered by `unique_model_types`
        colors = scalar_mappable.to_rgba(range(len(unique_model_types)))
        color_dict = {model: color for model, color in zip(unique_model_types, colors)}

    color_dict["random reject"] = "black"  # Set "random reject" to black

    # Append "random reject" back to model types for plotting
    unique_model_types = np.append(unique_model_types, "random reject")

    # Calculate number of splits and bar spacing
    # splits = stats_df["Split"].unique()
    bar_width = 0.12
    group_spacing = 0.6

    # **Set default inner plot box size**
    plot_width = fig_width if fig_width else 6
    plot_height = fig_height if fig_height else 6

    # **Compute total figure size dynamically**
    total_width = plot_width + 4  # Add space for legend
    total_height = plot_height + 4  # Add space for axis labels

    # **Create the figure with fixed plot box dimensions**
    fig = plt.figure(figsize=(total_width, total_height))
    gs = gridspec.GridSpec(
        1, 1, figure=fig, left=0.15, right=0.75, top=0.9, bottom=0.15
    )  # Define space for the main plot
    ax = fig.add_subplot(gs[0])
    ax.set_position([0.15, 0.15, plot_width / total_width, plot_height / total_height])

    tick_positions = []
    tick_labels = []

    # **Plot bars for each split and group**
    for i, split in enumerate(splits):
        split_data = stats_df[stats_df["Split"] == split]
        split_data.loc[:, "Group"] = pd.Categorical(
            split_data["Group"], categories=all_groups, ordered=True
        )
        # split_data = split_data.sort_values("Group").reset_index(drop=True)

        for j, (_, row) in enumerate(split_data.iterrows()):
            position = (
                i * (len(unique_model_types) * bar_width + group_spacing)
                + j * bar_width
            )

            # **Plot the bars**
            ax.bar(
                position,
                height=row["AUC-RRC_mean"],
                yerr=row["AUC-RRC_std"],
                color=color_dict[row["Model type"]],
                edgecolor="white" if row["Model type"] == "random reject" else "black",
                hatch=hatches_dict[row["Split"]],
                width=bar_width,
            )

        # **Add tick labels for each split**
        center_position = (
            i * (len(unique_model_types) * bar_width + group_spacing)
            + (len(unique_model_types) * bar_width) / 2
        )
        tick_positions.append(center_position)
        tick_labels.append(split)

    # **Create legend**
    def create_stats_legend(
        color_dict: Dict[str, str],
        hatches_dict: Dict[str, str],
        splits: List[str],
        model_types: Union[List[str], np.ndarray],
    ):
        patches = []
        for split in splits:
            for model in model_types:
                label = f"{split} {model}"
                hatch_color = "white" if model == "random reject" else "black"
                patch = mpatches.Patch(
                    facecolor=color_dict[model],
                    hatch=hatches_dict[split],
                    edgecolor=hatch_color,
                    label=label,
                )
                patches.append(patch)
        return patches

    if LEGEND_ON:
        legend_elements = create_stats_legend(
            color_dict, hatches_dict, splits, unique_model_types
        )

        # **Move the legend outside the plot box**
        ax.legend(
            handles=legend_elements,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0,
            frameon=False,
        )

    # **Axes settings**
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)
    # Uncomment if needed. Adjust as needed.
    # ax.set_xlabel("Splits")
    ax.set_ylabel("RRC-AUC")
    ax.set_ylim(min_y_axis, 1.0)

    # fig.subplots_adjust(left=0.12, right=0.85, top=0.9, bottom=0.2)
    # plt.tight_layout()  # Ensures compact figure layout

    # **Save and show the plot**
    plot_name = f"auc_comparison_barplot_{cmap}"
    plot_name += f"_{add_to_title}" if add_to_title else ""
    save_plot(fig, save_dir, plot_name, tighten=True)

    plt.show()
    plt.close()


# we want to create a function to save stats_df to a csv file
def save_stats_df(stats_df, save_dir, add_to_title=""):
    stats_df.to_csv(os.path.join(save_dir, f"stats_df_{add_to_title}.csv"), index=False)


def load_stats_df(save_dir, add_to_title=""):
    return pd.read_csv(os.path.join(save_dir, f"stats_df_{add_to_title}.csv"))


# lets run this file
if __name__ == "__main__":
    plt.style.use("tableau-colorblind10")

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Process input parameters for the script."
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="papyrus",
    )
    parser.add_argument(
        "--activity_type",
        type=str,
        required=True,
        help="The type of activity (e.g., 'kx', 'xc50').",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        required=True,
        # default='2025-01-30-kx-all',
        help="The name of the project.",
    )

    parser.add_argument(
        "--color", type=str, default="tab10_r", help="name of the color map"
    )

    parser.add_argument(
        "--corr_color", type=str, default="YlGnBu", help="name of the color map"
    )

    args = parser.parse_args()
    data_name = "papyrus"
    type_n_targets = "all"
    activity_type = args.activity_type
    project_name = args.project_name
    # LEGEND_ON = args.legend_on
    color_map = args.color
    corr_cmap = args.corr_color

    ############################# Testing ################################
    # color_map = None
    # color_map = "tableau-colorblind10"
    # data_name = "papyrus"
    # color_map = "tab10_r"
    # corr_cmap = "YlGnBu"
    # activity_type = "kx"
    # type_n_targets = "all"
    # # project_name = '2025-01-08-xc50-all'
    # project_name = "2025-02-19-kx-test"
    # ############################## Testing ################################

    project_out_name = project_name

    data_specific_path = f"{data_name}/{activity_type}/{type_n_targets}"

    file_1 = f"/users/home/bkhalil/Repos/uqdd/uqdd/figures/papyrus/{activity_type}/all/reassess-runs_ensemble_mcdp_{activity_type}/metrics.csv"
    file_2 = f"/users/home/bkhalil/Repos/uqdd/uqdd/figures/papyrus/{activity_type}/all/reassess-runs_evidential_{activity_type}/metrics.csv"

    save_dir = f"/users/home/bkhalil/Repos/uqdd/uqdd/figures/{data_specific_path}/{project_out_name}/{color_map}/"
    save_dir_no_time = f"/users/home/bkhalil/Repos/uqdd/uqdd/figures/{data_specific_path}/{project_out_name}-no-time/{color_map}/"
    base_path = "/users/home/bkhalil/Repos/uqdd/uqdd/figures/"

    df_1 = pd.read_csv(file_1, header=0)
    df_2 = pd.read_csv(file_2, header=0)
    df_main = pd.concat([df_1, df_2])

    # # replace random with stratified for the random split
    df_main["Split"] = df_main["Split"].apply(
        lambda x: "stratified" if x == "random" else x
    )

    df_merged = df_main.copy()

    # Remove some rows where MCDP experiment was run
    df_merged = df_merged[
        ~(
            (df_merged["Model type"] == "mcdropout")
            & (df_merged["Split"] == "scaffold_cluster")
            & (df_merged["dropout"] == 0.2)
        )
    ]
    df_merged = df_merged[
        ~(
            (df_merged["Model type"] == "mcdropout")
            & (df_merged["Split"] == "stratified")
            & (df_merged["dropout"] == 0.1)
        )
    ]
    df_merged = df_merged[
        ~(
            (df_merged["Model type"] == "mcdropout")
            & (df_merged["Split"] == "time")
            & (df_merged["dropout"] == 0.1)
        )
    ]

    df_merged["Group"] = df_merged.apply(
        lambda row: f"{row['Split']}_{row['Model type']}", axis=1
    )

    # Extracting the necessary parts for plotting and make copies to avoid SettingWithCopyWarning
    # df_pcm with exact match of task pcm
    df_pcm = df_merged[df_merged["Task"] == "PCM"].copy()
    df_before_calib = df_merged[df_merged["Task"] == "PCM_before_calibration"].copy()
    df_before_calib["Calibration"] = "Before Calibration"

    df_after_calib = df_merged[
        df_merged["Task"] == "PCM_after_calibration_with_isotonic_regression"
    ].copy()
    df_after_calib["Calibration"] = "After Calibration"

    df_calib = pd.concat([df_before_calib, df_after_calib])
    df_calib_no_time = df_calib.copy()[df_calib["Split"] != "time"]
    # SUBSET 100
    subdf_pcm = df_merged[df_merged["Task"] == "PCM_subset100"].copy()
    subdf_before_calib = df_merged[
        df_merged["Task"] == "PCM_before_calibration_subset100"
    ].copy()
    subdf_before_calib["Calibration"] = "Before Calibration"

    subdf_after_calib = df_merged[
        df_merged["Task"] == "PCM_after_calibration_with_isotonic_regression_subset100"
    ].copy()
    subdf_after_calib["Calibration"] = "After Calibration"

    subdf_calib = pd.concat([subdf_before_calib, subdf_after_calib])
    subdf_calib_no_time = subdf_calib.copy()[subdf_calib["Split"] != "time"]

    os.makedirs(save_dir, exist_ok=True)
    output_file_path = os.path.join(save_dir, "final_aggregated.csv")
    df_no_time = df_pcm.copy()[df_pcm["Split"] != "time"]

    print(f"{df_pcm.shape=}")
    print(f"{df_no_time.shape=}")

    final_aggregated = aggregate_results_csv(
        df_pcm, group_cols, numeric_cols, string_cols, order_by, output_file_path
    )
    final_aggregated["Group"] = final_aggregated.apply(
        lambda row: f"{row['Split']}_{row['Model type']}", axis=1
    )

    os.makedirs(save_dir_no_time, exist_ok=True)
    output_file_path_no_time = os.path.join(
        save_dir_no_time, "final_aggregated_no_time.csv"
    )
    final_aggregated_no_time = aggregate_results_csv(
        df_no_time,
        group_cols,
        numeric_cols,
        string_cols,
        order_by,
        output_file_path_no_time,
    )
    final_aggregated_no_time["Group"] = final_aggregated_no_time.apply(
        lambda row: f"{row['Split']}_{row['Model type']}", axis=1
    )

    df_pcm.to_csv(os.path.join(save_dir, "final.csv"), index=False)
    df_no_time.to_csv(os.path.join(save_dir_no_time, "final_no_time.csv"), index=False)

    highly_correlated_metrics = find_highly_correlated_metrics(
        df_pcm, accmetrics, threshold=0.9, save_dir=save_dir, cmap=corr_cmap
    )
    highly_correlated_metrics_no_time = find_highly_correlated_metrics(
        df_no_time, accmetrics, threshold=0.9, save_dir=save_dir_no_time, cmap=corr_cmap
    )
    highly_correlated_uctmetrics = find_highly_correlated_metrics(
        df_pcm, uctmetrics, threshold=0.9, save_dir=save_dir, cmap=corr_cmap
    )
    highly_correlated_uctmetrics_no_time = find_highly_correlated_metrics(
        df_no_time, uctmetrics, threshold=0.9, save_dir=save_dir_no_time, cmap=corr_cmap
    )
    uctmetrics_uncorr = [
        "Miscalibration Area",
        "Sharpness",
        "CRPS",
        "NLL",
        "Interval",
    ]

    highly_correlated_uctmetrics_uncorr = find_highly_correlated_metrics(
        df_pcm, uctmetrics_uncorr, threshold=0.9, save_dir=save_dir, cmap=corr_cmap
    )
    highly_correlated_uctmetrics_uncorr_no_time = find_highly_correlated_metrics(
        df_no_time,
        uctmetrics_uncorr,
        threshold=0.9,
        save_dir=save_dir_no_time,
        cmap=corr_cmap,
    )

    # plot_metrics(df_pcm, accmetrics, cmap="tab10_r", save_dir=save_dir_no_time, hatches_dict=hatches_dict_no_time, group_order=group_order_no_time)
    color_dict = plot_metrics(
        df_no_time,
        accmetrics,  # 6
        cmap=color_map,
        save_dir=save_dir_no_time,
        hatches_dict=hatches_dict_no_time,
        group_order=group_order_no_time,
        fig_width=12,
        fig_height=3,
    )

    # plot_metrics(df_pcm, accmetrics2, cmap=color_map, save_dir=save_dir, hatches_dict=hatches_dict, group_order=group_order)
    plot_metrics(
        df_no_time,
        accmetrics2,  # 3
        cmap=color_map,
        save_dir=save_dir_no_time,
        hatches_dict=hatches_dict_no_time,
        group_order=group_order_no_time,
        fig_width=6,
        fig_height=3,
    )

    # plot_metrics(df_pcm, uctmetrics_uncorr, cmap="tab10_r", save_dir=save_dir, hatches_dict=hatches_dict, group_order=group_order)
    plot_metrics(
        df_no_time,
        uctmetrics_uncorr,  # 5
        cmap=color_map,
        save_dir=save_dir_no_time,
        hatches_dict=hatches_dict_no_time,
        group_order=group_order_no_time,
        fig_width=10,
        fig_height=3,
    )

    # plot_metrics(df_pcm, uctmetrics_uncorr, cmap="tab10_r", save_dir=save_dir, hatches_dict=hatches_dict, group_order=group_order)
    plot_metrics(
        df_no_time,
        [
            "Miscalibration Area",
            "Sharpness",
            "CRPS",
        ],
        cmap=color_map,
        save_dir=save_dir_no_time,
        hatches_dict=hatches_dict_no_time,
        group_order=group_order_no_time,
        fig_width=6,  # 3
        fig_height=3,
    )
    plot_metrics(
        df_no_time,
        [
            "NLL",
            "Interval",
        ],
        cmap=color_map,
        save_dir=save_dir_no_time,
        hatches_dict=hatches_dict_no_time,
        group_order=group_order_no_time,
        fig_width=4,  # 2
        fig_height=3,
    )

    plot_metrics(
        df_no_time,
        [
            "CRPS",
            "Sharpness",
            "Interval",
        ],
        cmap=color_map,
        save_dir=save_dir_no_time,
        hatches_dict=hatches_dict_no_time,
        group_order=group_order_no_time,
        fig_width=6,  # 3
        fig_height=3,
    )

    for m in accmetrics2:
        # plot_metrics(df_pcm, [m], cmap="tab10_r", save_dir=save_dir, hatches_dict=hatches_dict, group_order=group_order)
        plot_metrics(
            df_no_time,
            [m],  # 1
            cmap=color_map,
            save_dir=save_dir_no_time,
            hatches_dict=hatches_dict_no_time,
            group_order=group_order_no_time,
            fig_width=2,
            fig_height=3,
        )

    for m in uctmetrics_uncorr:
        # plot_metrics(df_pcm, [m], cmap="tab10_r", save_dir=save_dir, hatches_dict=hatches_dict, group_order=group_order)
        plot_metrics(
            df_no_time,
            [m],
            cmap=color_map,
            save_dir=save_dir_no_time,
            hatches_dict=hatches_dict_no_time,
            group_order=group_order_no_time,
            fig_width=2,
            fig_height=3,
        )

    # plot_comparison_metrics(df_pcm, uctmetrics_uncorr, cmap=color_map, save_dir=save_dir)
    plot_comparison_metrics(
        df_calib_no_time,
        uctmetrics_uncorr,  # 10
        cmap=color_map,
        save_dir=save_dir_no_time,
        fig_width=20,
        fig_height=3,
    )

    mc_list = ["RMS Calibration", "MA Calibration", "Miscalibration Area"]
    # plot_comparison_metrics(df_pcm, mc_list, cmap=color_map, save_dir=save_dir)
    plot_comparison_metrics(
        df_calib_no_time,
        mc_list,  # 6
        cmap=color_map,
        save_dir=save_dir_no_time,
        fig_width=12,
        fig_height=3,
    )

    for mc in mc_list:
        # plot_comparison_metrics(df_calib, ['Miscalibration Area'], cmap="tab10_r", save_dir=save_dir)
        plot_comparison_metrics(
            df_calib_no_time,
            [mc],  # 2
            cmap=color_map,
            save_dir=save_dir_no_time,
            fig_width=4,
            fig_height=3,
        )

    plot_calibration_data(
        final_aggregated_no_time,
        base_path,
        save_dir_no_time,
        title="Calibration Curves for Models",
        color_name=color_map,
        group_order=group_order_no_time,
        fig_width=5,
        fig_height=5,
    )

    # Now lets do calibration for each one separately
    dfs = [
        final_aggregated_no_time.iloc[[i]] for i in range(len(final_aggregated_no_time))
    ]
    for i, df in enumerate(dfs):
        plot_calibration_data(
            df,
            base_path,
            save_dir_no_time,
            title=f"Calibration Curves for {df['Group'].values[0]}",
            color_name=color_map,
            group_order=group_order_no_time,
            fig_width=5,
            fig_height=5,
        )
    # Now lets do calibration curves for each split separately
    dfs = [
        final_aggregated_no_time[final_aggregated_no_time["Split"] == s]
        for s in final_aggregated_no_time["Split"].unique()
    ]
    for i, df in enumerate(dfs):
        plot_calibration_data(
            df,
            base_path,
            save_dir_no_time,
            title=f"Calibration Curves for {df['Split'].values[0]}",
            color_name=color_map,
            group_order=group_order_no_time,
            fig_width=5,
            fig_height=5,
        )

    df_pcm_stratified = df_pcm[df_pcm["Split"] == "stratified"]
    df_pcm_scaffold = df_pcm[df_pcm["Split"] == "scaffold_cluster"]
    df_pcm_time = df_pcm[df_pcm["Split"] == "time"]
    # print(df_pcm_stratified.shape, df_pcm_scaffold.shape, df_pcm_time.shape)

    save_dir_plot = os.path.join(save_dir_no_time, "rrcs")
    # save_dir_no_time_rrcs_plot = os.path.join(save_dir_no_time, "rrcs")

    # make dir if not exist
    os.makedirs(save_dir_plot, exist_ok=True)
    # os.makedirs(save_dir_no_time_rrcs_plot, exist_ok=True)

    uct_types = ["aleatoric", "epistemic", "both"]
    for uct_t in uct_types:
        for normalize_rmse in [True, False]:
            add_to_title = "-normalized" if normalize_rmse else ""
            add_to_title += "-" + uct_t
            stats_df = plot_rmse_rejection_curves(
                df_no_time,
                base_path,
                cmap=color_map,
                save_dir_plot=save_dir_plot,
                add_to_title="all" + add_to_title,
                normalize_rmse=normalize_rmse,
                unc_type=uct_t,
                max_rejection_ratio=0.95,
                group_order=group_order_no_time,
                fig_width=6,
                fig_height=5,
            )

            plot_auc_comparison(
                stats_df,
                cmap=color_map,
                color_dict=color_dict,
                save_dir=save_dir_plot,
                add_to_title="all" + add_to_title,
                hatches_dict=hatches_dict_no_time,
                group_order=group_order_no_time,
                fig_width=4,
                fig_height=3,
            )
            plot_auc_comparison(
                stats_df,
                cmap=color_map,
                color_dict=color_dict,
                save_dir=save_dir_plot,
                add_to_title="all" + add_to_title + "-min-0.5",
                hatches_dict=hatches_dict_no_time,
                group_order=group_order_no_time,
                min_y_axis=0.5,
                fig_width=4,
                fig_height=3,
            )

            save_stats_df(stats_df, save_dir_plot, add_to_title="all" + add_to_title)

            for name, df in zip(
                ["stratified", "scaffold"], [df_pcm_stratified, df_pcm_scaffold]
            ):
                stats_df = plot_rmse_rejection_curves(
                    df,
                    base_path,
                    cmap=color_map,
                    save_dir_plot=save_dir_plot,
                    add_to_title=name + add_to_title,
                    normalize_rmse=normalize_rmse,
                    unc_type=uct_t,
                    max_rejection_ratio=0.95,
                    group_order=group_order_no_time,
                    fig_width=6,
                    fig_height=5,
                )
                plot_auc_comparison(
                    stats_df,
                    cmap=color_map,
                    color_dict=color_dict,
                    save_dir=save_dir_plot,
                    add_to_title=name + add_to_title,
                    hatches_dict=hatches_dict_no_time,
                    group_order=group_order_no_time,
                    fig_width=2,
                    fig_height=3,
                )

                plot_auc_comparison(
                    stats_df,
                    cmap=color_map,
                    color_dict=color_dict,
                    save_dir=save_dir_plot,
                    add_to_title=name + add_to_title + "-min-0.5",
                    hatches_dict=hatches_dict_no_time,
                    group_order=group_order_no_time,
                    min_y_axis=0.5,
                    fig_width=2,
                    fig_height=3,
                )

                save_stats_df(stats_df, save_dir_plot, add_to_title=name + add_to_title)

    plot_pairplot(
        df_no_time,
        "Pairplot for Accuracy Metrics",
        accmetrics,
        save_dir=save_dir_no_time,
        cmap=color_map,
        group_order=group_order_no_time,
    )
    plot_pairplot(
        df_no_time,
        "Pairplot for Uncertainty Metrics",
        uctmetrics,
        save_dir=save_dir_no_time,
        cmap=color_map,
        group_order=group_order_no_time,
    )
