import math
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import scikit_posthocs as sp
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr, wilcoxon, friedmanchisquare
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import (
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

INTERACTIVE_MODE = hasattr(sys, "ps1") or sys.flags.interactive


# HELPER FUNCTIONS FOR PLOTTING #
def save_plot(fig, save_dir, plot_name, tighten=True, show_legend=False):
    # Get current axes
    ax = fig.gca()
    # Remove legend only if it exists and show_legend is False
    if not show_legend:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    if tighten:
        try:
            # Suppress the specific tight_layout warning
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="This figure includes Axes that are not compatible with tight_layout",
                )
                fig.tight_layout()
        except (ValueError, RuntimeError):
            # If tight_layout fails with an actual exception, use subplots_adjust instead
            fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    if save_dir and tighten:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(
            os.path.join(save_dir, f"{plot_name}.png"), dpi=300, bbox_inches="tight"
        )
        fig.savefig(os.path.join(save_dir, f"{plot_name}.svg"), bbox_inches="tight")
        fig.savefig(
            os.path.join(save_dir, f"{plot_name}.pdf"), dpi=300, bbox_inches="tight"
        )

    elif save_dir and not tighten:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f"{plot_name}.png"), dpi=300)
        fig.savefig(os.path.join(save_dir, f"{plot_name}.svg"))
        fig.savefig(os.path.join(save_dir, f"{plot_name}.pdf"), dpi=300)


def calc_regression_metrics(df, cycle_col, val_col, pred_col, thresh):
    """
    Calculate regression metrics (MAE, MSE, R2, prec, recall) for each method and split

    :param df: input dataframe must contain columns [method, split] as well the columns specified in the arguments
    :param cycle_col: column indicating the cross-validation fold
    :param val_col: column with the ground truth value
    :param pred_col: column with predictions
    :param thresh: threshold for binary classification
    :return: a dataframe with [cv_cycle, method, split, mae, mse, r2, prec, recall]
    """
    df_in = df.copy()
    metric_ls = ["mae", "mse", "r2", "rho", "prec", "recall"]
    metric_list = []
    df_in["true_class"] = df_in[val_col] > thresh
    # Make sure the thresh variable creates 2 classes
    assert (
        len(df_in.true_class.unique()) == 2
    ), "Binary classification requires two classes"
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
    metric_df = pd.DataFrame(
        metric_list, columns=["cv_cycle", "method", "split"] + metric_ls
    )
    return metric_df


# Bootstrap functionality
def bootstrap_ci(data, func=np.mean, n_bootstrap=1000, ci=95, random_state=42):
    """
    Calculate bootstrap confidence intervals for a statistic.
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
    Perform repeated measures Tukey HSD test on the given dataframe.

    Parameters:
    df (pd.DataFrame): Input dataframe containing the data.
    metric (str): The metric column name to perform the test on.
    group_col (str): The column name indicating the groups.
    alpha (float): Significance level for the test. Default is 0.05.
    sort (bool): Whether to sort the output tables. Default is False.

    Returns:
    tuple: A tuple containing:
        - result_tab (pd.DataFrame): DataFrame with pairwise comparisons and adjusted p-values.
        - df_means (pd.DataFrame): DataFrame with mean values for each group.
        - df_means_diff (pd.DataFrame): DataFrame with mean differences between groups.
        - pc (pd.DataFrame): DataFrame with adjusted p-values for pairwise comparisons.
    """
    if sort and direction_dict and metric in direction_dict:
        if direction_dict[metric] == "maximize":
            df_means = (
                df.groupby(group_col)
                .mean(numeric_only=True)
                .sort_values(metric, ascending=False)
            )
        elif direction_dict[metric] == "minimize":
            df_means = (
                df.groupby(group_col)
                .mean(numeric_only=True)
                .sort_values(metric, ascending=True)
            )
        else:
            raise ValueError("Invalid direction. Expected 'maximize' or 'minimize'.")
    else:
        df_means = df.groupby(group_col).mean(numeric_only=True)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="divide by zero encountered in scalar divide",
        )
        aov = pg.rm_anova(
            dv=metric, within=group_col, subject="cv_cycle", data=df, detailed=True
        )
    mse = aov.loc[1, "MS"]
    df_resid = aov.loc[1, "DF"]

    methods = df_means.index
    n_groups = len(methods)
    n_per_group = df[group_col].value_counts().mean()

    tukey_se = np.sqrt(2 * mse / (n_per_group))
    q = qsturng(1 - alpha, n_groups, df_resid)

    num_comparisons = len(methods) * (len(methods) - 1) // 2
    result_tab = pd.DataFrame(
        index=range(num_comparisons),
        columns=["group1", "group2", "meandiff", "lower", "upper", "p-adj"],
    )

    df_means_diff = pd.DataFrame(index=methods, columns=methods, data=0.0)
    pc = pd.DataFrame(index=methods, columns=methods, data=1.0)

    # Calculate pairwise mean differences and adjusted p-values
    row_idx = 0
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i < j:
                group1 = df[df[group_col] == method1][metric]
                group2 = df[df[group_col] == method2][metric]
                mean_diff = group1.mean() - group2.mean()
                studentized_range = np.abs(mean_diff) / tukey_se
                adjusted_p = psturng(studentized_range * np.sqrt(2), n_groups, df_resid)
                if isinstance(adjusted_p, np.ndarray):
                    adjusted_p = adjusted_p[0]
                lower = mean_diff - (q / np.sqrt(2) * tukey_se)
                upper = mean_diff + (q / np.sqrt(2) * tukey_se)
                result_tab.loc[row_idx] = [
                    method1,
                    method2,
                    mean_diff,
                    lower,
                    upper,
                    adjusted_p,
                ]
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


# -------------- Plotting routines -------------------#


def make_boxplots_parametric(df, metric_ls):
    """
    Create boxplots for each metric using repeated measures ANOVA.

    Parameters:
    df (pd.DataFrame): Input dataframe containing the data.
    metric_ls (list of str): List of metric column names to create boxplots for.

    Returns:
    None
    """
    sns.set_context("notebook")
    sns.set(rc={"figure.figsize": (4, 3)}, font_scale=1.5)
    sns.set_style("whitegrid")
    figure, axes = plt.subplots(
        1, len(metric_ls), sharex=False, sharey=False, figsize=(28, 8)
    )
    # figure, axes = plt.subplots(1, 3, sharex=False, sharey=False, figsize=(16, 8))

    for i, stat in enumerate(metric_ls):
        model = AnovaRM(
            data=df, depvar=stat, subject="cv_cycle", within=["method"]
        ).fit()
        p_value = model.anova_table["Pr > F"].iloc[0]
        ax = sns.boxplot(
            y=stat,
            x="method",
            hue="method",
            ax=axes[i],
            data=df,
            palette="Set2",
            legend=False,
        )
        title = stat.upper()
        ax.set_title(f"p={p_value:.1e}")
        ax.set_xlabel("")
        ax.set_ylabel(title)
        x_tick_labels = ax.get_xticklabels()
        label_text_list = [x.get_text() for x in x_tick_labels]
        new_xtick_labels = ["\n".join(x.split("_")) for x in label_text_list]
        ax.set_xticks(list(range(0, len(x_tick_labels))))
        ax.set_xticklabels(new_xtick_labels)
    plt.tight_layout()


def make_boxplots_nonparametric(df, metric_ls):
    sns.set_context("notebook")
    sns.set(rc={"figure.figsize": (4, 3)}, font_scale=1.5)
    sns.set_style("whitegrid")
    figure, axes = plt.subplots(1, 6, sharex=False, sharey=False, figsize=(28, 8))

    for i, stat in enumerate(metric_ls):
        friedman = pg.friedman(df, dv=stat, within="method", subject="cv_cycle")[
            "p-unc"
        ].values[0]
        ax = sns.boxplot(
            y=stat,
            x="method",
            hue="method",
            ax=axes[i],
            data=df,
            palette="Set2",
            legend=False,
        )
        title = stat.replace("_", " ").upper()
        ax.set_title(f"p={friedman:.1e}")
        ax.set_xlabel("")
        ax.set_ylabel(title)
        x_tick_labels = ax.get_xticklabels()
        label_text_list = [x.get_text() for x in x_tick_labels]
        new_xtick_labels = ["\n".join(x.split("_")) for x in label_text_list]
        ax.set_xticks(list(range(0, len(x_tick_labels))))
        ax.set_xticklabels(new_xtick_labels)
    plt.tight_layout()


def make_sign_plots_nonparametric(df, metric_ls):
    heatmap_args = {
        "linewidths": 0.25,
        "linecolor": "0.5",
        "clip_on": True,
        "square": True,
    }
    sns.set(rc={"figure.figsize": (4, 3)}, font_scale=1.5)
    figure, axes = plt.subplots(1, 6, sharex=False, sharey=True, figsize=(26, 8))

    for i, stat in enumerate(metric_ls):
        pc = sp.posthoc_conover_friedman(
            df,
            y_col=stat,
            group_col="method",
            block_col="cv_cycle",
            p_adjust="holm",
            melted=True,
        )
        sub_ax, sub_c = sp.sign_plot(
            pc, **heatmap_args, ax=axes[i], xticklabels=True
        )  # Update xticklabels parameter
        sub_ax.set_title(stat.upper())


def make_critical_difference_diagrams(df, metric_ls):
    figure, axes = plt.subplots(6, 1, sharex=True, sharey=False, figsize=(16, 10))
    for i, stat in enumerate(metric_ls):
        avg_rank = df.groupby("cv_cycle")[stat].rank(pct=True).groupby(df.method).mean()
        pc = sp.posthoc_conover_friedman(
            df,
            y_col=stat,
            group_col="method",
            block_col="cv_cycle",
            p_adjust="holm",
            melted=True,
        )
        sp.critical_difference_diagram(avg_rank, pc, ax=axes[i])
        axes[i].set_title(stat.upper())
    plt.tight_layout()


def make_normality_diagnostic(df, metric_ls):
    """
    Create a normality diagnostic plot grid with histograms and QQ plots for the given metrics.

    Parameters:
    df (pd.DataFrame): Input dataframe containing the data.
    metric_ls (list of str): List of metrics to create plots for.

    Returns:
    None
    """
    df_norm = df.copy()

    for metric in metric_ls:
        df_norm[metric] = df_norm[metric] - df_norm.groupby("method")[metric].transform(
            "mean"
        )

    df_norm = df_norm.melt(
        id_vars=["cv_cycle", "method", "split"],
        value_vars=metric_ls,
        var_name="metric",
        value_name="value",
    )

    sns.set_context("notebook", font_scale=1.5)
    sns.set_style("whitegrid")

    metrics = df_norm["metric"].unique()
    n_metrics = len(metrics)

    fig, axes = plt.subplots(2, n_metrics, figsize=(20, 10))

    for i, metric in enumerate(metrics):
        ax = axes[0, i]
        sns.histplot(df_norm[df_norm["metric"] == metric]["value"], kde=True, ax=ax)
        ax.set_title(f"{metric}", fontsize=16)

    for i, metric in enumerate(metrics):
        ax = axes[1, i]
        metric_data = df_norm[df_norm["metric"] == metric]["value"]
        stats.probplot(metric_data, dist="norm", plot=ax)
        ax.set_title("")

    plt.tight_layout()


def mcs_plot(
    pc,
    effect_size,
    means,
    labels=True,
    cmap=None,
    cbar_ax_bbox=None,
    ax=None,
    show_diff=True,
    cell_text_size=16,
    axis_text_size=12,
    show_cbar=True,
    reverse_cmap=False,
    vlim=None,
    **kwargs,
):
    """
    Create a multiple comparison of means plot using a heatmap.

    Parameters:
    pc (pd.DataFrame): DataFrame containing p-values for pairwise comparisons.
    effect_size (pd.DataFrame): DataFrame containing effect sizes for pairwise comparisons.
    means (pd.Series): Series containing mean values for each group.
    labels (bool): Whether to show labels on the axes. Default is True.
    cmap (str): Colormap to use for the heatmap. Default is None.
    cbar_ax_bbox (tuple): Bounding box for the colorbar axis. Default is None.
    ax (matplotlib.axes.Axes): The axes on which to plot the heatmap. Default is None.
    show_diff (bool): Whether to show the mean differences in the plot. Default is True.
    cell_text_size (int): Font size for the cell text. Default is 16.
    axis_text_size (int): Font size for the axis text. Default is 12.
    show_cbar (bool): Whether to show the colorbar. Default is True.
    reverse_cmap (bool): Whether to reverse the colormap. Default is False.
    vlim (float): Limit for the colormap. Default is None.
    **kwargs: Additional keyword arguments for the heatmap.

    Returns:
    matplotlib.axes.Axes: The axes with the heatmap.
    """
    for key in ["cbar", "vmin", "vmax", "center"]:
        if key in kwargs:
            del kwargs[key]

    if not cmap:
        cmap = "coolwarm"
    if reverse_cmap:
        cmap = cmap + "_r"

    significance = pc.copy().astype(object)
    significance[(pc < 0.001) & (pc >= 0)] = "***"
    significance[(pc < 0.01) & (pc >= 0.001)] = "**"
    significance[(pc < 0.05) & (pc >= 0.01)] = "*"
    significance[(pc >= 0.05)] = ""

    np.fill_diagonal(significance.values, "")

    # Create a DataFrame for the annotations
    if show_diff:
        annotations = effect_size.round(3).astype(str) + significance
    else:
        annotations = significance

    hax = sns.heatmap(
        effect_size,
        cmap=cmap,
        annot=annotations,
        fmt="",
        cbar=show_cbar,
        ax=ax,
        annot_kws={"size": cell_text_size},
        vmin=-2 * vlim if vlim else None,
        vmax=2 * vlim if vlim else None,
        **kwargs,
    )

    if labels:
        label_list = list(means.index)
        x_label_list = [x + f"\n{means.loc[x].round(2)}" for x in label_list]
        y_label_list = [x + f"\n{means.loc[x].round(2)}\n" for x in label_list]
        hax.set_xticklabels(
            x_label_list,
            size=axis_text_size,
            ha="center",
            va="top",
            rotation=0,
            rotation_mode="anchor",
        )
        hax.set_yticklabels(
            y_label_list,
            size=axis_text_size,
            ha="center",
            va="center",
            rotation=90,
            rotation_mode="anchor",
        )

    hax.set_xlabel("")
    hax.set_ylabel("")

    return hax


def make_mcs_plot_grid(
    df,
    stats,
    group_col,
    alpha=0.05,
    figsize=(20, 10),
    direction_dict={},
    effect_dict={},
    show_diff=True,
    cell_text_size=16,
    axis_text_size=12,
    title_text_size=16,
    sort_axes=False,
):
    """
    Create a grid of multiple comparison of means plots using Tukey HSD test results.

    Parameters:
    df (pd.DataFrame): Input dataframe containing the data.
    stats (list of str): List of statistical metrics to create plots for.
    group_col (str): The column name indicating the groups.
    alpha (float): Significance level for the Tukey HSD test. Default is 0.05.
    figsize (tuple): Size of the figure. Default is (20, 10).
    direction_dict (dict): Dictionary indicating whether to minimize or maximize each metric.
    effect_dict (dict): Dictionary with effect size limits for each metric.
    show_diff (bool): Whether to show the mean differences in the plot. Default is True.
    cell_text_size (int): Font size for the cell text. Default is 16.
    axis_text_size (int): Font size for the axis text. Default is 12.
    title_text_size (int): Font size for the title text. Default is 16.
    sort (bool): Whether to sort the axes. Default is False.

    Returns:
    None
    """
    nrow = math.ceil(len(stats) / 3)
    fig, ax = plt.subplots(nrow, 3, figsize=figsize)

    # Set defaults
    for key in ["r2", "rho", "prec", "recall", "mae", "mse"]:
        direction_dict.setdefault(
            key, "maximize" if key in ["r2", "rho", "prec", "recall"] else "minimize"
        )

    for key in ["r2", "rho", "prec", "recall"]:
        effect_dict.setdefault(key, 0.1)

    direction_dict = {k.lower(): v for k, v in direction_dict.items()}
    effect_dict = {k.lower(): v for k, v in effect_dict.items()}

    for i, stat in enumerate(stats):
        stat = stat.lower()

        row = i // 3
        col = i % 3

        if stat not in direction_dict:
            raise ValueError(
                f"Stat '{stat}' is missing in direction_dict. Please set its value."
            )
        if stat not in effect_dict:
            raise ValueError(
                f"Stat '{stat}' is missing in effect_dict. Please set its value."
            )

        reverse_cmap = False
        if direction_dict[stat] == "minimize":
            reverse_cmap = True

        _, df_means, df_means_diff, pc = rm_tukey_hsd(
            df, stat, group_col, alpha, sort_axes, direction_dict
        )

        hax = mcs_plot(
            pc,
            effect_size=df_means_diff,
            means=df_means[stat],
            show_diff=show_diff,
            ax=ax[row, col],
            cbar=True,
            cell_text_size=cell_text_size,
            axis_text_size=axis_text_size,
            reverse_cmap=reverse_cmap,
            vlim=effect_dict[stat],
        )
        hax.set_title(stat.upper(), fontsize=title_text_size)

    # If there are less plots than cells in the grid, hide the remaining cells
    if (len(stats) % 3) != 0:
        for i in range(len(stats), nrow * 3):
            row = i // 3
            col = i % 3
            ax[row, col].set_visible(False)

    plt.tight_layout()


def make_scatterplot(
    df, val_col, pred_col, thresh, cycle_col="cv_cycle", group_col="method"
):
    """
    Create scatter plots for each method showing the relationship between predicted and measured values.

    Parameters:
    df (pd.DataFrame): Input dataframe containing the data.
    val_col (str): The column name for the ground truth values.
    pred_col (str): The column name for the predicted values.
    thresh (float): Threshold for binary classification.
    cycle_col (str): The column name indicating the cross-validation fold. Default is "cv_cycle".
    group_col (str): The column name indicating the groups/methods. Default is "method".

    Returns:
    None
    """
    df_split_metrics = calc_regression_metrics(
        df, cycle_col=cycle_col, val_col=val_col, pred_col=pred_col, thresh=thresh
    )
    methods = df[group_col].unique()

    fig, axs = plt.subplots(nrows=1, ncols=len(methods), figsize=(25, 10))

    for ax, method in zip(axs, methods):
        df_method = df.query(f"{group_col} == @method")
        df_metrics = df_split_metrics.query(f"{group_col} == @method")
        ax.scatter(df_method[pred_col], df_method[val_col], alpha=0.3)
        ax.plot(
            [df_method[val_col].min(), df_method[val_col].max()],
            [df_method[val_col].min(), df_method[val_col].max()],
            "k--",
            lw=1,
        )

        ax.axhline(y=thresh, color="r", linestyle="--")
        ax.axvline(x=thresh, color="r", linestyle="--")
        ax.set_title(method)

        y_true = df_method[val_col] > thresh
        y_pred = df_method[pred_col] > thresh
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)

        metrics_text = f"MAE: {df_metrics['mae'].mean():.2f}\nMSE: {df_metrics['mse'].mean():.2f}\nR2: {df_metrics['r2'].mean():.2f}\nrho: {df_metrics['rho'].mean():.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}"
        ax.text(
            0.05, 0.5, metrics_text, transform=ax.transAxes, verticalalignment="top"
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Measured")

    plt.tight_layout()
    plt.show()


def ci_plot(result_tab, ax_in, name):
    """
    Create a confidence interval plot for the given result table.

    Parameters:
    result_tab (pd.DataFrame): DataFrame containing the results with columns 'meandiff', 'lower', and 'upper'.
    ax_in (matplotlib.axes.Axes): The axes on which to plot the confidence intervals.
    name (str): The title of the plot.

    Returns:
    None
    """
    result_err = np.array(
        [
            result_tab["meandiff"] - result_tab["lower"],
            result_tab["upper"] - result_tab["meandiff"],
        ]
    )
    sns.set(rc={"figure.figsize": (6, 2)})
    sns.set_context("notebook")
    sns.set_style("whitegrid")
    ax = sns.pointplot(
        x=result_tab.meandiff, y=result_tab.index, marker="o", linestyle="", ax=ax_in
    )
    ax.errorbar(
        y=result_tab.index,
        x=result_tab["meandiff"],
        xerr=result_err,
        fmt="o",
        capsize=5,
    )
    ax.axvline(0, ls="--", lw=3)
    ax.set_xlabel("Mean Difference")
    ax.set_ylabel("")
    ax.set_title(name)
    ax.set_xlim(-0.2, 0.2)


def make_ci_plot_grid(df_in, metric_list, group_col="method"):
    """
    Create a grid of confidence interval plots for multiple metrics using Tukey HSD test results.

    Parameters:
    df_in (pd.DataFrame): Input dataframe containing the data.
    metric_list (list of str): List of metric column names to create confidence interval plots for.
    group_col (str): The column name indicating the groups. Default is "method".

    Returns:
    None
    """
    figure, axes = plt.subplots(
        len(metric_list), 1, figsize=(8, 2 * len(metric_list)), sharex=False
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for i, metric in enumerate(metric_list):
        df_tukey, _, _, _ = rm_tukey_hsd(df_in, metric, group_col=group_col)
        ci_plot(df_tukey, ax_in=axes[i], name=metric)
    figure.suptitle("Multiple Comparison of Means\nTukey HSD, FWER=0.05")
    plt.tight_layout()


def recall_at_precision(y_true, y_score, precision_threshold=0.5, direction="greater"):
    if direction not in ["greater", "lesser"]:
        raise ValueError("Invalid direction. Expected one of: ['greater', 'lesser']")

    y_true = np.array(y_true)
    y_score = np.array(y_score)
    thresholds = np.unique(y_score)
    thresholds = np.sort(thresholds)

    if direction == "greater":
        thresholds = np.sort(thresholds)
    else:
        thresholds = np.sort(thresholds)[::-1]

    for threshold in thresholds:
        if direction == "greater":
            y_pred = y_score >= threshold
        else:
            y_pred = y_score <= threshold

        precision = precision_score(y_true, y_pred)
        if precision >= precision_threshold:
            recall = recall_score(y_true, y_pred)
            return recall, threshold
    return np.nan, None


def calc_classification_metrics(df_in, cycle_col, val_col, prob_col, pred_col):
    metric_list = []
    for k, v in df_in.groupby([cycle_col, "method", "split"]):
        cycle, method, split = k
        roc_auc = roc_auc_score(v[val_col], v[prob_col])
        pr_auc = average_precision_score(v[val_col], v[prob_col])
        mcc = matthews_corrcoef(v[val_col], v[pred_col])

        recall, _ = recall_at_precision(
            v[val_col].astype(bool),
            v[prob_col],
            precision_threshold=0.8,
            direction="greater",
        )
        tnr, _ = recall_at_precision(
            ~v[val_col].astype(bool),
            v[prob_col],
            precision_threshold=0.8,
            direction="lesser",
        )

        metric_list.append([cycle, method, split, roc_auc, pr_auc, mcc, recall, tnr])

    metric_df = pd.DataFrame(
        metric_list,
        columns=[
            "cv_cycle",
            "method",
            "split",
            "roc_auc",
            "pr_auc",
            "mcc",
            "recall",
            "tnr",
        ],
    )
    return metric_df


def make_curve_plots(df):
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
        precision, recall, thresholds_pr = precision_recall_curve(
            v[val_col], v[prob_col]
        )

        _, threshold_recall_pos = recall_at_precision(
            v[val_col].astype(bool),
            v[prob_col],
            precision_threshold=0.8,
            direction="greater",
        )
        _, threshold_recall_neg = recall_at_precision(
            ~v[val_col].astype(bool),
            v[prob_col],
            precision_threshold=0.8,
            direction="lesser",
        )

        fpr_recall_pos = fpr[np.abs(thresholds_roc - threshold_recall_pos).argmin()]
        fpr_recall_neg = fpr[np.abs(thresholds_roc - threshold_recall_neg).argmin()]
        recall_recall_pos = recall[
            np.abs(thresholds_pr - threshold_recall_pos).argmin()
        ]
        recall_recall_neg = recall[
            np.abs(thresholds_pr - threshold_recall_neg).argmin()
        ]

        axes[0].plot(
            fpr,
            recall_pos,
            label=f"{k} (ROC AUC={roc_auc:.03f})",
            color=color,
            alpha=0.75,
        )
        axes[1].plot(
            recall,
            precision,
            label=f"{k} (PR AUC={pr_auc:.03f})",
            color=color,
            alpha=0.75,
        )

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
    plt.tight_layout()


###################################################
# Other utility functions
###################################################


def harmonize_columns(df):
    df = df.copy()
    # Map your columns to the function expectations
    rename_map = {
        "Model type": "method",  # or 'project_model' -> 'method' if you prefer that
        "Split": "split",
        "seed": "cv_cycle",
    }
    df.rename(
        columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True
    )
    # Ensure 'method','split','cv_cycle' exist
    assert {"method", "split", "cv_cycle"}.issubset(df.columns)
    return df


# STATISTICAL RELEVANCE ANALYSIS FUNCTIONS
def cliffs_delta(x, y):
    """
    Calculate Cliff's Delta effect size measure.

    Parameters:
    - x, y: array-like, the two samples to compare

    Returns:
    - delta: float, Cliff's Delta value
    - interpretation: str, interpretation of the effect size
    """
    x, y = np.array(x), np.array(y)
    m, n = len(x), len(y)

    # Calculate all pairwise comparisons
    comparisons = 0
    for xi in x:
        for yi in y:
            if xi > yi:
                comparisons += 1
            elif xi < yi:
                comparisons -= 1

    delta = comparisons / (m * n)

    # Interpret effect size
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


def wilcoxon_pairwise_test(
    df, metric, model_a, model_b, task=None, split=None, seed_col=None
):
    """
    Perform Wilcoxon signed-rank test between two models for a specific metric.

    Parameters:
    - df: DataFrame with performance data
    - metric: str, name of the metric column
    - model_a, model_b: str, model names to compare
    - task: str, optional task filter
    - split: str, optional split filter
    - seed_col: str, column name for seeds/runs

    Returns:
    - dict with test results
    """
    # Filter data
    data = df.copy()
    if task is not None:
        data = data[data["Task"] == task]
    if split is not None:
        data = data[data["Split"] == split]

    # Get metric values for each model
    values_a = data[data["Model type"] == model_a][metric].values
    values_b = data[data["Model type"] == model_b][metric].values

    if len(values_a) == 0 or len(values_b) == 0:
        return None

    # Ensure same length (paired test)
    min_len = min(len(values_a), len(values_b))
    values_a = values_a[:min_len]
    values_b = values_b[:min_len]

    # Wilcoxon signed-rank test
    statistic, p_value = wilcoxon(values_a, values_b, alternative="two-sided")

    # Cliff's Delta effect size
    delta, effect_size_interpretation = cliffs_delta(values_a, values_b)

    # Median difference and bootstrap CI
    differences = values_a - values_b
    median_diff = np.median(differences)
    ci_lower, ci_upper = bootstrap_ci(differences, np.median, n_bootstrap=1000)

    # Practical significance assessment
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
    Apply Holm-Bonferroni correction for multiple comparisons.

    Parameters:
    - p_values: list or array of p-values

    Returns:
    - corrected_p_values: array of corrected p-values
    - rejected: boolean array indicating which hypotheses are rejected
    """
    p_values = np.array(p_values)
    n = len(p_values)

    # Sort p-values and keep track of original indices
    sorted_indices = np.argsort(p_values)
    sorted_p_values = p_values[sorted_indices]

    # Apply Holm-Bonferroni correction
    corrected_p_values = np.zeros(n)
    rejected = np.zeros(n, dtype=bool)

    for i in range(n):
        # Correction factor decreases with each step
        correction_factor = n - i
        corrected_p_values[sorted_indices[i]] = min(
            1.0, sorted_p_values[i] * correction_factor
        )

        # Check if hypothesis is rejected (alpha = 0.05)
        if corrected_p_values[sorted_indices[i]] < 0.05:
            rejected[sorted_indices[i]] = True
        else:
            # Once we fail to reject, all subsequent hypotheses are also not rejected
            break

    return corrected_p_values, rejected


def pairwise_model_comparison(
    df, metrics, models=None, tasks=None, splits=None, alpha=0.05
):
    """
    Perform comprehensive pairwise model comparisons with statistical tests.

    Parameters:
    - df: DataFrame with performance data
    - metrics: list of metrics to compare
    - models: list of models to compare (if None, use all models)
    - tasks: list of tasks to include (if None, use all tasks)
    - splits: list of splits to include (if None, use all splits)
    - alpha: significance level

    Returns:
    - DataFrame with test results
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
                # All pairwise combinations
                for i, model_a in enumerate(models):
                    for j, model_b in enumerate(models):
                        if i < j:  # Avoid duplicate comparisons
                            result = wilcoxon_pairwise_test(
                                df, metric, model_a, model_b, task, split
                            )
                            if result is not None:
                                results.append(result)

    if not results:
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    # Apply Holm-Bonferroni correction
    p_values = results_df["p_value"].values
    corrected_p_values, rejected = holm_bonferroni_correction(p_values)

    results_df["corrected_p_value"] = corrected_p_values
    results_df["significant_after_correction"] = rejected

    return results_df


def friedman_nemenyi_test(df, metrics, models=None, alpha=0.05):
    """
    Perform Friedman test followed by Nemenyi post-hoc test for multiple model comparison.

    Parameters:
    - df: DataFrame with performance data
    - metrics: list of metrics to compare
    - models: list of models to compare
    - alpha: significance level

    Returns:
    - dict with Friedman test results and post-hoc comparisons
    """
    if models is None:
        models = df["Model type"].unique()

    results = {}

    for metric in metrics:
        # Prepare data for Friedman test
        # Each row should be a "block" (task/split combination)
        # Each column should be a model
        pivot_data = df.pivot_table(
            values=metric, index=["Task", "Split"], columns="Model type", aggfunc="mean"
        )

        # Filter to only include specified models
        available_models = [m for m in models if m in pivot_data.columns]
        pivot_data = pivot_data[available_models]

        # Remove rows with any NaN values
        pivot_data = pivot_data.dropna()

        if pivot_data.shape[0] < 2 or pivot_data.shape[1] < 3:
            results[metric] = {
                "error": "Insufficient data for Friedman test",
                "data_shape": pivot_data.shape,
            }
            continue

        # Friedman test
        try:
            friedman_stat, friedman_p = friedmanchisquare(
                *[pivot_data[col].values for col in pivot_data.columns]
            )

            # Calculate average ranks
            ranks = pivot_data.rank(
                axis=1, ascending=False
            )  # Higher values get better ranks
            mean_ranks = ranks.mean()

            result = {
                "friedman_statistic": friedman_stat,
                "friedman_p_value": friedman_p,
                "mean_ranks": mean_ranks.to_dict(),
                "significant": friedman_p < alpha,
            }

            # If Friedman test is significant, perform Nemenyi post-hoc test
            if friedman_p < alpha:
                try:
                    # Convert to format expected by scikit_posthocs
                    data_array = pivot_data.values
                    nemenyi_result = sp.posthoc_nemenyi_friedman(
                        data_array.T  # Transpose for correct format
                    )
                    nemenyi_result.index = available_models
                    nemenyi_result.columns = available_models

                    result["nemenyi_p_values"] = nemenyi_result.to_dict()
                    result["critical_difference"] = calculate_critical_difference(
                        len(available_models), pivot_data.shape[0], alpha
                    )
                except Exception as e:
                    result["nemenyi_error"] = str(e)

            results[metric] = result

        except Exception as e:
            results[metric] = {"error": str(e)}

    return results


def calculate_critical_difference(k, n, alpha=0.05):
    """
    Calculate critical difference for Nemenyi test.

    Parameters:
    - k: number of models
    - n: number of data sets (blocks)
    - alpha: significance level

    Returns:
    - critical difference value
    """
    from scipy.stats import studentized_range

    # Critical value from studentized range distribution
    q_alpha = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)

    # Critical difference
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))

    return cd


def bootstrap_auc_difference(
    auc_values_a, auc_values_b, n_bootstrap=1000, ci=95, random_state=42
):
    """
    Calculate bootstrap confidence intervals for AUC differences.

    Parameters:
    - auc_values_a, auc_values_b: arrays of AUC values for two methods
    - n_bootstrap: number of bootstrap samples
    - ci: confidence interval percentage
    - random_state: random seed

    Returns:
    - dict with difference statistics and CI
    """
    np.random.seed(random_state)

    differences = []
    for _ in range(n_bootstrap):
        # Bootstrap sample from each group
        sample_a = resample(auc_values_a, random_state=np.random.randint(0, 10000))
        sample_b = resample(auc_values_b, random_state=np.random.randint(0, 10000))

        # Calculate difference of means
        diff = np.mean(sample_a) - np.mean(sample_b)
        differences.append(diff)

    differences = np.array(differences)

    # Calculate confidence interval
    alpha = (100 - ci) / 2
    ci_lower = np.percentile(differences, alpha)
    ci_upper = np.percentile(differences, 100 - alpha)

    # Original difference
    original_diff = np.mean(auc_values_a) - np.mean(auc_values_b)

    return {
        "mean_difference": original_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "bootstrap_differences": differences,
    }


def comprehensive_statistical_analysis(
    df, metrics, models=None, tasks=None, splits=None, save_dir=None, alpha=0.05
):
    """
    Perform comprehensive statistical analysis including all requested tests.

    Parameters:
    - df: DataFrame with performance data
    - metrics: list of metrics to analyze
    - models: list of models to compare
    - tasks: list of tasks to include
    - splits: list of splits to include
    - save_dir: directory to save results
    - alpha: significance level

    Returns:
    - dict with all statistical test results
    """
    print("Performing comprehensive statistical analysis...")

    results = {}

    # 1. Pairwise Wilcoxon tests with Cliff's Delta
    print("1. Running pairwise Wilcoxon signed-rank tests...")
    pairwise_results = pairwise_model_comparison(
        df, metrics, models, tasks, splits, alpha
    )
    results["pairwise_tests"] = pairwise_results

    # 2. Friedman test with Nemenyi post-hoc
    print("2. Running Friedman tests with Nemenyi post-hoc...")
    friedman_results = friedman_nemenyi_test(df, metrics, models, alpha)
    results["friedman_nemenyi"] = friedman_results

    # 3. AUC bootstrap comparisons (if AUC columns exist)
    auc_columns = [col for col in df.columns if "AUC" in col or "auc" in col]
    if auc_columns:
        print("3. Running bootstrap comparisons for AUC metrics...")
        auc_bootstrap_results = {}

        for auc_col in auc_columns:
            auc_bootstrap_results[auc_col] = {}

            if models is None:
                available_models = df["Model type"].unique()
            else:
                available_models = models

            # Pairwise AUC comparisons
            for i, model_a in enumerate(available_models):
                for j, model_b in enumerate(available_models):
                    if i < j:
                        auc_a = df[df["Model type"] == model_a][auc_col].dropna().values
                        auc_b = df[df["Model type"] == model_b][auc_col].dropna().values

                        if len(auc_a) > 0 and len(auc_b) > 0:
                            bootstrap_result = bootstrap_auc_difference(auc_a, auc_b)
                            auc_bootstrap_results[auc_col][
                                f"{model_a}_vs_{model_b}"
                            ] = bootstrap_result

        results["auc_bootstrap"] = auc_bootstrap_results

    # Save results if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        # Save pairwise results
        if not pairwise_results.empty:
            pairwise_results.to_csv(
                os.path.join(save_dir, "pairwise_statistical_tests.csv"), index=False
            )

        # Save Friedman/Nemenyi results
        import json

        with open(os.path.join(save_dir, "friedman_nemenyi_results.json"), "w") as f:
            # Convert numpy types to regular Python types for JSON serialization
            json_compatible_results = {}
            for metric, result in friedman_results.items():
                json_compatible_results[metric] = {}
                for key, value in result.items():
                    if isinstance(value, (np.ndarray, np.generic)):
                        json_compatible_results[metric][key] = value.tolist()
                    elif isinstance(value, dict):
                        json_compatible_results[metric][key] = {
                            str(k): (
                                float(v)
                                if isinstance(v, (np.ndarray, np.generic))
                                else v
                            )
                            for k, v in value.items()
                        }
                    else:
                        json_compatible_results[metric][key] = (
                            float(value)
                            if isinstance(value, (np.ndarray, np.generic))
                            else value
                        )

            json.dump(json_compatible_results, f, indent=2)

        # Save AUC bootstrap results
        if auc_columns:
            with open(os.path.join(save_dir, "auc_bootstrap_results.json"), "w") as f:
                json_compatible_auc = {}
                for auc_col, comparisons in results["auc_bootstrap"].items():
                    json_compatible_auc[auc_col] = {}
                    for comparison, result in comparisons.items():
                        json_compatible_auc[auc_col][comparison] = {
                            k: v.tolist() if isinstance(v, np.ndarray) else v
                            for k, v in result.items()
                        }
                json.dump(json_compatible_auc, f, indent=2)

    return results


def plot_critical_difference_diagram(
    friedman_results, metric, save_dir=None, alpha=0.05
):
    """
    Plot Critical Difference diagram for Nemenyi test results.

    Parameters:
    - friedman_results: results from friedman_nemenyi_test
    - metric: metric name to plot
    - save_dir: directory to save plot
    - alpha: significance level
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

    # Sort by rank (lower rank = better performance)
    sorted_indices = np.argsort(ranks)
    sorted_models = [models[i] for i in sorted_indices]
    sorted_ranks = [ranks[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot ranks on x-axis
    y_pos = 0
    ax.scatter(sorted_ranks, [y_pos] * len(sorted_ranks), s=100, c="blue")

    # Add model labels
    for i, (model, rank) in enumerate(zip(sorted_models, sorted_ranks)):
        ax.annotate(
            model,
            (rank, y_pos),
            xytext=(0, 20),
            textcoords="offset points",
            ha="center",
            rotation=45,
        )

    # Add critical difference bar
    if "critical_difference" in result:
        cd = result["critical_difference"]

        # Find groups of models that are not significantly different
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

        # Draw lines connecting non-significantly different models
        colors = plt.cm.Set3(np.linspace(0, 1, len(groups)))
        for group, color in zip(groups, colors):
            if len(group) > 1:
                group_ranks = [sorted_ranks[sorted_models.index(m)] for m in group]
                min_rank, max_rank = min(group_ranks), max(group_ranks)
                ax.plot(
                    [min_rank, max_rank],
                    [y_pos - 0.05, y_pos - 0.05],
                    color=color,
                    linewidth=3,
                    alpha=0.7,
                )

    ax.set_xlim(min(sorted_ranks) - 0.5, max(sorted_ranks) + 0.5)
    ax.set_ylim(-0.3, 0.5)
    ax.set_xlabel("Average Rank")
    ax.set_title(f"Critical Difference Diagram - {metric}")
    ax.grid(True, alpha=0.3)

    # Remove y-axis
    ax.set_yticks([])

    if save_dir:
        plot_name = f"critical_difference_{metric.replace(' ', '_')}"
        save_plot(fig, save_dir, plot_name)

    plt.tight_layout()
    if INTERACTIVE_MODE:
        plt.show()
    plt.close()


def analyze_significance(df_raw, metrics, direction_dict, effect_dict):
    df = harmonize_columns(df_raw)
    for split in df["split"].unique():
        df_s = df[df["split"] == split].copy()
        print(f"\n=== Split: {split} ===")

        # 3.a) Normality diagnostics for all metrics (one go)
        make_normality_diagnostic(df_s, metrics)

        # 3.b) For each metric: parametric vs non-parametric branch
        for metric in metrics:
            print(f"\n-- Metric: {metric}")
            # quick normality probe on within-seed residuals
            # (same idea you used in earlier sketch)
            wide = df_s.pivot(index="seed", columns="method", values=metric)
            resid = (wide.T - wide.mean(axis=1)).T
            vals = resid.values.flatten()
            vals = vals[~np.isnan(vals)]
            from scipy.stats import shapiro

            W, p_norm = shapiro(vals) if len(vals) >= 3 else (None, 0.0)

            # 3.c) Parametric RM-ANOVA + Tukey path
            if (p_norm is not None) and (p_norm >= 0.05):
                # Boxplot w/ RM-ANOVA p-value in title
                make_boxplots_parametric(df_s, [metric])
                # Tukey-style matrix + MCS heatmap and CI plots
                tukey_tab, means, means_diff, pmat = rm_tukey_hsd(
                    df=df_s.rename(
                        columns={metric: metric.lower()}
                    ),  # rm_tukey_hsd expects lower-case metric key
                    metric=metric.lower(),
                    group_col="method",
                    alpha=0.05,
                    sort=True,
                    direction_dict={k.lower(): v for k, v in direction_dict.items()},
                )
                # Multiple-comparisons (MCS) heatmap
                make_mcs_plot_grid(
                    df=df_s.rename(columns={metric: metric.lower()}),
                    stats=[metric],
                    group_col="method",
                    alpha=0.05,
                    figsize=(6, 4),
                    direction_dict=direction_dict,
                    effect_dict=effect_dict,
                    show_diff=True,
                    sort_axes=True,
                )
                # Confidence-interval forest for pairwise diffs
                make_ci_plot_grid(
                    df_s.rename(columns={metric: metric.lower()}),
                    [metric.lower()],
                    group_col="method",
                )

            # 3.d) Non-parametric Friedman + Conover path
            else:
                make_boxplots_nonparametric(df_s, [metric])
                # Significance heatmap (Conover-Holm) & critical difference
                make_sign_plots_nonparametric(df_s, [metric])
                make_critical_difference_diagrams(df_s, [metric])


def generate_statistical_report(results, save_dir=None):
    """
    Generate a comprehensive statistical analysis report.

    Parameters:
    - results: output from comprehensive_statistical_analysis
    - save_dir: directory to save report
    """
    report = []

    report.append("=" * 80)
    report.append("COMPREHENSIVE STATISTICAL ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")

    # 1. Pairwise Tests Summary
    if "pairwise_tests" in results and not results["pairwise_tests"].empty:
        pairwise_df = results["pairwise_tests"]

        report.append("1. PAIRWISE MODEL COMPARISONS (Wilcoxon Signed-Rank Test)")
        report.append("-" * 60)

        # Significant differences after correction
        significant = pairwise_df[pairwise_df["significant_after_correction"] == True]

        report.append(f"Total pairwise comparisons performed: {len(pairwise_df)}")
        report.append(
            f"Significant differences (after Holm-Bonferroni correction): {len(significant)}"
        )
        report.append("")

        if len(significant) > 0:
            report.append("Significant differences found:")
            for _, row in significant.iterrows():
                effect_size = row["effect_size_interpretation"]
                report.append(
                    f"  • {row['model_a']} vs {row['model_b']} ({row['metric']}, {row['split']}):"
                )
                report.append(
                    f"    - p-value: {row['p_value']:.4f} (corrected: {row['corrected_p_value']:.4f})"
                )
                report.append(
                    f"    - Cliff's Δ: {row['cliffs_delta']:.3f} ({effect_size} effect)"
                )
                report.append(
                    f"    - Median difference: {row['median_difference']:.4f} [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]"
                )
                report.append(f"    - {row['practical_significance']}")
                report.append("")
        else:
            report.append(
                "No significant differences found after multiple comparison correction."
            )
            report.append("")

    # 2. Friedman/Nemenyi Tests Summary
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
                report.append(
                    "  Result: Significant difference between models detected"
                )

                # Show mean ranks
                mean_ranks = result["mean_ranks"]
                sorted_ranks = sorted(mean_ranks.items(), key=lambda x: x[1])
                report.append("  Model rankings (lower rank = better performance):")
                for i, (model, rank) in enumerate(sorted_ranks, 1):
                    report.append(f"    {i}. {model}: {rank:.2f}")

                # Critical difference
                if "critical_difference" in result:
                    report.append(
                        f"  Critical difference: {result['critical_difference']:.3f}"
                    )

            else:
                report.append("  Result: No significant difference between models")

            report.append("")

    # 3. AUC Bootstrap Results Summary
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

                # Practical significance
                if ci_lower <= 0 <= ci_upper:
                    significance = "difference is small (CI includes 0)"
                else:
                    significance = "difference may be meaningful"

                report.append(f"  {model_a} vs {model_b}:")
                report.append(
                    f"    Mean difference: {mean_diff:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]"
                )
                report.append(f"    {significance}")

            report.append("")

    # Save report
    report_text = "\n".join(report)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "statistical_analysis_report.txt"), "w") as f:
            f.write(report_text)

    # Print to console
    print(report_text)

    return report_text
