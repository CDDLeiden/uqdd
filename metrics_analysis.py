import pandas as pd
import matplotlib.pyplot as plt
import itertools
import os
import plotly.graph_objects as go
import numpy as np
from plotly.offline import init_notebook_mode, iplot

import shutil

import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.cm import ScalarMappable


# DESCRIPTORS
descriptor_protein='ankh-large'
descriptor_chemical='ecfp2048'
prot_input_dim=1536
chem_input_dim=2048

group_cols = ['Model type', 'Task', 'Activity', 'Split', 'desc_prot', 'desc_chem', 'dropout']
numeric_cols = ['RMSE', 'R2', 'MAE', 'MDAE', 'MARPD', 'PCC', 'RMS Calibration', 'MA Calibration',
                    'Miscalibration Area', 'Sharpness', 'NLL', 'CRPS', 'Check', 'Interval', 'rho_rank',
                    'rho_rank_sim', 'rho_rank_sim_std', 'uq_mis_cal', 'uq_NLL', 'uq_NLL_sim', 
                    'uq_NLL_sim_std', 'Z_var', 'Z_var_CI_low', 'Z_var_CI_high', 'Z_mean', 
                    'Z_mean_CI_low', 'Z_mean_CI_high', 'rmv_rmse_slope', 'rmv_rmse_r_sq','rmv_rmse_intercept', 'aleatoric_uct_mean', 'epistemic_uct_mean', 
                    'total_uct_mean']
string_cols = ['wandb project', 'wandb run', 'model name']
order_by = ['Split', 'Model type']


mc_group_order = [
    "stratified_dropout0.1",
    "stratified_dropout0.2",
    "scaffold_cluster_dropout0.1",
    "scaffold_cluster_dropout0.2",
    "time_dropout0.1",
    "time_dropout0.2"
]
mc_group_order_no_time = [
    "stratified_dropout0.1",
    "stratified_dropout0.2",
    "scaffold_cluster_dropout0.1",
    "scaffold_cluster_dropout0.2",
]

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
        # "time": "....",
        "time": "///",
    }

hatches_dict_no_time = {
        "stratified": "\\\\",
        "scaffold_cluster": "",
    }


# RESULTS AGGREGATION TO CREATE THE FINAL RESULTS TABLE
def aggregate_results_csv(df, group_cols, numeric_cols, string_cols, order_by=None, output_file_path=None):
    # Group the DataFrame by the specified columns
    grouped = df.groupby(group_cols)
    # Aggregate the numeric columns
    aggregated = grouped[numeric_cols].agg(['mean', 'std'])
    # Combine mean and std into the required format
    for col in numeric_cols:
        aggregated[(col, 'combined')] = aggregated[(col, 'mean')].round(3).astype(str) + '(' + aggregated[(col, 'std')].round(3).astype(str) + ')'
    # Drop the separate mean and std columns, keeping only the combined column
    aggregated = aggregated[[col for col in aggregated.columns if col[1] == 'combined']]

    # Rename the columns to a simpler format
    aggregated.columns = [col[0] for col in aggregated.columns]

    # Step 6: Aggregate the string columns into lists
    string_aggregated = grouped[string_cols].agg(lambda x: list(x))

    # Step 7: Create the new column combining wandb project and model name
    df['project_model'] = 'papyrus' + '/' + df['Activity'] + '/' + 'all' + '/' + df['wandb project'] + '/' + df['model name'] + '/'
    project_model_aggregated = grouped['project_model'].agg(lambda x: list(x))
    
    # Combine the numeric and string aggregations
    final_aggregated = pd.concat([aggregated, string_aggregated, project_model_aggregated], axis=1).reset_index()

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
def save_plot(fig, save_dir, plot_name, tighten=True):
    if tighten:
        plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f"{plot_name}.png"), dpi=1200)
        fig.savefig(os.path.join(save_dir, f"{plot_name}.svg"))
        fig.savefig(os.path.join(save_dir, f"{plot_name}.pdf"))
        fig.savefig(os.path.join(save_dir, f"{plot_name}.eps"))
        
# Function to handle inf values
def handle_inf_values(df):
    df = df.replace([float('inf'), -float('inf')], float('nan'))
    return df


# Pair plot for visualizing relationships
def plot_pairplot(df, title, metrics, save_dir=None, cmap="viridis", group_order=group_order):
    df = handle_inf_values(df)
    sns.pairplot(
        df, 
        hue='Group', 
        hue_order=group_order,
        # markers=['o', 's'],
        vars=metrics, 
        palette=cmap, 
        plot_kws={'alpha': 0.7}
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
        sns.lineplot(data=df, x='wandb run', y=metric, hue='Group', marker='o', palette="Set2", hue_order=group_order)
        plt.title(f'{title} - {metric}')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()

        plot_name = f"line_{title.replace(' ', '_')}_{metric}"
        save_plot(plt.gcf(), save_dir, plot_name)

# Function to plot histograms for metrics
def plot_histogram_metrics(df, title, metrics, save_dir=None, group_order=group_order, cmap="crest"):
    df = handle_inf_values(df)
    plt.figure(figsize=(14, 7))
    for metric in metrics:
        sns.histplot(data=df, x=metric, hue='Group', kde=True, palette=cmap, element="step", hue_order=group_order, fill=True, alpha=0.7)
        plt.title(f'{title} - {metric}')
        plt.show()

        plot_name = f"histogram_{title.replace(' ', '_')}_{metric}"
        save_plot(plt.gcf(), save_dir, plot_name)

# Function to plot pairwise scatter plots for metrics
def plot_pairwise_scatter_metrics(df, title, metrics, save_dir=None, group_order=group_order, cmap="crest_r"):
    df = handle_inf_values(df)
    num_metrics = len(metrics)
    fig, axes = plt.subplots(num_metrics, num_metrics, figsize=(15, 15))
    
    for i, j in itertools.product(range(num_metrics), range(num_metrics)):
        if i != j:  # Only plot the lower triangle
            ax = sns.scatterplot(data=df, x=metrics[j], y=metrics[i], hue='Group', palette=cmap, hue_order=group_order, ax=axes[i, j], legend=False if not (i == 1 and j == 0) else 'brief')
            if i == 1 and j == 0:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend().remove()
        else:
            axes[i, j].set_visible(False)  # Hide the diagonal and upper triangle subplots

        if j == 0 and i > 0:
            axes[i, j].set_ylabel(metrics[i])
        else:
            axes[i, j].set_ylabel('')
        
        if i == num_metrics - 1:
            axes[i, j].set_xlabel(metrics[j])
        else:
            axes[i, j].set_xlabel('')
    
    # Add a single legend
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.15, 1))
    fig.suptitle(title, y=1.02)
    fig.subplots_adjust(top=0.95, wspace=0.4, hspace=0.4)
    plot_name = f"pairwise_scatter_{title.replace(' ', '_')}"
    save_plot(fig, save_dir, plot_name)
    plt.show()


def plot_metrics(
        df, 
        metrics, 
        cmap="crest_r", 
        save_dir=None,
        hatches_dict=hatches_dict,
        group_order=group_order,
        show=True
):
    stats_dfs = []

    # Prepare data for each metric
    for metric in metrics:
        mean_df = (
            df.loc[:, ["Split", "Model type", metric]]
            .groupby(["Split", "Model type"])
            .mean()
            .rename(columns={metric: f"{metric}_mean"})
        )
        std_df = (
            df.loc[:, ["Split", "Model type", metric]]
            .groupby(["Split", "Model type"])
            .std()
            .rename(columns={metric: f"{metric}_std"})
        )
        stats_df = (
            pd.merge(mean_df, std_df, on=["Split", "Model type"])
            .sort_values(["Split", "Model type"])
            .reset_index()
            .assign(Group=lambda df: df.apply(lambda row: f"{row['Split']}_{row['Model type']}", axis=1))
        )
        stats_df['Metric'] = metric
        stats_dfs.append(stats_df)

    combined_stats_df = pd.concat(stats_dfs)
    # Ensure 'Group' column is categorical with the specified order
    if group_order:
        combined_stats_df['Group'] = pd.Categorical(combined_stats_df['Group'], categories=group_order, ordered=True)
    else:
        group_order = combined_stats_df['Group'].unique().tolist()
        
    scalar_mappable = ScalarMappable(cmap=cmap)
    model_types = combined_stats_df["Model type"].unique()
    color_dict = {m: c for m, c in zip(model_types, scalar_mappable.to_rgba(range(len(model_types)), alpha=1).tolist())}
    
    # Calculate appropriate figsize based on the number of metrics
    fig_width = max(10, len(metrics) * 2)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    # fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.12 # 0.12
    # bar_spacing = 0.4 # 0.01
    group_spacing = 0.4 # 0.7
    num_bars = len(model_types) * len(hatches_dict)  #  len(hatches_dict)  = number of splits
    positions = []
    tick_positions = []
    tick_labels = []

    for i, metric in enumerate(metrics):
        metric_data = combined_stats_df[combined_stats_df['Metric'] == metric]
        # Sort the data according to group_order
        metric_data['Group'] = pd.Categorical(metric_data['Group'], categories=group_order, ordered=True)
        metric_data = metric_data.sort_values('Group').reset_index(drop=True)
        for j, (_, row) in enumerate(metric_data.iterrows()):
            position = i * (num_bars * bar_width + group_spacing) + (j % num_bars) * bar_width
            # position = i * (len(model_types) * (bar_width + bar_spacing) + group_spacing) + j * (bar_width + bar_spacing)
            positions.append(position)
            ax.bar(
                position,
                height=row[f"{metric}_mean"],
                color=color_dict[row["Model type"]],
                hatch=hatches_dict[row["Split"]],
                width=bar_width,
            )
        center_position = i * (num_bars * bar_width + group_spacing) + (num_bars * bar_width) / 2
        # center_position = (positions[-1] + positions[-len(model_types)]) / 2
        tick_positions.append(center_position)
        tick_labels.append(metric)
        # tick_positions.append((positions[-1] + positions[-len(model_types)]) / 2)
        # tick_labels.append(metric)

    def create_stats_legend(df, color_mapping, hatches_dict, group_order):
        patches_dict = {}
        for idx, row in df.iterrows():
            label = f"{row['Split']} {row['Model type']}"
            group_label = f"{row['Split']}_{row['Model type']}"
            if group_label not in patches_dict:
                patches_dict[group_label] = mpatches.Patch(
                    facecolor=color_mapping[row["Model type"]],
                    hatch=hatches_dict[row["Split"]],
                    label=label
                )
        # Collect patches in order of group_order
        patches = [patches_dict[group] for group in group_order if group in patches_dict]
        return patches

        
        # def create_stats_legend(df, color_mapping, hatches_dict, group_order):
        #     patches = []
        #     # patches_dict = {}
        #     for idx, row in df.iterrows():
        #         label = f"{row['Split']} {row['Model type']}"
        #         # group_label = f"{row['Split']}_{row['Model type']}"
        #         if label not in [patch.get_label() for patch in patches]:
        #             patches.append(
        #                 mpatches.Patch(
        #                     facecolor=color_mapping[row["Model type"]], 
        #                     hatch=hatches_dict[row["Split"]], 
        #                     label=label
        #                 )
        #             )
        #     return patches

    legend_elements = create_stats_legend(combined_stats_df, color_dict, hatches_dict, group_order)

    ax.legend(
        handles=legend_elements,
        bbox_to_anchor=(1.0, 1.0),
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
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", rotation_mode="anchor", fontsize=9)
    ax.set_xlabel("Metrics")
    ax.set_ylabel("Values")
    metrics_names = "_".join(metrics)
    plot_name = f"barplot_{cmap}_{metrics_names}"
    save_plot(fig, save_dir, plot_name)
    if show:
        plt.show()
        plt.close()


def find_highly_correlated_metrics(df, metrics, threshold=0.8, save_dir=None, cmap="coolwarm"):
    # Calculate the correlation matrix
    corr_matrix = df[metrics].corr().abs()

    # Find pairs of metrics with correlation above the threshold
    highly_correlated_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] > threshold:
                pair = (corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])
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

def plot_comparison_metrics(df, metrics, cmap="crest_r", save_dir=None):  #, draw_points_on_error_bars=False
    stats_dfs = []

    # Prepare data for each metric
    for metric in metrics:
        mean_df = (
            df.loc[:, ["Split", "Model type", "Calibration", metric]]
            .groupby(["Split", "Model type", "Calibration"])
            .mean()
            .rename(columns={metric: f"{metric}_mean"})
        )
        std_df = (
            df.loc[:, ["Split", "Model type", "Calibration", metric]]
            .groupby(["Split", "Model type", "Calibration"])
            .std()
            .rename(columns={metric: f"{metric}_std"})
        )
        stats_df = (
            pd.merge(mean_df, std_df, on=["Split", "Model type", "Calibration"])
            .sort_values(["Split", "Model type", "Calibration"])
            .reset_index()
            .assign(Group=lambda df: df.apply(lambda row: f"{row['Split']}_{row['Model type']}_{row['Calibration']}", axis=1))
        )
        stats_df['Metric'] = metric
        stats_dfs.append(stats_df)

    combined_stats_df = pd.concat(stats_dfs)

    scalar_mappable = ScalarMappable(cmap=cmap)
    model_types = combined_stats_df["Model type"].unique()
    color_dict = {m: c for m, c in zip(model_types, scalar_mappable.to_rgba(range(len(model_types)), alpha=1).tolist())}

    hatches_dict = {
        "Before Calibration": "\\\\",
        "After Calibration": "",
    }
    
    # Calculate appropriate figsize based on the number of metrics
    fig_width = max(7, len(metrics) * 3)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    
    bar_width = 0.1
    group_spacing = 0.2  # Adjusted for closer split groups
    split_spacing = 0.6
    num_bars = len(model_types) * 2  # 2 calibration statuses (Before and After)
    positions = []
    tick_positions = []
    tick_labels = []

    for i, metric in enumerate(metrics):
        metric_data = combined_stats_df[combined_stats_df['Metric'] == metric]
        split_types = metric_data["Split"].unique()
        for j, split in enumerate(split_types):
            split_data = metric_data[metric_data["Split"] == split]
            for k, model_type in enumerate(model_types):
                for l, calibration in enumerate(["Before Calibration", "After Calibration"]):
                    position = (i * (split_spacing + len(split_types) * (num_bars * bar_width + group_spacing)) + 
                                j * (num_bars * bar_width + group_spacing) + 
                                k * 2 * bar_width + l * bar_width)
                    positions.append(position)
                    height = split_data[(split_data["Model type"] == model_type) & (split_data["Calibration"] == calibration)][f"{metric}_mean"].values[0]
                    ax.bar(
                        position,
                        height=height, 
                        color=color_dict[model_type],
                        hatch=hatches_dict[calibration],
                        width=bar_width,
                    )
                    # # Draw point on the error bar
                    # if draw_points_on_error_bars:
                    #     ax.plot(position, height, 'o', color='black')
            # Add tick positions and labels for each split within each metric
            center_position = (i * (split_spacing + len(split_types) * (num_bars * bar_width + group_spacing)) + 
                               j * (num_bars * bar_width + group_spacing) + 
                               (num_bars * bar_width) / 2)
            tick_positions.append(center_position)
            tick_labels.append(f"{metric}\n{split}")

    def create_stats_legend(color_dict, hatches_dict):
        patches = []
        for label, color in color_dict.items():
            patches.append(mpatches.Patch(facecolor=color, edgecolor='black', label=label))
        for label, hatch in hatches_dict.items():
            patches.append(mpatches.Patch(facecolor='white', edgecolor='black', hatch=hatch, label=label))
        return patches

    legend_elements = create_stats_legend(color_dict, hatches_dict)

    ax.legend(
        handles=legend_elements,
        bbox_to_anchor=(1.0, 1.0),
        loc="upper left",
        borderaxespad=0,
        frameon=False,
    )

    for (_, row), bar in zip(combined_stats_df.iterrows(), ax.patches):
        x_bar = bar.get_x() + bar.get_width() / 2
        y_bar = bar.get_height()
        # print(row[f"{row['Metric']}_std"])
        yerr_lower = y_bar - max(0, y_bar - row[f"{row['Metric']}_std"])
        yerr_upper = row[f"{row['Metric']}_std"]
        ax.errorbar(
            x_bar,
            y_bar,
            # yerr=row[f"{row['Metric']}_std"],
            yerr=[[yerr_lower], [yerr_upper]], 
            color="black",
            # fmt='o' if draw_points_on_error_bars else 'none',  # Option to draw points on error bars
            fmt="none",  # Option to draw points on error bars
            elinewidth=1,
            capsize=3,
            alpha=0.5,
            
        )

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", rotation_mode="anchor", fontsize=9)
    ax.set_xlabel("Metrics and Splits")
    ax.set_ylabel("Values")
    metrics_names = "_".join(metrics)
    plot_name = f"comparison_barplot_{cmap}_{metrics_names}"
    # plot_name += "_points" if draw_points_on_error_bars else ""
    save_plot(fig, save_dir, plot_name)
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
        file_path = os.path.join(base_path, path, 'calibration_plot_data.csv')
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)
            expected_values = data['Expected Proportion']
            # expected_values.append(data['Expected Proportion'])
            observed_values.append(data['Observed Proportion'])
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

def plot_calibration_data(df_aggregated, base_path, save_dir=None, title="Calibration Plot"):
    """
    Iterates over models in df_aggregated, loads and plots calibration data.
    """
    plt.figure(figsize=(12, 8))
    
    # color_map = plt.cm.get_cmap('tab10', len(df_aggregated))  # Generate unique colors

    for idx, row in df_aggregated.iterrows():
        model_paths = row['project_model']
        group_label = row['Group']
        # color = color_map(idx)  # Assign a color to each group

        # Load and aggregate calibration data
        expected, mean_observed, lower_bound, upper_bound = load_and_aggregate_calibration_data(base_path, model_paths)

        # Plot the mean line
        plt.plot(expected, mean_observed, label=group_label) # , color=color
        
        # Fill the shaded area
        plt.fill_between(expected, lower_bound, upper_bound,  alpha=0.2) # color=color,

    # Perfect calibration line
    plt.plot([0, 1], [0, 1], 'k--', label="Perfect Calibration")

    # Plot settings
    plt.title(title)
    plt.xlabel("Expected Proportion")
    plt.ylabel("Observed Proportion")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    if save_dir:
        plot_name = row['Activity'] + f"{title.replace(' ', '_')}.png"
        save_plot(plt.gcf(), save_dir, plot_name, tighten=True)
        
        # os.makedirs(save_dir, exist_ok=True)
        # plt.savefig(os.path.join(save_dir, f"{title.replace(' ', '_')}.png"), dpi=300, bbox_inches='tight')
    
    plt.show()


def move_model_folders(df, search_dirs, output_dir, overwrite=False):
    """
    Moves folders matching the unique 'model name' entries from df to output_dir.

    Parameters:
    df (pd.DataFrame): DataFrame containing a 'model name' column.
    search_dirs (list): List of directories to search for model folders.
    output_dir (str): Destination directory to move the model folders to.
    """
    # Get unique list of model names
    model_names = df['model name'].unique()

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
            subdirs = [d for d in os.listdir(search_dir)
                       if os.path.isdir(os.path.join(search_dir, d))]

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
            print(f"Model folder '{model_name}' not found in any of the search directories.")



def load_predictions(model_path):
    preds_path = os.path.join(model_path, 'preds.pkl')
    return pd.read_pickle(preds_path)
    
def calculate_rmse_rejection_curve(preds, uncertainty_col='y_alea', true_label_col='y_true', pred_label_col='y_pred', normalize_rmse=False, random_rejection=False, unc_type=None, max_rejection_ratio=0.95):
    # First we choose which type of uncertainty to use
    if unc_type == 'aleatoric':
        uncertainty_col = 'y_alea'
    elif unc_type == 'epistemic':
        uncertainty_col = 'y_eps'
    elif unc_type == 'both':
        preds['y_unc'] = preds['y_alea'] + preds['y_eps']
        uncertainty_col = 'y_unc'
    elif unc_type is None and uncertainty_col in preds.columns:
        pass
    else:
        raise ValueError(f"Either provide valid uncertainty type or provide the uncertainty column name in the DataFrame"
                         f"unc_type: {unc_type}, uncertainty_col: {uncertainty_col}")
    
    # Sort the DataFrame based on the uncertainty column or shuffle it randomly
    if random_rejection:
        preds = preds.sample(frac=max_rejection_ratio).reset_index(drop=True)  # Shuffle the DataFrame randomly
    else:
        preds = preds.sort_values(by=uncertainty_col, ascending=False)
    
    max_rejection_index = int(len(preds) * max_rejection_ratio)
    rejection_steps = np.arange(0, max_rejection_index, step=int(len(preds) * 0.01))
    rejection_rates = rejection_steps / len(preds)
    rmses = []
    
    initial_rmse = mean_squared_error(preds[true_label_col], preds[pred_label_col], squared=False)
    
    # RRC calculation
    for i in rejection_steps:
        selected_preds = preds.iloc[i:]
        rmse = mean_squared_error(selected_preds[true_label_col], selected_preds[pred_label_col], squared=False)
        if normalize_rmse:
            rmse /= initial_rmse
        rmses.append(rmse)
    # AUC calculation
    auc_arc = auc(rejection_rates, rmses)
    
    return rejection_rates, rmses, auc_arc


def plot_rmse_rejection_curves(df_pcm, base_dir, cmap="crest_r", save_dir_plot=None, add_to_title="", normalize_rmse=False, unc_type='aleatoric', max_rejection_ratio=0.95):
    assert unc_type in ['aleatoric', 'epistemic', 'both'], "unc_type should be either 'aleatoric' or 'epistemic' or 'both'"
    unc_col = 'y_alea' if unc_type == 'aleatoric' else 'y_eps'
    stats_dfs = []
    model_types = ["ensemble", "eoe", "evidential", "emc", "mcdropout"]
    splits = df_pcm["Split"].unique()
    # print(splits)
    df_pcm['model_path'] = df_pcm['project_model'].apply(
        lambda x: str(os.path.join(base_dir, x)) if not str(x).startswith(base_dir) else x
    )
    
    scalar_mappable = ScalarMappable(cmap=cmap)
    color_dict = {m: c for m, c in zip(model_types, scalar_mappable.to_rgba(range(len(model_types)), alpha=1).tolist())}

    fig, ax = plt.subplots(figsize=(12, 8))

    for model_type in model_types:
        for split in splits:
            model_names = df_pcm[(df_pcm["Model type"] == model_type) & (df_pcm["Split"] == split)]["model name"].unique()
            model_paths = df_pcm[(df_pcm["Model type"] == model_type) & (df_pcm["Split"] == split)]["model_path"].unique()
            aggregated_rmses = []
            auc_values = []
            # print(len(model_names))
            # print(len(model_paths))
            for model_name, model_path in zip(model_names, model_paths):
                # print(df_pcm[df_pcm['model name'] == model_name]['project_model']
                # model_path = os.path.join(base_dir, df_pcm[df_pcm['model name'] == model_name]['project_model'].iloc[0])
                # print(df_pcm[df_pcm['model name'] == model_name])
                # print(model_name)
                # print(model_path)
                preds = load_predictions(model_path)
                # print(preds.shape)
                rejection_rates, rmses, auc_arc = calculate_rmse_rejection_curve(preds, uncertainty_col=unc_col, normalize_rmse=normalize_rmse, max_rejection_ratio=max_rejection_ratio)
                
                
                aggregated_rmses.append(rmses)
                auc_values.append(auc_arc)

            # Average RMSE values across models
            # print(aggregated_rmses)
            mean_rmses = np.mean(aggregated_rmses, axis=0)
            std_rmses = np.std(aggregated_rmses, axis=0)
            
            mean_auc = np.mean(auc_values)
            std_auc = np.std(auc_values)

            # print(rejection_rates.shape)
            # print(rejection_rates)
            # print(mean_rmses.shape)
            # print(mean_rmses)
            
            # Plot the aggregated RMSE-Rejection curve            
            ax.plot(rejection_rates, mean_rmses, label=f"{model_type}-{split} (AUC-RRC:{mean_auc:.3f} ($\sigma${std_auc:.3f}))", color=color_dict[model_type])
            ax.fill_between(rejection_rates, mean_rmses - std_rmses, mean_rmses + std_rmses, color=color_dict[model_type], alpha=0.2)
            
            # Store aggregated AUC values
            stats_dfs.append({'Model type': model_type, 'Split': split, 'AUC-RRC_mean': mean_auc, 'AUC-RRC_std': std_auc})
    
    # Plot the baseline random rejection curve
    for split in splits:
        model_names = df_pcm[df_pcm["Split"] == split]["model name"].unique()
        model_paths = df_pcm[df_pcm["Split"] == split]["model_path"].unique()
        aggregated_rmses_random = []
        auc_values_random = []
        # for model_name in df_pcm["model name"].unique():
        for model_name, model_path in zip(model_names, model_paths):
            # model_path = os.path.join(base_dir, df_pcm[df_pcm['model name'] == model_name]['project_model'].iloc[0])
            preds = load_predictions(model_path)
            # print(preds.shape)
            rejection_rates, rmses_random, auc_rrc_random = calculate_rmse_rejection_curve(preds, uncertainty_col=unc_col, random_rejection=True, normalize_rmse=normalize_rmse, max_rejection_ratio=max_rejection_ratio)
            
            aggregated_rmses_random.append(rmses_random)
            auc_values_random.append(auc_rrc_random)
            
        mean_rmses_random = np.mean(aggregated_rmses_random, axis=0)
        std_rmses_random = np.std(aggregated_rmses_random, axis=0)
        
        mean_auc_random = np.mean(auc_values_random)
        std_auc_random = np.std(auc_values_random)
        
        # print(rejection_rates.shape)
        # print(rejection_rates)
        # print(mean_rmses_random.shape)
        # print(mean_rmses_random)
        
        ax.plot(rejection_rates, mean_rmses_random, label=f"random-reject-{split} (AUC-RRC:{mean_auc_random:.3f} ($\sigma${std_auc_random:.3f}))", color='black', linestyle='--')
        ax.fill_between(rejection_rates, mean_rmses_random - std_rmses_random, mean_rmses_random + std_rmses_random, color='grey', alpha=0.2)
        
        stats_dfs.append({'Model type': 'random', 'Split': split, 'AUC-RRC_mean': mean_auc_random, 'AUC-RRC_std': std_auc_random})
        
    ax.set_xlabel("Rejection Rate")
    ax.set_ylabel("RMSE" if not normalize_rmse else "Normalized RMSE")
    ax.set_title("RMSE-Rejection Curves" if not normalize_rmse else "Normalized RMSE-Rejection Curves")
    ax.set_ylim(0.75,1.05) if normalize_rmse else None
    ax.set_xlim(0, max_rejection_ratio) # Rejection rate from 0 to 1
    ax.set_xticks(np.append(np.arange(0, max_rejection_ratio+0.05, 0.1), max_rejection_ratio))
    # ax.set_yticks(np.arange(0.75, 1.05, 0.05))
    # Custom legend order
    # handles, labels = ax.get_legend_handles_labels()
    # stratified_handles = [h for h, l in zip(handles, labels) if 'stratified' in l]
    # scaffold_handles = [h for h, l in zip(handles, labels) if 'scaffold_cluster' in l]
    # ordered_handles = [item for pair in zip(stratified_handles, scaffold_handles) for item in pair]
    # ordered_handles += [h for h, l in zip(handles, labels) if 'Random' in l]  # Add random baseline last
    #     
    # ax.legend(
    #     ordered_handles,
    #     bbox_to_anchor=(0, 0),  # Place the legend inside the plot area, bottom-left corner
    #     loc='lower left',       # Align the legend to the lower left
    #     borderaxespad=0,        # No padding between legend and axes
    #     frameon=False,          # No frame around the legend
    #     fontsize='small',       # Adjust font size
    #     # ncol=2                  # Number of columns in the legend
    # )
    # Custom legend order
    handles, labels = ax.get_legend_handles_labels()
    ordered_labels = [f"{model_type}-" for model_type in model_types+["random"]]
    ordered_handles = [handles[labels.index(label)] for label in labels if any(label.startswith(ol) for ol in ordered_labels)]
    ax.legend(handles=ordered_handles, loc='lower left')
    
    plot_name = "rmse_rejection_curve"
    plot_name += f"_{add_to_title}" if add_to_title else ""
    save_plot(fig, save_dir_plot, plot_name, tighten=True)
    
    plt.show()

    return pd.DataFrame(stats_dfs)


def plot_auc_comparison(stats_df, cmap="crest_r", save_dir=None, add_to_title="", min_y_axis=0.5):
    model_types = ["ensemble", "eoe", "evidential", "emc", "mcdropout", "random"]  # Ordered model types including Random
    # model_types = ["ensemble", "evidential", "mcdropout", "random"]  # Ordered model types including Random

    splits = stats_df["Split"].unique()
    
    scalar_mappable = ScalarMappable(cmap=cmap)
    color_dict = {m: c for m, c in zip(model_types, scalar_mappable.to_rgba(range(len(model_types)), alpha=1).tolist())}
    color_dict["random"] = 'black'  # Color for Random baseline

    fig, ax = plt.subplots(figsize=(12, 8))
    bar_width = 0.35
    group_spacing = 0.6
    num_bars = len(model_types)
    positions = []
    tick_positions = []
    tick_labels = []

    for i, split in enumerate(splits):
        split_data = stats_df[stats_df['Split'] == split]
        for j, model_type in enumerate(model_types):
            model_data = split_data[split_data['Model type'] == model_type]
            position = i * (num_bars * bar_width + group_spacing) + j * bar_width
            positions.append(position)
            height = model_data['AUC-RRC_mean'].values[0]
            yerr = model_data['AUC-RRC_std'].values[0]
            ax.bar(
                position,
                height=height,
                yerr=yerr,
                color=color_dict[model_type],
                width=bar_width,
                label=model_type if i == 0 else ""
            )
            # Add tick positions and labels
            if j == len(model_types) - 1:
                center_position = (i * (num_bars * bar_width + group_spacing)) + (num_bars * bar_width - bar_width / 2) / 2
                tick_positions.append(center_position)
                tick_labels.append(f"{split}")

    def create_stats_legend(color_dict):
        patches = []
        for label, color in color_dict.items():
            patches.append(mpatches.Patch(facecolor=color, edgecolor='black', label=label))
        return patches

    legend_elements = create_stats_legend(color_dict)

    ax.legend(handles=legend_elements, bbox_to_anchor=(1.0, 1.0), loc="upper left", borderaxespad=0, frameon=False)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", rotation_mode="anchor", fontsize=9)
    ax.set_xlabel("Splits")
    ax.set_ylabel("AUC-RRC")
    ax.set_ylim(min_y_axis, 1.0)
    
    plot_name = f"auc_comparison_barplot_{cmap}"
    plot_name += f"_{add_to_title}" if add_to_title else ""
    save_plot(fig, save_dir, plot_name, tighten=True)
    plt.show()
    plt.close()

# def plot_auc_comparison(stats_df, cmap="crest_r", save_dir=None, add_to_title=""):
#     # model_types = stats_df["Model type"].unique()
#     model_types = ["ensemble", "evidential", "mcdropout"]  # Ordered model types
# 
#     splits = stats_df["Split"].unique()
# 
#     scalar_mappable = ScalarMappable(cmap=cmap)
#     color_dict = {m: c for m, c in zip(model_types, scalar_mappable.to_rgba(range(len(model_types)), alpha=1).tolist())}
# 
#     fig, ax = plt.subplots(figsize=(12, 8))
#     bar_width = 0.35
#     group_spacing = 0.6
#     num_bars = len(model_types)
#     positions = []
#     tick_positions = []
#     tick_labels = []
# 
#     for i, split in enumerate(splits):
#         split_data = stats_df[stats_df['Split'] == split]
#         for j, model_type in enumerate(model_types):
#             model_data = split_data[split_data['Model type'] == model_type]
#             position = i * (num_bars * bar_width + group_spacing) + j * bar_width
#             positions.append(position)
#             height = model_data['AUC-RRC_mean'].values[0]
#             yerr = model_data['AUC-RRC_std'].values[0]
#             ax.bar(
#                 position,
#                 height=height,
#                 yerr=yerr,
#                 color=color_dict[model_type],
#                 width=bar_width,
#                 label=model_type if i == 0 else ""
#             )
#             # Add tick positions and labels
#             if j == len(model_types) - 1:
#                 # center_position = (i * (num_bars * bar_width + group_spacing)) + (num_bars * bar_width) / 2
#                 center_position = (i * (num_bars * bar_width + group_spacing)) + (num_bars * bar_width - bar_width / 2) / 2
#                 tick_positions.append(center_position)
#                 tick_labels.append(f"{split}")
# 
#     def create_stats_legend(color_dict):
#         patches = []
#         for label, color in color_dict.items():
#             patches.append(mpatches.Patch(facecolor=color, edgecolor='black', label=label))
#         return patches
# 
#     legend_elements = create_stats_legend(color_dict)
# 
#     ax.legend(handles=legend_elements, bbox_to_anchor=(1.0, 1.0), loc="upper left", borderaxespad=0, frameon=False)
#     ax.set_xticks(tick_positions)
#     ax.set_xticklabels(tick_labels, rotation=45, ha="right", rotation_mode="anchor", fontsize=9)
#     ax.set_xlabel("Splits")
#     ax.set_ylabel("AUC-RRC")
#     plot_name = f"auc_comparison_barplot_{cmap}"
#     plot_name += f"_{add_to_title}" if add_to_title else ""
#     save_plot(fig, save_dir, plot_name, tighten=True)
#     plt.show()
#     plt.close()
