import argparse
import os
# In all plotting functions, only call plt.show() if running interactively
import sys
import pandas as pd

from uqdd.metrics import (
    group_cols,
    numeric_cols,
    string_cols,
    order_by,
    group_order_no_time,
    hatches_dict_no_time,
    accmetrics,
    accmetrics2,
    uctmetrics,
    aggregate_results_csv,
    plot_pairplot,
    plot_metrics,
    find_highly_correlated_metrics,
    plot_comparison_metrics,
    plot_calibration_data,
    plot_rmse_rejection_curves,
    plot_auc_comparison,
    save_stats_df,
)

INTERACTIVE_MODE = hasattr(sys, "ps1") or sys.flags.interactive

# lets run this file
if __name__ == "__main__":
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
        help="The name of the project.",
    )

    parser.add_argument(
        "--color", type=str, default="tab10_r", help="name of the color map"
    )
    parser.add_argument(
        "--color_2", type=str, default=None, help="name of the color map"
    )
    parser.add_argument(
        "--corr_color", type=str, default="YlGnBu", help="name of the color map"
    )
    parser.add_argument(
        "--show_legend",
        action="store_true",
        help="Whether to show the legend on the plot",
    )
    args = parser.parse_args()

    data_name = "papyrus"
    type_n_targets = "all"
    activity_type = args.activity_type
    project_name = args.project_name
    show_legend = args.show_legend
    color_map = args.color
    color_map_2 = args.color_2
    color_map_2 = color_map if color_map_2 is None else color_map_2
    corr_cmap = args.corr_color

    project_out_name = project_name

    data_specific_path = f"{data_name}/{activity_type}/{type_n_targets}"

    file_1 = f"/users/home/bkhalil/Repos/uqdd/uqdd/figures/papyrus/{activity_type}/all/reassess-runs_ensemble_mcdp_{activity_type}/metrics.csv"
    file_2 = f"/users/home/bkhalil/Repos/uqdd/uqdd/figures/papyrus/{activity_type}/all/reassess-runs_evidential_{activity_type}/metrics.csv"
    file_3 = f"/users/home/bkhalil/Repos/uqdd/uqdd/figures/papyrus/{activity_type}/all/reassess-runs_pnn_{activity_type}/metrics.csv"

    save_dir = f"/users/home/bkhalil/Repos/uqdd/uqdd/figures/{data_specific_path}/{project_out_name}/{color_map}/"
    save_dir_no_time = f"/users/home/bkhalil/Repos/uqdd/uqdd/figures/{data_specific_path}/{project_out_name}-no-time/{color_map}/"
    base_path = "/users/home/bkhalil/Repos/uqdd/uqdd/figures/"

    df_1 = pd.read_csv(file_1, header=0)
    df_2 = pd.read_csv(file_2, header=0)
    df_3 = pd.read_csv(file_3, header=0)
    df_main = pd.concat([df_1, df_2, df_3])
    num_duplicates = df_main.duplicated(subset=["wandb run", "Task"]).sum()
    if num_duplicates > 0:
        print(f"Found {num_duplicates} duplicate entries based on 'wandb run' column.")
    else:
        print("No duplicate entries found based on 'wandb run' column.")

    print(f"Dataframe shape before removing duplicates: {df_main.shape}")
    df_main = df_main.drop_duplicates(subset=["wandb run", "Task"], keep="first")
    print(f"Dataframe shape after removing duplicates: {df_main.shape}")

    df_main["Split"] = df_main["Split"].apply(
        lambda x: "stratified" if x == "random" else x
    )

    df_merged = df_main.copy()

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

    df_pcm = df_merged[df_merged["Task"] == "PCM"].copy()

    df_before_calib = df_merged[df_merged["Task"] == "PCM_before_calibration"].copy()
    df_before_calib["Calibration"] = "Before Calibration"

    df_after_calib = df_merged[
        df_merged["Task"] == "PCM_after_calibration_with_isotonic_regression"
    ].copy()
    df_after_calib["Calibration"] = "After Calibration"

    df_calib = pd.concat([df_before_calib, df_after_calib])
    df_calib_no_time = df_calib.copy()[df_calib["Split"] != "time"]

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
        df_pcm,
        accmetrics,
        threshold=0.9,
        save_dir=save_dir,
        cmap=corr_cmap,
        show_legend=show_legend,
    )
    highly_correlated_metrics_no_time = find_highly_correlated_metrics(
        df_no_time,
        accmetrics,
        threshold=0.9,
        save_dir=save_dir_no_time,
        cmap=corr_cmap,
        show_legend=show_legend,
    )
    highly_correlated_uctmetrics = find_highly_correlated_metrics(
        df_pcm,
        uctmetrics,
        threshold=0.9,
        save_dir=save_dir,
        cmap=corr_cmap,
        show_legend=show_legend,
    )
    highly_correlated_uctmetrics_no_time = find_highly_correlated_metrics(
        df_no_time,
        uctmetrics,
        threshold=0.9,
        save_dir=save_dir_no_time,
        cmap=corr_cmap,
        show_legend=show_legend,
    )
    uctmetrics_uncorr = [
        "Miscalibration Area",
        "Sharpness",
        "CRPS",
        "NLL",
        "Interval",
    ]

    find_highly_correlated_metrics(
        df_pcm,
        uctmetrics_uncorr,
        threshold=0.9,
        save_dir=save_dir,
        cmap=corr_cmap,
        show_legend=show_legend,
    )
    find_highly_correlated_metrics(
        df_no_time,
        uctmetrics_uncorr,
        threshold=0.9,
        save_dir=save_dir_no_time,
        cmap=corr_cmap,
        show_legend=show_legend,
    )

    color_dict = plot_metrics(
        df_no_time,
        accmetrics,
        cmap=color_map,
        save_dir=save_dir_no_time,
        hatches_dict=hatches_dict_no_time,
        group_order=group_order_no_time,
        fig_width=12,
        fig_height=3,
        show_legend=show_legend,
    )

    plot_metrics(
        df_no_time,
        accmetrics2,
        cmap=color_map,
        save_dir=save_dir_no_time,
        hatches_dict=hatches_dict_no_time,
        group_order=group_order_no_time,
        fig_width=6,
        fig_height=3,
        show_legend=show_legend,
    )

    plot_metrics(
        df_no_time,
        uctmetrics_uncorr,
        cmap=color_map,
        save_dir=save_dir_no_time,
        hatches_dict=hatches_dict_no_time,
        group_order=group_order_no_time,
        fig_width=10,
        fig_height=3,
        show_legend=show_legend,
    )

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
        fig_width=6,
        fig_height=3,
        show_legend=show_legend,
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
        fig_width=4,
        fig_height=3,
        show_legend=show_legend,
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
        fig_width=6,
        fig_height=3,
        show_legend=show_legend,
    )

    for m in accmetrics2:
        plot_metrics(
            df_no_time,
            [m],
            cmap=color_map,
            save_dir=save_dir_no_time,
            hatches_dict=hatches_dict_no_time,
            group_order=group_order_no_time,
            fig_width=2,
            fig_height=3,
            show_legend=show_legend,
        )

    for m in uctmetrics_uncorr:
        plot_metrics(
            df_no_time,
            [m],
            cmap=color_map,
            save_dir=save_dir_no_time,
            hatches_dict=hatches_dict_no_time,
            group_order=group_order_no_time,
            fig_width=2,
            fig_height=3,
            show_legend=show_legend,
        )

    models_order = ["pnn", "ensemble", "mcdropout", "evidential", "eoe", "emc"]
    plot_comparison_metrics(
        df_calib_no_time,
        uctmetrics_uncorr,
        cmap=color_map,
        color_dict=color_dict,
        save_dir=save_dir_no_time,
        fig_width=20,
        fig_height=3,
        show_legend=show_legend,
        models_order=models_order,
    )

    mc_list = ["RMS Calibration", "MA Calibration", "Miscalibration Area"]
    plot_comparison_metrics(
        df_calib_no_time,
        mc_list,
        cmap=color_map,
        color_dict=color_dict,
        save_dir=save_dir_no_time,
        fig_width=12,
        fig_height=3,
        show_legend=show_legend,
        models_order=models_order,
    )

    for mc in mc_list:
        plot_comparison_metrics(
            df_calib_no_time,
            [mc],
            cmap=color_map,
            color_dict=color_dict,
            save_dir=save_dir_no_time,
            fig_width=4,
            fig_height=3,
            show_legend=show_legend,
            models_order=models_order,
        )

    if color_map == color_map_2:
        color_dict_2 = color_dict
        splits = final_aggregated_no_time["Split"].unique()
        color_dict_2 = {
            f"{split}_{model}": color_dict[model]
            for split in splits
            for model in color_dict.keys()
        }
    else:
        color_dict_2 = None

    plot_calibration_data(
        final_aggregated_no_time,
        base_path,
        save_dir_no_time,
        title="Calibration Curves for Models",
        color_name=color_map_2,
        group_order=group_order_no_time,
        fig_width=5,
        fig_height=5,
        show_legend=show_legend,
    )

    dfs = [final_aggregated_no_time.iloc[[i]] for i in range(len(final_aggregated_no_time))]
    for i, df in enumerate(dfs):
        plot_calibration_data(
            df,
            base_path,
            save_dir_no_time,
            title=f"Calibration Curves for {df['Group'].values[0]}",
            color_name=color_map_2,
            color_dict=color_dict_2,
            group_order=group_order_no_time,
            fig_width=5,
            fig_height=5,
            show_legend=show_legend,
        )

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
            color_name=color_map_2,
            color_dict=color_dict_2,
            group_order=group_order_no_time,
            fig_width=5,
            fig_height=5,
            show_legend=show_legend,
        )

    df_pcm_stratified = df_pcm[df_pcm["Split"] == "stratified"]
    df_pcm_scaffold = df_pcm[df_pcm["Split"] == "scaffold_cluster"]
    df_pcm_time = df_pcm[df_pcm["Split"] == "time"]

    save_dir_plot = os.path.join(save_dir_no_time, "rrcs")
    os.makedirs(save_dir_plot, exist_ok=True)

    uct_types = ["aleatoric", "epistemic", "both"]
    for uct_t in uct_types:
        for normalize_rmse in [True, False]:
            add_to_title = "-normalized" if normalize_rmse else ""
            add_to_title += "-" + uct_t
            stats_df = plot_rmse_rejection_curves(
                df_no_time,
                base_path,
                cmap=color_map_2,
                save_dir_plot=save_dir_plot,
                add_to_title="all" + add_to_title,
                normalize_rmse=normalize_rmse,
                unc_type=uct_t,
                max_rejection_ratio=0.95,
                group_order=group_order_no_time,
                fig_width=6,
                fig_height=5,
                show_legend=show_legend,
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
                show_legend=show_legend,
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
                show_legend=show_legend,
            )

            save_stats_df(stats_df, save_dir_plot, add_to_title="all" + add_to_title)

            for name, df in zip(["stratified", "scaffold"], [df_pcm_stratified, df_pcm_scaffold]):
                stats_df = plot_rmse_rejection_curves(
                    df,
                    base_path,
                    cmap=color_map_2,
                    color_dict=color_dict_2,
                    save_dir_plot=save_dir_plot,
                    add_to_title=name + add_to_title,
                    normalize_rmse=normalize_rmse,
                    unc_type=uct_t,
                    max_rejection_ratio=0.95,
                    group_order=group_order_no_time,
                    fig_width=6,
                    fig_height=5,
                    show_legend=show_legend,
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
                    show_legend=show_legend,
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
                    show_legend=show_legend,
                )

                save_stats_df(stats_df, save_dir_plot, add_to_title=name + add_to_title)

    plot_pairplot(
        df_no_time,
        "Pairplot for Accuracy Metrics",
        accmetrics,
        save_dir=save_dir_no_time,
        cmap=color_map_2,
        group_order=group_order_no_time,
        show_legend=show_legend,
    )
    plot_pairplot(
        df_no_time,
        "Pairplot for Uncertainty Metrics",
        uctmetrics,
        save_dir=save_dir_no_time,
        cmap=color_map_2,
        group_order=group_order_no_time,
        show_legend=show_legend,
    )

    # Optional: Statistical analysis placeholders can be implemented in future modules
