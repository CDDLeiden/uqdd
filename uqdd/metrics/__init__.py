"""Metrics subpackage for UQDD

The ``uqdd.metrics`` subpackage provides tools to compute, analyze, and
visualize performance and uncertainty metrics for UQDD models. It includes
plotting and analysis helpers, statistical testing routines, and reassessment
utilities to benchmark and compare models rigorously.

Modules
-------
- ``constants``: Canonical metric names, grouping orders, hatches, and helper
    structures to standardize plots and reports.
- ``analysis``: Functions for aggregating results, loading predictions,
    computing rejection curves, and producing comparison plots and calibration
    visualizations.
- ``stats``: Statistical metrics and tests (regression/classification metrics,
    bootstrapping, Wilcoxon, Holm–Bonferroni, Friedman–Nemenyi, Cliff's delta),
    along with boxplots, curve plots, and significance analysis/reporting.
- ``reassessment``: Utilities to reassess runs and models (e.g., NLL for
    evidential models), export predictions, and post-process metrics from CSV.

Public API
----------
Commonly used names are re-exported for convenient access via
``uqdd.metrics.<name>``. They are grouped by module below for discoverability.

- Constants
    ``group_cols``, ``numeric_cols``, ``string_cols``, ``order_by``,
    ``group_order``, ``group_order_no_time``, ``hatches_dict``,
    ``hatches_dict_no_time``, ``accmetrics``, ``accmetrics2``, ``uctmetrics``,
    ``uctmetrics2``

- Analysis
    ``aggregate_results_csv``, ``save_plot``, ``handle_inf_values``,
    ``plot_pairplot``, ``plot_line_metrics``, ``plot_histogram_metrics``,
    ``plot_pairwise_scatter_metrics``, ``plot_metrics``,
    ``find_highly_correlated_metrics``, ``plot_comparison_metrics``,
    ``load_and_aggregate_calibration_data``, ``plot_calibration_data``,
    ``move_model_folders``, ``load_predictions``,
    ``calculate_rmse_rejection_curve``, ``calculate_rejection_curve``,
    ``get_handles_labels``, ``plot_rmse_rejection_curves``,
    ``plot_auc_comparison``, ``save_stats_df``, ``load_stats_df``

- Statistics
    ``calc_regression_metrics``, ``bootstrap_ci``, ``rm_tukey_hsd``,
    ``make_boxplots``, ``make_boxplots_parametric``,
    ``make_boxplots_nonparametric``, ``make_sign_plots_nonparametric``,
    ``make_critical_difference_diagrams``, ``make_normality_diagnostic``,
    ``mcs_plot``, ``make_mcs_plot_grid``, ``make_scatterplot``, ``ci_plot``,
    ``make_ci_plot_grid``, ``recall_at_precision``,
    ``calc_classification_metrics``, ``make_curve_plots``,
    ``harmonize_columns``, ``cliffs_delta``, ``wilcoxon_pairwise_test``,
    ``holm_bonferroni_correction``, ``pairwise_model_comparison``,
    ``friedman_nemenyi_test``, ``calculate_critical_difference``,
    ``bootstrap_auc_difference``, ``plot_critical_difference_diagram``,
    ``analyze_significance``, ``comprehensive_statistical_analysis``,
    ``generate_statistical_report``

- Reassessment
    ``nll_evidentials``, ``convert_to_list``, ``preprocess_runs``,
    ``get_model_class``, ``get_predict_fn``, ``get_preds``, ``pkl_preds_export``,
    ``csv_nll_post_processing``, ``reassess_metrics``

Usage Notes
-----------
- Reproducibility: Prefer functions that accept random seeds and write
    diagnositics under ``uqdd/logs``; capture versions and configurations for
    statistical comparisons.
- Data paths: Use the global paths from ``uqdd.__init__`` to keep file/plot
    outputs consistent.
- Plot styles: Use constants from ``metrics.constants`` to standardize the
    look and ordering across figures.
"""

from .constants import (
    group_cols,
    numeric_cols,
    string_cols,
    order_by,
    group_order,
    group_order_no_time,
    hatches_dict,
    hatches_dict_no_time,
    accmetrics,
    accmetrics2,
    uctmetrics,
    uctmetrics2,
)

from .analysis import (
    aggregate_results_csv,
    save_plot,
    handle_inf_values,
    plot_pairplot,
    plot_line_metrics,
    plot_histogram_metrics,
    plot_pairwise_scatter_metrics,
    plot_metrics,
    find_highly_correlated_metrics,
    plot_comparison_metrics,
    load_and_aggregate_calibration_data,
    plot_calibration_data,
    move_model_folders,
    load_predictions,
    calculate_rmse_rejection_curve,
    calculate_rejection_curve,
    get_handles_labels,
    plot_rmse_rejection_curves,
    plot_auc_comparison,
    save_stats_df,
    load_stats_df,
)

from .stats import (
    calc_regression_metrics,
    bootstrap_ci,
    rm_tukey_hsd,
    make_boxplots,
    make_boxplots_parametric,
    make_boxplots_nonparametric,
    make_sign_plots_nonparametric,
    make_critical_difference_diagrams,
    make_normality_diagnostic,
    mcs_plot,
    make_mcs_plot_grid,
    make_scatterplot,
    ci_plot,
    make_ci_plot_grid,
    recall_at_precision,
    calc_classification_metrics,
    make_curve_plots,
    harmonize_columns,
    cliffs_delta,
    wilcoxon_pairwise_test,
    holm_bonferroni_correction,
    pairwise_model_comparison,
    friedman_nemenyi_test,
    calculate_critical_difference,
    bootstrap_auc_difference,
    plot_critical_difference_diagram,
    analyze_significance,
    comprehensive_statistical_analysis,
    generate_statistical_report,
)

from .reassessment import (
    nll_evidentials,
    convert_to_list,
    preprocess_runs,
    get_model_class,
    get_predict_fn,
    get_preds,
    pkl_preds_export,
    csv_nll_post_processing,
    reassess_metrics,
)

__all__ = [
    # constants
    "group_cols",
    "numeric_cols",
    "string_cols",
    "order_by",
    "group_order",
    "group_order_no_time",
    "hatches_dict",
    "hatches_dict_no_time",
    "accmetrics",
    "accmetrics2",
    "uctmetrics",
    "uctmetrics2",
    # analysis
    "aggregate_results_csv",
    "save_plot",
    "handle_inf_values",
    "plot_pairplot",
    "plot_line_metrics",
    "plot_histogram_metrics",
    "plot_pairwise_scatter_metrics",
    "plot_metrics",
    "find_highly_correlated_metrics",
    "plot_comparison_metrics",
    "load_and_aggregate_calibration_data",
    "plot_calibration_data",
    "move_model_folders",
    "load_predictions",
    "calculate_rmse_rejection_curve",
    "calculate_rejection_curve",
    "get_handles_labels",
    "plot_rmse_rejection_curves",
    "plot_auc_comparison",
    "save_stats_df",
    "load_stats_df",
    # stats
    "calc_regression_metrics",
    "bootstrap_ci",
    "rm_tukey_hsd",
    "make_boxplots",
    "make_boxplots_parametric",
    "make_boxplots_nonparametric",
    "make_sign_plots_nonparametric",
    "make_critical_difference_diagrams",
    "make_normality_diagnostic",
    "mcs_plot",
    "make_mcs_plot_grid",
    "make_scatterplot",
    "ci_plot",
    "make_ci_plot_grid",
    "recall_at_precision",
    "calc_classification_metrics",
    "make_curve_plots",
    "harmonize_columns",
    "cliffs_delta",
    "wilcoxon_pairwise_test",
    "holm_bonferroni_correction",
    "pairwise_model_comparison",
    "friedman_nemenyi_test",
    "calculate_critical_difference",
    "bootstrap_auc_difference",
    "plot_critical_difference_diagram",
    "analyze_significance",
    "comprehensive_statistical_analysis",
    "generate_statistical_report",
    # reassessment
    "nll_evidentials",
    "convert_to_list",
    "preprocess_runs",
    "get_model_class",
    "get_predict_fn",
    "get_preds",
    "pkl_preds_export",
    "csv_nll_post_processing",
    "reassess_metrics",
]
