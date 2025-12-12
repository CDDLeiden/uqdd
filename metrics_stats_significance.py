import os
import pandas as pd

from uqdd.metrics import (
    analyze_significance,
    comprehensive_statistical_analysis,
)


if __name__ == "__main__":
    xc50 = "/home/bkhalil/Repos/uqdd/results_revision/final_xc50.csv"
    kx = "/home/bkhalil/Repos/uqdd/results_revision/final_kx.csv"

    df_xc50 = pd.read_csv(xc50)
    df_kx = pd.read_csv(kx)

    metrics = [
        "RMSE",
        "Miscalibration Area",
        "NLL",
        "CRPS",
        "Interval",
        "Sharpness",
    ]

    direction_dict = {
        "RMSE": "minimize",
        "Miscalibration Area": "minimize",
        "Sharpness": "minimize",
        "NLL": "minimize",
        "Interval": "minimize",
        "CRPS": "minimize",
    }

    effect_dict = {k: 0.1 for k in metrics}

    model_order = ["pnn", "ensemble", "eoe", "evidential", "emc", "mcdropout"]

    save_dir = "stat_analysis_results"
    os.makedirs(save_dir, exist_ok=True)

    save_dir_xc50 = os.path.join(save_dir, "xc50")
    os.makedirs(save_dir_xc50, exist_ok=True)
    print(f"\n=== Activity: xc50 ===")
    analyze_significance(
        df_xc50,
        metrics,
        direction_dict,
        effect_dict,
        save_dir=save_dir_xc50,
        model_order=model_order,
        activity="xc50",
    )

    results_xc50 = comprehensive_statistical_analysis(
        df_xc50,
        metrics=metrics,
        models=None,
        tasks=None,
        splits=None,
        save_dir=save_dir_xc50,
        alpha=0.05,
    )

    save_dir_kx = os.path.join(save_dir, "kx")
    os.makedirs(save_dir_kx, exist_ok=True)
    print(f"\n=== Activity: kx ===")
    analyze_significance(
        df_kx,
        metrics,
        direction_dict,
        effect_dict,
        save_dir=save_dir_kx,
        model_order=model_order,
        activity="kx",
    )

    results_kx = comprehensive_statistical_analysis(
        df_kx,
        metrics=metrics,
        models=None,
        tasks=None,
        splits=None,
        save_dir=save_dir_kx,
        alpha=0.05,
    )
