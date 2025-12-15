# Constants and configuration for metrics analysis
import sys

INTERACTIVE_MODE = hasattr(sys, "ps1") or sys.flags.interactive

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
    "stratified_pnn",
    "stratified_ensemble",
    "stratified_mcdropout",
    "stratified_evidential",
    "stratified_eoe",
    "stratified_emc",
    "scaffold_cluster_pnn",
    "scaffold_cluster_ensemble",
    "scaffold_cluster_mcdropout",
    "scaffold_cluster_evidential",
    "scaffold_cluster_eoe",
    "scaffold_cluster_emc",
    "time_pnn",
    "time_ensemble",
    "time_mcdropout",
    "time_evidential",
    "time_eoe",
    "time_emc",
]

group_order_no_time = [
    "stratified_pnn",
    "stratified_ensemble",
    "stratified_mcdropout",
    "stratified_evidential",
    "stratified_eoe",
    "stratified_emc",
    "scaffold_cluster_pnn",
    "scaffold_cluster_ensemble",
    "scaffold_cluster_mcdropout",
    "scaffold_cluster_evidential",
    "scaffold_cluster_eoe",
    "scaffold_cluster_emc",
]

hatches_dict = {
    "stratified": "\\\\",
    "scaffold_cluster": "",
    "time": "...",
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

