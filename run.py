import pandas as pd
import xgboost
from papyrus_scripts.modelling import pcm

from uqdd.chem_utils import generate_ecfp, generate_mol_descriptors
from uqdd.models.papyrus import Papyrus

# generate xc50 and ki dataframes
# papyrus_xc50 = Papyrus(
#     path="data/",
#     chunksize=1000000,
#     accession=None,
#     activity_type=["IC50", "EC50"],
#     protein_class=None,
#     verbose_files=True
# )
# df_xc50 = papyrus_xc50()
#

papyrus_xc50 = Papyrus(
    path="uqdd/data/papyrus_filtered_high_quality_xc50_00_preprocessed.csv",
    chunksize=1000000,
    accession=None,
    activity_type=["IC50", "EC50"],
    protein_class=None,
    verbose_files=True
)
df_xc50 = papyrus_xc50()

df_xc50 = pd.read_csv("uqdd/data/papyrus_filtered_high_quality_xc50.csv")
df_xc50 = generate_ecfp(df_xc50, 2, 1024, False, False)
df_xc50.to_csv("data/papyrus_filtered_high_quality_xc50_04_with_ECFP.csv", index=False)

df_xc50 = generate_mol_descriptors(df_xc50, 'smiles', None)
df_xc50.to_csv("data/papyrus_filtered_high_quality_xc50_05_with_molecular_descriptors.csv", index=False)
# # calculate ECFP fingerprints
# self.df_filtered = generate_ecfp(self.df_filtered, 2, 1024, False, False)
#
# # calculate mol descriptors
# self.df_filtered = generate_mol_descriptors(self.df_filtered, 'smiles', None)

# Filtering the top 25 targets in number of datapoints.
# step 1: group the dataframe by protein target
grouped = df_xc50.groupby('accession')
# step 2: count the number of measurements for each protein target
counts = grouped['accession'].count()
# step 3: sort the counts in descending order
sorted_counts = counts.sort_values(ascending=False)

# step 4: select the 20 protein targets with the highest counts
top_targets = sorted_counts.head(25)

# step 5: filter the original dataframe to only include rows corresponding to these 20 protein targets
filtered_df = df_xc50[df_xc50['accession'].isin(top_targets.index)]

# step 6: iterate over the 25 protein targets
for protein_target in top_targets.index:
    # create a new dataframe containing only the rows corresponding to the current protein target
    target_df = df_xc50[df_xc50['protein_target'] == protein_target]



    # model the current protein target using target_df







papyrus_ki = Papyrus(
    path="uqdd/data/",
    chunksize=1000000,
    accession=None,
    activity_type=["Ki", "Kd"],
    protein_class=None,
    verbose_files=True
)
df_ki = papyrus_ki()



# (A) modeling with papyrus xc50:
pcm_reg_model = xgboost.XGBRegressor(verbosity=0)

# need to iterate over targets!!!
pcm_reg_results, pcm_reg_trained_model = pcm(data=target_data,
                                             version='latest',
                                             endpoint='pchembl_value_Mean',
                                             num_points=30,
                                             delta_activity=2,
                                             mol_descriptors='mold2',
                                             mol_descriptor_chunksize=50000,
                                             prot_descriptors='unirep',
                                             prot_descriptor_chunksize=50000,
                                             activity_threshold=6.5,
                                             model=pcm_reg_model,
                                             folds=5,
                                             stratify=False,
                                             split_by='Year',
                                             split_year=2013,
                                             test_set_size=0.30,
                                             cluster_method=None,
                                             custom_groups=None,
                                             random_state=1234,
                                             verbose=True
                                             )