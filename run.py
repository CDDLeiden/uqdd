import pandas as pd
from papyrus import Papyrus
from papyrus_scripts.modelling import pcm, qsar
import xgboost
# generate xc50 and ki dataframes
papyrus_xc50 = Papyrus(
    path="data/",
    chunksize=1000000,
    accession=None,
    activity_type=["IC50", "EC50"],
    protein_class=None,
    verbose_files=True
)
df_xc50 = papyrus_xc50()


# Filtering the top 20 targets in number of datapoints.
# step 1: group the dataframe by protein target
grouped = df_xc50.groupby('accession')
# step 2: count the number of measurements for each protein target
counts = grouped['accession'].count()
# step 3: sort the counts in descending order
sorted_counts = counts.sort_values(ascending=False)

# step 4: select the 20 protein targets with the highest counts
top_targets = sorted_counts.head(20)

# step 5: filter the original dataframe to only include rows corresponding to these 20 protein targets
filtered_df = df_xc50[df_xc50['accession'].isin(top_targets.index)]

# step 6: iterate over the 20 protein targets
for protein_target in top_targets.index:
    # create a new dataframe containing only the rows corresponding to the current protein target
    target_df = df_xc50[df_xc50['protein_target'] == protein_target]

    # model the current protein target using target_df







papyrus_ki = Papyrus(
    path="data/",
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