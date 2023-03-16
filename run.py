from papyrus import Papyrus

papyrus_xc50 = Papyrus(
    path="data/",
    chunksize=100000,
    accession=None,
    activity_type=["IC50", "EC50"],
    protein_class=None,
    verbose_files=True
)
df_xc50 = papyrus_xc50()

papyrus_ki = Papyrus(
    path="data/",
    chunksize=100000,
    accession=None,
    activity_type=["Ki", "Kd"],
    protein_class=None,
    verbose_files=True
)
df_ki = papyrus_ki()
