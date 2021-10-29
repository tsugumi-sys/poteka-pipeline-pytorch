import logging

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def imputer(df):
    label = df["target"]
    features_df = df.drop(columns="target", axis=1)
    features_df_columns = features_df.columns

    imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean")
    imp_mean.fit(features_df)

    imputed_features_arr = imp_mean.fit_transform(features_df)

    imputed_df = pd.DataFrame(imputed_features_arr, columns=features_df_columns, index=features_df.index)
    imputed_df = imputed_df.merge(label, how="inner", right_index=True, left_index=True)

    return imputed_df, imp_mean


def fit_imputer(df, transformer):
    label = df["target"]
    features_df = df.drop(columns="target", axis=1)
    features_df_columns = features_df.columns

    imputed_features_arr = transformer.fit_transform(features_df)

    imputed_df = pd.DataFrame(imputed_features_arr, columns=features_df_columns, index=features_df.index)
    imputed_df = imputed_df.merge(label, how="inner", right_index=True, left_index=True)

    return imputed_df


def parse_parquet(df, root_dir: str) -> str:
    filename = "data.parquet.gzip"
    df.to_parquet(
        f"{root_dir}/{filename}",
        engine="pyarrow",
        compression="gzip",
    )

    return filename
