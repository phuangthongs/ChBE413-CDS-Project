from enum import Enum

import pandas
import numpy
import sklearn
import rdkit
import rdkit.Chem

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, *args, **kwargs: x


class OutlierType(Enum):
    IQR = 1
    Z_SCORE = 2

# Helper functions to clean the dataframe
def isoutlier(
    dataframe: pandas.Series,
    method: OutlierType = OutlierType.IQR,
    cutoff: float = 1.5,
    exclude_cols: list[str] = None,
    progress=None,
) -> pandas.Series:
    outlier_df = {column: None for column in dataframe}
    df_len = len(dataframe)
    if exclude_cols is None:
        exclude_cols = []
    if progress is None:
        progress = lambda x, *args, **kwargs: x

    for column in progress(
        dataframe, desc="Detecting outliers", total=len(dataframe.columns)
    ):
        if not numpy.issubdtype(dataframe[column].dtype, numpy.number):
            outlier_df[column] = [False] * df_len
            continue
        if column in exclude_cols:
            outlier_df[column] = [False] * df_len
            continue

        col_data = dataframe[column]
        if method == OutlierType.Z_SCORE:
            col_mean = col_data.mean()
            col_std = col_data.std()
            lower = col_mean - cutoff * col_std
            upper = col_mean + cutoff * col_std
            outlier_df[column] = [
                (d < lower) or (upper < d) for _, d in col_data.items()
            ]
        elif method == OutlierType.IQR:
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            lower = q1 - cutoff * (q3 - q1)
            upper = q3 + cutoff * (q3 - q1)
            outlier_df[column] = [
                (d < lower) or (upper < d) for _, d in col_data.items()
            ]
        else:
            raise ValueError("Unknown method -- use 'z_score' or 'iqr'")

    return pandas.DataFrame(outlier_df)


def clean_dataframe(
    dataframe: pandas.Series,
    method: OutlierType = OutlierType.IQR,
    cutoff: float = 1.5,
    exclude_cols: list[str] = None,
    smiles_cols: list[str] = None,
    *,
    drop_na: bool = True,
    drop_duplicates: bool = True,
    verbose=True,
) -> pandas.Series:
    if exclude_cols is None:
        exclude_cols = []
    elif isinstance(exclude_cols, str):
        exclude_cols = [exclude_cols]

    if not verbose:
        f = lambda x, *args, **kwargs: x
    else:
        f = tqdm

    contains_outlier = ~isoutlier(
        dataframe, method, cutoff, exclude_cols, progress=f
    ).any(axis=1)
    clean_df = dataframe[contains_outlier].reset_index(drop=True)

    if smiles_cols is not None:
        invalid_smiles = {c: [] for c in smiles_cols}
        with rdkit.rdBase.BlockLogs():
            for c in smiles_cols:
                for _, smiles in f(
                    clean_df[c].items(), desc=f"Checking SMILES validity for column {c}"
                ):
                    mol = rdkit.Chem.MolFromSmiles(smiles)
                    invalid_smiles[c].append(mol is None)

        invalid_smiles = pandas.DataFrame(invalid_smiles)
        contains_invalid_smiles = ~invalid_smiles.any(axis=1)
        clean_df = clean_df[contains_invalid_smiles].reset_index(drop=True)

    if drop_na:
        clean_df = clean_df.dropna().reset_index(drop=True)
    if drop_duplicates:
        clean_df = clean_df.drop_duplicates().reset_index(drop=True)
    return clean_df


# Scales each column, then return the scaled dataframe. Also returns StandardScaler
# object to perform inverse transform later.
def rescale_features(
    dataframe: pandas.Series,
    exclude_cols: list[str] = None,
) -> tuple[pandas.Series, sklearn.preprocessing.StandardScaler]:
    rescaled_df = dataframe.copy()
    if exclude_cols is None:
        exclude_cols = []

    tf_cols = []
    for column in tqdm(
        dataframe, desc="Rescaling features", total=len(dataframe.columns)
    ):
        if (
            not numpy.issubdtype(dataframe[column].dtype, numpy.number)
            or column in exclude_cols
        ):
            continue
        tf_cols.append(column)

    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(rescaled_df[tf_cols].to_numpy())
    rescaled_df[tf_cols] = scaler.transform(rescaled_df[tf_cols].to_numpy())

    return pandas.DataFrame(rescaled_df), scaler
