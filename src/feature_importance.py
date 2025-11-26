import sys
import pathlib

import pandas
import sklearn
import numpy
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, *args, **kwargs: x

_PROJ_ROOT = pathlib.Path(__file__).parent.parent
if _PROJ_ROOT not in sys.path:
    sys.path.append(_PROJ_ROOT)


import load_utils


def read_and_process() -> pandas.Series:
    cond_df = pandas.read_csv(
        _PROJ_ROOT / pathlib.Path("./data/featurized_ionic_conductivity_dataset.csv")
    )
    return load_utils.rescale_features(
        load_utils.clean_dataframe(
            cond_df,
            method="iqr",
            cutoff=1.5,
            exclude_cols=[
                "Num_positive_atoms",
                "Num_negative_atoms",
                "Count_P",
                "Count_Cl",
                "Count_O",
                "Count_Si",
                "Count_C",
                "Count_F",
                "Count_N",
                "Count_S",
                "is_valid",
                "Heteroatom_count",
                "RingCount",
                "NumAromaticRings",
                "NumAliphaticRings",
            ],
            smiles_cols=["Mol_smiles_clean"],
            drop_na=True,
            drop_duplicates=True,
        ),
        exclude_cols=["CONDUCTIVITY"],
    )


def feat_importance_rf(df: pandas.Series):
    label_col = "CONDUCTIVITY"
    feat_exclude = ["Mol_smiles_clean", "is_valid"]
    feats = list(df.columns)

    for l in [label_col, *feat_exclude]:
        try:
            feats.remove(l)
        except ValueError:
            pass

    data_numpy = numpy.zeros((len(df), len(feats) + 1))

    data_numpy[:, 0] = df[label_col].to_numpy()
    for i, f in enumerate(feats):
        data_numpy[:, i + 1] = df[f].to_numpy()

    rf_reg = sklearn.ensemble.RandomForestRegressor()
    rf_reg.fit(data_numpy[:, 1:], data_numpy[:, 0])
    rf_pred = rf_reg.predict(data_numpy[:, 1:])

    sse = sum((rf_pred - data_numpy[:, 0]) ** 2)
    mse = sse / len(data_numpy)
    print(f"SSE = {sse}\nMSE={mse}")

    imp_feat = [
        (name, val)
        for name, val in zip(feats, rf_reg.feature_importances_)
        if val > 0.005
    ]
    imp_feat.sort(key=lambda x: x[1], reverse=True)

    imp_fig, imp_ax = plt.subplots(1, 1, figsize=(8, 8))
    imp_ax.bar([f[0] for f in imp_feat], [f[1] for f in imp_feat])
    imp_ax.set_xticks(range(len(imp_feat)), [f[0] for f in imp_feat], rotation=90)
    # imp_ax.set_xticklabels([f[0] for f in imp_feat], rotation=90)
    plt.tight_layout()
    plt.savefig("feat_importance_rf.png", dpi=300)  # transparent=True)
    # plt.show()

    return imp_feat


if __name__ == "__main__":
    clean_df, scaler = read_and_process()
    imp_feat_rf = feat_importance_rf(clean_df)
