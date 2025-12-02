import sys
import pathlib

import pandas
import sklearn
import numpy
import torch
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, *args, **kwargs: x

_PROJ_ROOT = pathlib.Path(__file__).parent.parent
if _PROJ_ROOT not in sys.path:
    sys.path.append(_PROJ_ROOT)


import src.load_utils


def read_no_rescale(path="./data/combined_featurized_data.csv") -> pandas.Series:
    cond_df = pandas.read_csv(
        _PROJ_ROOT / pathlib.Path(path)
    )
    return src.load_utils.clean_dataframe(
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
    )


def read_and_process() -> pandas.Series:
    return src.load_utils.rescale_features(
        read_no_rescale(),
        exclude_cols=["CONDUCTIVITY"],
    )


def to_np_data(df: pandas.Series):
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

    return feats, data_numpy


def feat_importance_rf(df: pandas.Series):
    feats, data_numpy = to_np_data(df)
    rf_reg = sklearn.ensemble.RandomForestRegressor()
    rf_reg.fit(data_numpy[:, 1:], data_numpy[:, 0])
    rf_pred = rf_reg.predict(data_numpy[:, 1:])

    sse = sum((rf_pred - data_numpy[:, 0]) ** 2)
    mse = sse / len(data_numpy)
    print(f"SSE = {sse}\nMSE={mse}")

    imp_feat = [
        (name, val)
        for name, val in zip(feats, rf_reg.feature_importances_)
        # if val > 0.005
    ]
    imp_feat.sort(key=lambda x: x[1], reverse=True)

    imp_fig, imp_ax = plt.subplots(1, 1, figsize=(8, 8))
    plot_cutoff = 0.005
    plot_label = [f[0] for f in imp_feat if f[1] > plot_cutoff]
    plot_value = [f[1] for f in imp_feat if f[1] > plot_cutoff]
    imp_ax.bar(plot_label, plot_value)
    imp_ax.set_xticks(range(len(plot_label)), plot_label, rotation=90)
    # imp_ax.set_xticklabels([f[0] for f in imp_feat], rotation=90)
    plt.tight_layout()
    plt.savefig("feat_importance_rf.png", dpi=300)  # transparent=True)
    # plt.show()

    return imp_feat


def construct_mlp(sizes: list[int], sample_frame: torch.Tensor = None):
    if sizes[0] < 0 and sample_frame is not None:
        sizes[0] = len(sample_frame)

    modules = []
    for i, (size_in, size_out) in enumerate(zip(sizes, sizes[1:])):
        modules.append(torch.nn.Linear(size_in, size_out, dtype=torch.float64))
        if i < len(sizes) - 2:
            modules.append(torch.nn.ReLU())

    return torch.nn.Sequential(*modules)


def train_nn(df: pandas.Series, col_names: list[str]):
    names = col_names
    if "CONDUCTIVITY" not in names:
        names.insert(0, "CONDUCTIVITY")
    feats, data_tensor = to_np_data(df[names])
    data_tensor = torch.tensor(data_tensor)

    model = construct_mlp([-1, 16, 4, 1], data_tensor[0, 1:])
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"There are {n_params} trainable parameters.")

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for _ in tqdm(range(256), desc="Training"):
        pred = model(data_tensor[:, 1:])
        loss = loss_fn(pred.reshape((-1,)), data_tensor[:, 0])
        # print(f"Current Loss = {loss}\033[A\r", end="")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Final Loss = {loss}")


if __name__ == "__main__":
    clean_df, scaler = read_and_process()
    imp_feat_list = []
    try:
        with open("imp_feat_list.txt", "r") as file:
            for line in file:
                imp_feat_list.append(line.strip())
    except FileNotFoundError:
        imp_feat_rf = feat_importance_rf(clean_df)
        with open("imp_feat_list.txt", "w") as file:
            for name, _ in imp_feat_rf:
                file.write(f"{name}\n")
        imp_feat_list = [f[0] for f in imp_feat_rf]

    train_nn(clean_df, imp_feat_list)
    train_nn(clean_df, imp_feat_list[10:])
