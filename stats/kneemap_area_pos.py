# %%
import numpy as np
import pickle
import pandas as pd
import os
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

proc_data_path = "../experiment_two/"  # // either "experiment_one" or "experiment_two"

with open(f"{proc_data_path}all_maps_frontal.pkl", "rb") as f:
    all_fr_img_stack = pickle.load(f)
with open(f"{proc_data_path}all_maps_transverse.pkl", "rb") as f:
    all_tr_img_stack = pickle.load(f)

moments = pd.read_csv(os.path.join(proc_data_path, "all_maps_moments.csv"))

num_conditions = 3

# %%
# // collate moment area (sum across n blobs per picture and condition)

un_pid = moments["participant"].unique()
views = ["Page 1", "Page 2"]
conds = ["1", "2", "3"]

varnames = ["pid", "views", "conds", "area", "x_pos", "y_pos"]
moment_df_ = {varn: [] for varn in varnames}

area_dict = {}
for p in un_pid:
    row_id = np.where(moments["participant"] == p)[0]
    data_ = moments.loc[row_id, :].reset_index()

    for v in views:
        v_id = np.where(data_["view"] == v)[0]
        data_filt = data_.loc[v_id, :].reset_index()

        area_, x_, y_ = [], [], []
        for c in conds:
            c_id = np.where(data_filt["cond"] == int(c))[0]
            area_.append(data_filt.loc[c_id, "area_perc"].sum())
            x_.append(data_filt.loc[c_id, "x"].mean())
            y_.append(data_filt.loc[c_id, "y"].mean())

        df_ = {
            "pid": [p, p, p],
            "views": [v, v, v],
            "conds": conds,
            "area": area_,
            "x_pos": x_,
            "y_pos": y_,
        }
        for varn in varnames:
            moment_df_[varn].extend(df_[varn])
moment_df = pd.DataFrame(moment_df_)

moment_df["views"] = moment_df["views"].replace(
    {"Page 1": "frontal view", "Page 2": "transverse view"}
)

summary_df = moment_df.groupby(["views", "conds"]).describe()

with pd.ExcelWriter(
    f"{proc_data_path}results/painregions_summary_data.xlsx", mode="w"
) as writer:
    summary_df.to_excel(writer)


# %% mean change in area across trials/condtions
def summarize_area_change(data, view):
    area_diff_1_2 = []
    area_diff_1_2_perc_change = []
    area_diff_1_3 = []
    area_diff_1_3_perc_change = []

    for i, p in enumerate(data["pid"].unique()):
        t1_data = data.loc[
            (data["pid"] == p)
            & (data["conds"] == "1")
            & (data["views"] == f"{view} view"),
            "area",
        ].values

        area_diff_1_2.append(
            t1_data
            - data.loc[
                (data["pid"] == p)
                & (data["conds"] == "2")
                & (data["views"] == f"{view} view"),
                "area",
            ].values
        )
        area_diff_1_2_perc_change.append(area_diff_1_2[-1] / t1_data)

        area_diff_1_3.append(
            t1_data
            - data.loc[
                (data["pid"] == p)
                & (data["conds"] == "3")
                & (data["views"] == f"{view} view"),
                "area",
            ].values
        )
        area_diff_1_3_perc_change.append(area_diff_1_3[-1] / t1_data)

    tbl1 = pd.DataFrame(np.stack(area_diff_1_2)).describe() * 100
    tbl2 = pd.DataFrame(np.stack(area_diff_1_3)).describe() * 100
    tbl3 = pd.DataFrame(np.stack(area_diff_1_2_perc_change)).describe() * 100
    tbl4 = pd.DataFrame(np.stack(area_diff_1_3_perc_change)).describe() * 100

    fname = f"{proc_data_path}results/painregions_area_changedata.xlsx"
    file_exists = os.path.isfile(fname)

    with pd.ExcelWriter(fname, mode="w" if not file_exists else "a") as writer:
        tbl1.to_excel(writer, sheet_name=f"1v2_change_{view}")
    with pd.ExcelWriter(fname, mode="a") as writer:
        tbl2.to_excel(writer, sheet_name=f"1v3_change_{view}")
    with pd.ExcelWriter(fname, mode="a") as writer:
        tbl3.to_excel(writer, sheet_name=f"1v2_perc_change_{view}")
    with pd.ExcelWriter(fname, mode="a") as writer:
        tbl4.to_excel(writer, sheet_name=f"1v3_perc_change_{view}")


summarize_area_change(moment_df, view="frontal")
summarize_area_change(moment_df, view="transverse")


# %% Check dists
plt.style.use("default")
fig, ax = plt.subplots(3, 3, figsize=(10, 10))
for ni, n in enumerate(varnames[3:]):
    for vi, v in enumerate(moment_df["views"].unique()):
        diff = (
            moment_df[n][
                (moment_df["views"] == v) & (moment_df["conds"] == str(1))
            ].reset_index()
            - moment_df[n][
                (moment_df["views"] == v) & (moment_df["conds"] == str(2))
            ].reset_index()
        )
        sm.qqplot(
            diff.loc[:, n],
            line="q",
            ax=ax[ni, 0],
            markerfacecolor="black" if vi == 0 else "white",
            markeredgecolor="black",
            label=f"{v}" if ni == 0 else None,
        )
        ax[ni, 0].set_title(f"{n} {v} 1 vs 2")

        diff = (
            moment_df[n][
                (moment_df["views"] == v) & (moment_df["conds"] == str(1))
            ].reset_index()
            - moment_df[n][
                (moment_df["views"] == v) & (moment_df["conds"] == str(3))
            ].reset_index()
        )
        sm.qqplot(
            diff.loc[:, n],
            line="q",
            ax=ax[ni, 1],
            markerfacecolor="black" if vi == 0 else "white",
            markeredgecolor="black",
        )
        ax[ni, 1].set_title(f"{n} {v} 1 vs 3")
        diff = (
            moment_df[n][
                (moment_df["views"] == v) & (moment_df["conds"] == str(2))
            ].reset_index()
            - moment_df[n][
                (moment_df["views"] == v) & (moment_df["conds"] == str(3))
            ].reset_index()
        )
        sm.qqplot(
            diff.loc[:, n],
            line="q",
            ax=ax[ni, 2],
            markerfacecolor="black" if vi == 0 else "white",
            markeredgecolor="black",
        )
        ax[ni, 2].set_title(f"{n} {v} 2 vs 3")
fig.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.52, 1.05))
plt.tight_layout()
plt.show()


# %% Statistical comparisons
def wilcox_test(data, view, t_1, t_2):
    return stats.wilcoxon(
        data[n][(data["views"] == view) & (data["conds"] == str(t_1))],
        data[n][(data["views"] == view) & (data["conds"] == str(t_2))],
        zero_method="wilcox",
        correction=False,
        alternative="two-sided",
        method="auto",
    )


res_wilcox = {"frontal view": {}, "transverse view": {}}
for v in moment_df["views"].unique():
    for ni, n in enumerate(varnames[3:]):
        res_wilcox[v][f"{n}_1-2"] = wilcox_test(moment_df, v, 1, 2)
        res_wilcox[v][f"{n}_1-3"] = wilcox_test(moment_df, v, 1, 3)
        res_wilcox_df = pd.DataFrame.from_dict(res_wilcox[f"{v}"], orient="index")

    fname = f"{proc_data_path}results/painregions_statistics.xlsx"
    file_exists = os.path.isfile(fname)
    with pd.ExcelWriter(fname, mode="w" if not file_exists else "a") as writer:
        res_wilcox_df.to_excel(writer, sheet_name=v)
