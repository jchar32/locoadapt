# %%

import numpy as np
import pandas as pd
import pickle
import os
import scipy.stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

proc_data_path = "../experiment_one/"

#  Load data
with open(os.path.join(proc_data_path, "habituation_fit_results.pkl"), "rb") as f:
    tonic_fits = pickle.load(f)

with open(os.path.join(proc_data_path, "pain_time_data.pkl"), "rb") as f2:
    tonic_data = pickle.load(f2)

# %% Statistical analysis

timepoints = list(np.arange(0, 630, 30))
indiv_timepoint_results_nonpara = {"t1": {}, "t2": {}, "t3": {}}

ro = [0, 1, 2, 3, 4] * 4
co = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
for t in range(1, 4):
    fig, ax = plt.subplots(5, 4, figsize=(10, 10))
    for idx, i in enumerate(timepoints[1:]):
        tonic_data_diff = (
            tonic_data["painrating"][
                (tonic_data["trial"] == t) & (tonic_data["timepoint"] == 5)
            ].to_numpy()
            - tonic_data["painrating"][
                (tonic_data["trial"] == t) & (tonic_data["timepoint"] == i)
            ].to_numpy()
        )

        sm.qqplot(tonic_data_diff, line="q", ax=ax[ro[idx], co[idx]])
        ax[ro[idx], co[idx]].set_title(f"Timepoint {i}sec")

        indiv_timepoint_results_nonpara[f"t{t}"][f"{i}"] = scipy.stats.wilcoxon(
            tonic_data_diff,
            zero_method="pratt",  # chosen b/c it is most conservative (zero diff ranks are dropped)
            correction=False,
            alternative="greater",
            method="auto",
            nan_policy="omit",  # type: ignore
        )

    fig.suptitle(f"Pain intensity 5-second vs Timepoint x: Trial {t}")
    plt.tight_layout()
    plt.show()

for t in range(1, 4):
    painrating_wilcox = pd.DataFrame.from_dict(
        indiv_timepoint_results_nonpara[f"t{t}"], orient="index"
    )
    # bonferroni correction
    painrating_wilcox["pvalue_bonf"] = painrating_wilcox["pvalue"] * (
        len(timepoints) - 1
    )

    with pd.ExcelWriter(
        f"{proc_data_path}results/painrating_wilcox.xlsx", mode="w" if t == 1 else "a"
    ) as writer:
        painrating_wilcox.to_excel(writer, sheet_name=f"t{t}")

pain_intensity_arr = np.full((17, 3, 21), np.nan)
for i, p in enumerate(tonic_data["pid"].unique()):
    for t in range(1, 4):
        pain_intensity_arr[i, t - 1, :] = np.expand_dims(
            np.expand_dims(
                tonic_data["painrating"][
                    (tonic_data["pid"] == p) & (tonic_data["trial"] == t)
                ],
                axis=0,
            ),
            axis=0,
        )
pain_intensity_df_t1 = pd.DataFrame(pain_intensity_arr[:, 0, :]).describe()
pain_intensity_df_t2 = pd.DataFrame(pain_intensity_arr[:, 1, :]).describe()
pain_intensity_df_t3 = pd.DataFrame(pain_intensity_arr[:, 2, :]).describe()

with pd.ExcelWriter(
    f"{proc_data_path}results/painrating_summary_statistics.xlsx", mode="w"
) as writer:
    pain_intensity_df_t1.to_excel(writer, sheet_name="t1")
with pd.ExcelWriter(
    f"{proc_data_path}results/painrating_summary_statistics.xlsx", mode="a"
) as writer:
    pain_intensity_df_t2.to_excel(writer, sheet_name="t2")
with pd.ExcelWriter(
    f"{proc_data_path}results/painrating_summary_statistics.xlsx", mode="a"
) as writer:
    pain_intensity_df_t3.to_excel(writer, sheet_name="t3")

# %% test difference in 5-second pain intensity rating across trials

sumdata = tonic_data.loc[:, ["painrating", "trial", "timepoint"]]
sumdata_described_5s = sumdata[(sumdata["timepoint"] == 5)].groupby("trial").describe()

t1_v_t2 = scipy.stats.wilcoxon(
    (
        tonic_data["painrating"][
            (tonic_data["trial"] == 1) & (tonic_data["timepoint"] == 5)
        ].to_numpy()
        - tonic_data["painrating"][
            (tonic_data["trial"] == 2) & (tonic_data["timepoint"] == 5)
        ].to_numpy()
    ),
    zero_method="pratt",  # chosen b/c it is most conservative (zero diff ranks are dropped)
    correction=False,
    alternative="two-sided",
    method="auto",
)
t1_v_t3 = scipy.stats.wilcoxon(
    (
        tonic_data["painrating"][
            (tonic_data["trial"] == 1) & (tonic_data["timepoint"] == 5)
        ].to_numpy()
        - tonic_data["painrating"][
            (tonic_data["trial"] == 3) & (tonic_data["timepoint"] == 5)
        ].to_numpy()
    ),
    zero_method="pratt",  # chosen b/c it is most conservative (zero diff ranks are dropped)
    correction=False,
    alternative="two-sided",
    method="auto",
)
t2_v_t3 = scipy.stats.wilcoxon(
    (
        tonic_data["painrating"][
            (tonic_data["trial"] == 2) & (tonic_data["timepoint"] == 5)
        ].to_numpy()
        - tonic_data["painrating"][
            (tonic_data["trial"] == 3) & (tonic_data["timepoint"] == 5)
        ].to_numpy()
    ),
    zero_method="pratt",
    correction=False,
    alternative="two-sided",
    method="auto",
)

first_pain_diff_dict = {}
first_pain_diff_dict["t1_v_t2"] = t1_v_t2
first_pain_diff_dict["t1_v_t3"] = t1_v_t3
first_pain_diff_dict["t2_v_t3"] = t2_v_t3
first_pain_diff_results = pd.DataFrame.from_dict(first_pain_diff_dict).T
first_pain_diff_results.columns = ["w", "pvalue"]


# compile and save results to excel
with pd.ExcelWriter(
    f"{proc_data_path}results/initialpain_comparison.xlsx", mode="w"
) as writer:
    first_pain_diff_results.to_excel(writer, sheet_name="wilcox")
with pd.ExcelWriter(
    f"{proc_data_path}results/initialpain_comparison.xlsx", mode="a"
) as writer:
    sumdata_described_5s.to_excel(writer, sheet_name="summary_data")
