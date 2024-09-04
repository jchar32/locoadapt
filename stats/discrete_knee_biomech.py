# %%
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
import pickle

proc_data_path = "../experiment_one/"
# %%

filesubnames = ["_first_ten", "_last_ten"]
for filesubname in filesubnames:
    with open(
        os.path.join(proc_data_path, f"summary_discrete{filesubname}.pkl"), "rb"
    ) as f:
        discrete_data = pickle.load(f)

    outcomenames = list(discrete_data["joint"]["rknee"].keys())[:3]

    biomech_discrete_wilcox = {o: {} for o in outcomenames}
    plt.style.use("default")
    fig, ax = plt.subplots(3, 3, figsize=(10, 10))

    for i, outcome in enumerate(outcomenames):
        data = discrete_data["joint"]["rknee"][outcome].copy()
        # !! REMOVE p17 Trial 3 due to sensor issue
        data[-1, -1] = np.nan
        # !! ------------------

        summary_data = pd.DataFrame(data, columns=["base", "t1", "t2", "t3"]).describe()

        fname = f"{proc_data_path}results/discrete_biomech_summary{filesubname}.xlsx"
        file_exists = os.path.isfile(fname)
        with pd.ExcelWriter(
            fname,
            mode="w" if not file_exists else "a",
            engine="openpyxl",
        ) as writer:
            summary_data.to_excel(writer, sheet_name=outcome)

        diff_df = []
        test_direction = "greater"
        for t in range(3):
            diff = data[:, 0] - data[:, t + 1]

            sm.qqplot(diff, line="q", ax=ax[t, i])
            ax[t, i].set_title(f"{outcome} Trial 1-{t+1}")

            biomech_discrete_wilcox[outcome][f"b-t{t+1}"] = wilcoxon(
                diff,
                zero_method="pratt",
                correction=False,
                alternative=test_direction,
                method="auto",
                nan_policy="omit",  # type: ignore
            )

            diff_df.append(diff)
        diff_df = np.vstack(diff_df)

        fname = f"{proc_data_path}results/discrete_biomech_difference_summary{filesubname}.xlsx"
        file_exists = os.path.isfile(fname)
        with pd.ExcelWriter(
            fname,
            mode="w" if not file_exists else "a",
            engine="openpyxl",
        ) as writer:
            pd.DataFrame(diff_df.T, columns=["t1", "t2", "t3"]).describe().to_excel(
                writer, sheet_name=outcome
            )

    plt.tight_layout()
    plt.show()

    # // Compile stats results
    biomech_discrete_wilcox_df = pd.DataFrame()
    for i, outcome in enumerate(outcomenames):
        for comp in biomech_discrete_wilcox[outcome].keys():
            biomech_discrete_wilcox_df.loc[comp, "p"] = biomech_discrete_wilcox[
                outcome
            ][comp][1]
            biomech_discrete_wilcox_df.loc[comp, "w"] = biomech_discrete_wilcox[
                outcome
            ][comp][0]

        fname = f"{proc_data_path}results/discrete_biomech_wilcox{filesubname}.xlsx"
        file_exists = os.path.isfile(fname)
        with pd.ExcelWriter(
            fname,
            mode="w" if not file_exists else "a",
            engine="openpyxl",
        ) as writer:
            biomech_discrete_wilcox_df.to_excel(writer, sheet_name=f"{outcome}")
