# %%load libraries
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
from utils import fitting
from scipy import stats


# %% Main


proc_data_path = "./experiment_one/"

# load data
with open(os.path.join(proc_data_path, "habituation_fit_results.pkl"), "rb") as f:
    tonic_fits = pickle.load(f)

with open(os.path.join(proc_data_path, "pain_time_data.pkl"), "rb") as f2:
    tonic_data = pickle.load(f2)


def hex_to_rgba(h, alpha):
    """
    converts color value in hex format to rgba format with alpha transparency
    """
    return tuple([int(h.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)] + [alpha])


p_colours = [f"rgba{hex_to_rgba(c,0.2)}" for c in px.colors.qualitative.Dark24]
timepoints = np.arange(0, 630, 30).reshape(1, 21)  # up to 630 so 600 is included.

# %%
overall_fits = {}
for t in range(1, 4):
    overall_fits[t] = np.vstack(
        [
            tonic_fits[f"participant_trial{t}"]["exp"]["coeffs"][j]
            for j in tonic_fits[f"participant_trial{t}"]["exp"]["coeffs"].keys()
        ]
    )

timeconstants = pd.DataFrame(
    1
    / np.vstack(
        [overall_fits[1][:, 1], overall_fits[2][:, 1], overall_fits[3][:, 1]]
    ).T,
    columns=["t1", "t2", "t3"],
)

with pd.ExcelWriter(
    f"{proc_data_path}results/timeconstants_pertrial.xlsx",
    mode="w",
    engine="openpyxl",
) as writer:
    timeconstants.to_excel(writer)

participant_numbers = np.unique(tonic_data["pid"])


painint = np.full((17, 3, 21), np.nan)
for i, p in enumerate(participant_numbers):
    for t in range(1, 4):
        painint[i, t - 1, :] = np.expand_dims(
            np.expand_dims(
                tonic_data["painrating"][
                    (tonic_data["pid"] == p) & (tonic_data["trial"] == t)
                ],
                axis=0,
            ),
            axis=0,
        )
mean_painint = np.nanmean(painint, axis=0)
std_painint = np.nanstd(painint, axis=0)
median_painint = np.median(painint, axis=0)
iqr_painint = np.percentile(painint, [25, 75], axis=0)

fit_from_mean, fit_from_median = [], []
for t in range(1, 4):
    fit1, _, _, _ = fitting.fit_exp(timepoints.flatten(), mean_painint[t - 1])
    fit_from_mean.append(fit1)
    fit2, _, _, _ = fitting.fit_exp(timepoints.flatten(), median_painint[t - 1])
    fit_from_median.append(fit2)

pain_cis = np.full((2, 3, 21), np.nan)
for t in range(1, 4):
    for i in range(painint.shape[-1]):
        pain_data = painint[:, t - 1, i]
        pain_cis[:, t - 1, i] = stats.t.interval(
            confidence=0.95,
            df=pain_data.shape[0] - 1,
            loc=np.nanmean(pain_data),
            scale=stats.sem(pain_data),
        )

# // make plot
subp = make_subplots(
    rows=1,
    cols=3,
    shared_yaxes=True,
    x_title="Time (s)",
    y_title="Pain rating (0-10)",
    horizontal_spacing=0.04,
    vertical_spacing=0.1,
)
tplot = {}
for i, p in enumerate(participant_numbers):
    for t in range(1, 4):
        # t=1
        subp.add_trace(
            go.Scatter(
                x=tonic_data["timepoint"][
                    (tonic_data["pid"] == p) & (tonic_data["trial"] == t)
                ],
                y=tonic_data["painrating"][
                    (tonic_data["pid"] == p) & (tonic_data["trial"] == t)
                ]
                .to_numpy()
                .flatten(),
                mode="lines",
                name=f"P{p}",
                line=dict(dash="solid", color=p_colours[i], width=1.5),
                showlegend=False,
            ),
            row=1,
            col=t,
        )


for t in range(1, 4):
    subp.add_trace(
        go.Scatter(
            x=timepoints.flatten(),
            y=median_painint[t - 1],
            error_y=dict(
                type="data",  # value of error bar given in data coordinates
                symmetric=False,
                array=iqr_painint[0, t - 1] - median_painint[t - 1],
                arrayminus=median_painint[t - 1] - iqr_painint[1, t - 1],
                visible=True,
                color="black",
            ),
            mode="markers",
            name="Mean Habituation Trial 1",
            marker=dict(color="white", size=8, line=dict(color="black", width=2)),
            showlegend=False,
        ),
        row=1,
        col=t,
    )

    subp.add_trace(
        go.Scatter(
            x=timepoints.flatten(),
            y=fitting.exponential(timepoints.flatten(), *fit_from_median[t - 1]),
            mode="lines",
            name=f"Mean Habituation Trial{t}",
            line=dict(dash="solid", color="black", width=2),
            showlegend=False,
        ),
        row=1,
        col=t,
    )
subp.update_layout(
    width=1300,
    height=600,
    template="simple_white",
    title="Pain Ratings Over Time",
    legend=dict(orientation="v"),
    font_family="Arial, sans-serif",
    font_size=22,
)
subp.update_xaxes(range=[-5, 605])
subp.update_annotations(font_size=22, font_family="Arial, sans-serif")
subp.show()
subp.write_image(f"{proc_data_path}results/pain_habituation_plot-3panel.svg")
