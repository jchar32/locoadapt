# %%
import os
import numpy as np
import pickle
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import event_detectors  # needed to load "waveforms.pkl" #type: ignore


# // To allow for Latex rendering in plotly
from IPython.display import display, HTML

plotly.offline.init_notebook_mode()
display(
    HTML(
        '<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_SVG"></script>'
    )
)
# !! Mathjax version must match what is installed.

proc_data_path = "./experiment_one/"


# generate
def hex_to_rgba(h, alpha):
    """
    converts color value in hex format to rgba format with alpha transparency
    """
    return tuple([int(h.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)] + [alpha])


# Load the data
with open(os.path.join(proc_data_path, "summary_waveforms_first_ten.pkl"), "rb") as f:
    summary_data_waveform_first_ten = pickle.load(f)
with open(os.path.join(proc_data_path, "summary_discrete.pkl"), "rb") as f:
    summary_data_discrete = pickle.load(f)


# %% Custom Plot Functions


def line_sd_band_plot(
    fig_obj,
    line_data,
    upper_bound,
    lower_bound,
    xdata=None,
    nameval=None,
    line_colour="black",
    fill_colour="rgba(50,50,50,0.5)",
    row=1,
    col=1,
    linestyle_name="solid",
    linewidth_value=2,
):
    fig_obj.add_trace(
        go.Scatter(
            x=xdata,
            y=line_data,
            mode="lines",
            name=nameval,
            line=dict(color=line_colour, dash=linestyle_name, width=linewidth_value),
            showlegend=True if nameval is not None else False,
        ),
        row=row,
        col=col,
    )
    fig_obj.add_trace(
        go.Scatter(
            x=xdata,
            y=upper_bound,
            # fill="tonexty",
            mode="lines",
            name=f"{nameval} Upper",
            marker=dict(color=line_colour),
            line=dict(width=0),
            showlegend=False,
        ),
        row=row,
        col=col,
    )
    fig_obj.add_trace(
        go.Scatter(
            x=xdata,
            y=lower_bound,
            fill="tonexty",
            mode="lines",
            name=f"{nameval} Lower",
            marker=dict(color=line_colour),
            line=dict(width=0),
            fillcolor=fill_colour,
            showlegend=False,
        ),
        row=row,
        col=col,
    )
    return fig_obj


# // set colours
colour_list = px.colors.qualitative.Dark24
# colour_list = ["#6E66E6", "#E66E66", "#66E66E"]
p_colours_alpha = [f"rgba{hex_to_rgba(c,0.2)}" for c in colour_list]
p_colours_alpha.append("rgba(0,0,0,0.1)")
p_colours = [f"rgba{hex_to_rgba(c,1)}" for c in colour_list]


# // raw angles
#  load raw waveform from participant
filename = f"{proc_data_path}/p8_waveforms.pkl"
with open(filename, "rb") as f:
    raw_waveform = pickle.load(f)

raw_rknee_x_baseline = raw_waveform["joint"]["rknee"][5]["x"]
raw_rknee_x_t1 = raw_waveform["joint"]["rknee"][6]["x"]
hs_baseline = raw_waveform["events"]["r"][0].hs
hs_t1 = raw_waveform["events"]["r"][1].hs
# %%
# // Raw participant data
wfrm_raw_plot = make_subplots(
    rows=1,
    cols=1,
    vertical_spacing=0.1,
)
wfrm_raw_plot.add_trace(
    go.Scatter(
        x=np.arange(0, (hs_baseline[4] - hs_baseline[2]) / 120, 1 / 120),
        y=raw_rknee_x_baseline[hs_baseline[2] : hs_baseline[4]],
        mode="lines",
        line=dict(color="black", width=2),
        name="Baseline",
    ),
    row=1,
    col=1,
)
wfrm_raw_plot.add_trace(
    go.Scatter(
        x=np.arange(0, (hs_t1[4] - hs_t1[2]) / 120, 1 / 120),
        y=raw_rknee_x_t1[hs_t1[2] : hs_t1[4]],
        mode="lines",
        line=dict(color=p_colours[0], width=2),
        name="Trial 1",
    ),
    row=1,
    col=1,
)
wfrm_raw_plot.update_layout(
    width=750,
    height=525,
    title_text="",
    template="simple_white",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.4,
        x=0.3,
    ),
    font_family="Arial, sans-serif",
    font_size=18,
)
wfrm_raw_plot.update_yaxes(title_text="Angles (\u00b0)", row=1, col=1)
wfrm_raw_plot.update_xaxes(title_text="Time (s)", row=1, col=1)
wfrm_raw_plot.show()

fname = f"{proc_data_path}results/representative_partic_knee_angle.svg"
wfrm_raw_plot.write_image(fname)


# // Mean per trial plot
mean_waveform = summary_data_waveform_first_ten["joint"]["rknee"]

# !! REMOVED participant 17 trial 3
mean_waveform["mean"][16, 3, :, :] = np.nan
mean_over_partic = np.nanmean(mean_waveform["mean"], axis=0)
mean_sd_over_partic = np.nanstd(mean_waveform["mean"], axis=0)

# baseline
baseline_mean_y = mean_over_partic[0, :, 0]
baseline_y_upper = baseline_mean_y + mean_sd_over_partic[0, :, 0]
baseline_y_lower = baseline_mean_y - mean_sd_over_partic[0, :, 0]

mean_plot = make_subplots(
    rows=1,
    cols=1,
)
mean_plot = line_sd_band_plot(
    mean_plot,
    baseline_mean_y,
    baseline_y_upper,
    baseline_y_lower,
    xdata=np.arange(0, 101, 1),
    nameval="Baseline",
    line_colour="black",
    fill_colour="rgba(90,90,90,0.3)",
    row=1,
    col=1,
    linewidth_value=3,
)
for t in range(1, 4):
    mean_y = mean_over_partic[t, :, 0]
    y_upper = mean_y + mean_sd_over_partic[t, :, 0]
    y_lower = mean_y - mean_sd_over_partic[t, :, 0]
    mean_plot = line_sd_band_plot(
        mean_plot,
        mean_y,
        y_upper,
        y_lower,
        xdata=np.arange(0, 101, 1),
        nameval=f"Trial {t}",
        line_colour=p_colours[t - 1],
        fill_colour="rgba(255,0,0,0.0)",
        row=1,
        col=1,
        linewidth_value=3,
    )
mean_plot.update_layout(
    width=750,
    height=525,
    title_text="",
    template="simple_white",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.4,
        x=0.3,
    ),
    font_family="Arial, sans-serif",
    font_size=18,
)
mean_plot.update_yaxes(title_text="Angles (\u00b0)", row=1, col=1)
mean_plot.update_xaxes(title_text="Gait Cycle (%)", row=1, col=1)
mean_plot.show()
fname = f"{proc_data_path}results/mean_waveforms_first_ten.svg"

mean_plot.write_image(fname)

# // Point Plots
wfrm_point_plot = make_subplots(
    rows=3,
    cols=1,
    vertical_spacing=0.09,
    subplot_titles=[
        "Minimum Knee Angle: Early Stance",
        "Maximum Knee Angle: Stance",
        "Minimum Knee Angle: Swing",
    ],
)

varname = ["peak_KSA_stance", "min_KSA_stance", "peak_KSA_swing"]

for k in range(3):  # loop over outcomes
    var = summary_data_discrete["joint"]["rknee"][varname[k]]
    for t in [1, 2, 3]:  # look over trials
        pts_arr = []
        for p in range(len(var)):
            # !! REMOVED p17 trial 3
            if (p == len(var) - 1) and (t == 3):
                continue
            pts = np.stack(var[p][t] - np.mean(np.stack(var[p][0])))

            # 10-stride moving average using Convolvution with array of ones of length window
            window = 10
            pts_ten = np.convolve(pts, np.ones(window), "valid") / window

            pts_arr.append(pts_ten)

        longestarray = max(map(len, pts_arr))
        for j, i in enumerate(pts_arr):
            if i.shape[0] < longestarray:
                padlen = longestarray - i.shape[0]
                pts_arr[j] = np.pad(
                    i, (0, padlen), mode="constant", constant_values=np.nan
                )

        pts_arr_mean = np.nanmean(np.stack(pts_arr), axis=0)
        wfrm_point_plot.add_trace(
            go.Scatter(
                x=np.arange(0, pts_arr_mean.shape[0], 1),
                y=pts_arr_mean,
                mode="lines",
                line=dict(color=p_colours[t - 1], width=1),
                name=f"Trial {t}",
                showlegend=True if k == 0 else False,
            ),
            row=k + 1,
            col=1,
        )

wfrm_point_plot.update_yaxes(
    title_text="Difference in Angle (\u00b0): Trial(n) - Baseline", row=2, col=1
)
wfrm_point_plot.update_xaxes(title_text="Stride #", row=3, col=1)
wfrm_point_plot.update_layout(
    height=1000,
    width=700,
    title_text="",
    template="simple_white",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        x=0.3,
    ),
    font_family="Arial, sans-serif",
    font_size=18,
)
wfrm_point_plot.update_annotations(font=dict(size=20, family="Sans-serif"))
wfrm_point_plot.show()
fname = f"{proc_data_path}results/discrete_knee_angle_3panel.svg"
wfrm_point_plot.write_image(fname)
