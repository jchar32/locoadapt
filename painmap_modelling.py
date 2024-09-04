# %%
from utils import fitting
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pickle


proc_data_path = "./experiment_two/"


with open(f"{proc_data_path}pain_current_fits.pkl", "rb") as f:
    p_data = pickle.load(f)
p_colors = px.colors.qualitative.Vivid + px.colors.qualitative.Plotly
grey_color = "#808080"
color_list = [grey_color] * 20


#  %% Get exponential summary statistics across participants
exp_params = []
for i, p in enumerate(p_data.keys()):
    if p == "all":
        continue
    exp_params.append(p_data[p]["efit_norm"][0])
mean_exp_params = np.mean(np.vstack(exp_params), axis=0)
median_exp_params = np.median(np.vstack(exp_params), axis=0)
std_exp_params = np.vstack(exp_params).std(axis=0)

# Get linear summary statistics across participants
lin_params = []
for i, p in enumerate(p_data.keys()):
    if p == "all":
        continue
    lin_params.append(p_data[p]["lfit_norm"][0])
mean_lin_params = np.mean(np.vstack(lin_params), axis=0)
median_lin_params = np.median(np.vstack(lin_params), axis=0)
std_lin_params = np.vstack(lin_params).std(axis=0)

# Get piecewise summary statistics across participants
pwise_params = []
for i, p in enumerate(p_data.keys()):
    if p == "all":
        continue
    pwise_params.append(p_data[p]["pwise_norm"][0])
mean_pwise_params = np.mean(np.vstack(pwise_params), axis=0)
median_pwise_params = np.median(np.vstack(pwise_params), axis=0)
std_pwise_params = np.vstack(pwise_params).std(axis=0)


# %% Plot Data
fig_sup = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=(
        "Representative Participants",
        f"All Participants n={len(p_data.keys())-1}",
    ),
    shared_xaxes=True,
    shared_yaxes=True,
    horizontal_spacing=0.05,
    x_title="Normalized Current (mA/max(mA))",
    y_title="Pain Intensity",
)
fig = go.Figure()

fit_adj_r2_df = pd.DataFrame(
    np.stack(
        [
            p_data["all"]["exp_adjusted_r2_exp"],
            p_data["all"]["lin_adjusted_r2_lin"],
            p_data["all"]["pwise_adjusted_r2_pwise"],
        ],
        axis=1,
    ),
    columns=["Exp", "Lin", "Pwise"],
)
pwise_fit_params_median = p_data["all"]["pwise_fit_params_median"]
representative_participants = [
    "p07",
    "p12",
]

for i, p in enumerate(p_data.keys()):
    if p == "all":
        continue
    xdata = p_data[p]["data"]["mA_norm"]
    ydata = p_data[p]["data"]["painreport"]
    linfitparams = p_data[p]["lfit_norm"]
    expfitparams = p_data[p]["efit_norm"]

    # Function to plot pain as function of current with fitting results
    # min and max x values
    xmin = min(xdata)
    xmax = max(xdata)

    # build figure
    fig.add_trace(
        go.Scatter(
            x=xdata,
            y=ydata,
            mode="markers",
            name=str(p),
            marker=dict(size=10, color=grey_color),
            marker_opacity=0.1,
            showlegend=False,
        )
    )
    fig_sup.add_trace(
        go.Scatter(
            x=xdata,
            y=ydata,
            mode="markers",
            name=str(p),
            marker=dict(size=10, color=grey_color),
            marker_opacity=0.1,
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    if p in representative_participants:
        fig_sup.add_trace(
            go.Scatter(
                x=xdata,
                y=ydata,
                mode="markers",
                name=str(p),
                marker=dict(size=10, color=p_colors[i]),
                marker_opacity=0.2,
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        # individual participant fits
        fig_sup.add_trace(
            go.Scatter(
                x=np.linspace(xmin, xmax, xdata.shape[0]),
                y=fitting.linear(
                    np.linspace(xmin, xmax, xdata.shape[0]), *linfitparams[0]
                ),
                mode="lines",
                name=str(p) + "Linear",
                line=dict(dash="solid", width=4, color=p_colors[i]),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig_sup.add_trace(
            go.Scatter(
                x=np.linspace(xmin, xmax, xdata.shape[0]),
                y=fitting.exponential(
                    np.linspace(xmin, xmax, xdata.shape[0]), *expfitparams[0]
                ),
                mode="lines",
                name=str(p) + "Exponential",
                line=dict(dash="dot", width=4, color=p_colors[i]),
                showlegend=False,
            ),
            row=1,
            col=1,
        )


fig_sup.add_trace(
    go.Scatter(
        x=np.linspace(
            np.min(p_data["all"]["data"]["mA_norm"]),
            np.max(p_data["all"]["data"]["mA_norm"]),
            p_data["all"]["data"]["mA_norm"].shape[0],
        ),
        y=fitting.linear(
            np.linspace(
                np.min(p_data["all"]["data"]["mA_norm"]),
                np.max(p_data["all"]["data"]["mA_norm"]),
                p_data["all"]["data"]["mA_norm"].shape[0],
            ),
            *median_lin_params,
        ),
        mode="lines",
        name=f"Linear Adj R\u00b2 = {np.round(fit_adj_r2_df['Lin'].mean(),3)} ({np.round(fit_adj_r2_df['Lin'].std(),3)})",
        line=dict(dash="solid", width=3, color="black"),
        showlegend=True,
    ),
    row=1,
    col=2,
)

fig_sup.add_trace(
    go.Scatter(
        x=np.linspace(
            np.min(p_data["all"]["data"]["mA_norm"]),
            np.max(p_data["all"]["data"]["mA_norm"]),
            p_data["all"]["data"]["mA_norm"].shape[0],
        ),
        y=fitting.exponential(
            np.linspace(
                np.min(p_data["all"]["data"]["mA_norm"]),
                np.max(p_data["all"]["data"]["mA_norm"]),
                p_data["all"]["data"]["mA_norm"].shape[0],
            ),
            *median_exp_params,
        ),
        mode="lines",
        name=f"Exponential Adj R\u00b2 = {np.round(fit_adj_r2_df['Exp'].mean(),3)} ({np.round(fit_adj_r2_df['Exp'].std(),3)})",
        line=dict(dash="dot", width=4, color="black"),
        showlegend=True,
    ),
    row=1,
    col=2,
)

# // Piecewise linear fit
fig_sup.add_trace(
    go.Scatter(
        x=np.linspace(
            np.min(p_data["all"]["data"]["mA_norm"]),
            np.max(p_data["all"]["data"]["mA_norm"]),
            p_data["all"]["data"]["mA_norm"].shape[0],
        ),
        y=fitting.piecewise_linear(
            np.linspace(
                np.min(p_data["all"]["data"]["mA_norm"]),
                np.max(p_data["all"]["data"]["mA_norm"]),
                p_data["all"]["data"]["mA_norm"].shape[0],
            ),
            *pwise_fit_params_median,
        ),
        mode="lines",
        name=f"Piecewise Adj R\u00b2 = {np.round(fit_adj_r2_df['Pwise'].mean(),3)} ({np.round(fit_adj_r2_df['Pwise'].std(),3)})",
        line=dict(dash="dash", width=3, color="black"),
        showlegend=True,
    ),
    row=1,
    col=2,
)

fig_sup.update_layout(
    width=1200,
    height=500,
    template="simple_white",
    showlegend=True,
    font_size=20,
    font_family="Arial, sans-serif",
    legend=dict(
        orientation="h", yanchor="top", y=1.3, xanchor="left", x=0.0, entrywidth=300
    ),
)
fig_sup.update_yaxes(range=[-0.5, 6])
fig_sup.update_annotations(font_size=20, font_family="Arial, sans-serif")

fig_sup.show()

fig_sup.write_image(f"{proc_data_path}results/pain_current_plot_allp_2panel.svg")
