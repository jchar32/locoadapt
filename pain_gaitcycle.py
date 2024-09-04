# %%
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pickle

proc_data_path = "./experiment_two/"

n_strides_2keep = 75

# %% Calculate summary data
with open(f"{proc_data_path}pain_traces_50spm.pkl", "rb") as f:
    all_traces = pickle.load(f)

# // Create plots
# Compile traces per participant into ndarray
traces = {p: np.ndarray for p in all_traces.keys()}
for p in all_traces.keys():
    traces[p] = np.stack(all_traces[p]["nd"][:n_strides_2keep])
traces_list = [traces_all for traces_all in traces.values()]
traces_arr = np.array(traces_list)
mean_traces = traces_arr.mean(axis=0)

# // 3 Panel Plot
p_colours = px.colors.qualitative.Bold[:2]

p_colours_faded = [
    "rgba(127, 60, 141, 0.05)",
    "rgba(17, 165, 121, 0.05)",
]
p_colours_semi_faded = [
    "rgba(127, 60, 141, 0.3)",
    "rgba(17, 165, 121, 0.3)",
]
grey_individual_colours = "rgba(80,80,80,0.4)"

paintrace_fig = make_subplots(
    rows=1,
    cols=3,
    shared_yaxes=False,
    subplot_titles=(
        "Representative data",
        "Segmented and Normalized",
        "All Participant Ensembles",
    ),
    horizontal_spacing=0.1,
    vertical_spacing=0.1,
)

paintrace_fig.update_yaxes(range=[0, 5.5], row=1, col=1)
paintrace_fig.update_yaxes(range=[0, 5.5], row=1, col=2)
paintrace_fig.update_yaxes(range=[0, 75], row=1, col=3)

# // Left Panel -  representative data
trace1_partic = "p07"
trace1_colour_id = p_colours[0]
trace1 = all_traces[trace1_partic]["full"][
    all_traces[trace1_partic]["events"].hs[0] : all_traces[trace1_partic]["events"].hs[
        -1
    ]
]
paintrace_fig.add_trace(
    go.Scatter(
        x=np.linspace(
            0,
            trace1.shape[0] * (1 / 200),
            trace1.shape[0],
        ),
        y=trace1,
        name=trace1_partic,
        mode="lines",
        line=dict(dash="solid", color=trace1_colour_id, width=2),
        showlegend=False,
    ),
    row=1,
    col=1,
)


trace2_partic = "p11"
trace2_colour_id = p_colours[1]
trace2 = all_traces[trace2_partic]["full"][
    all_traces[trace2_partic]["events"].hs[0] : all_traces[trace2_partic]["events"].hs[
        -1
    ]
]
paintrace_fig.add_trace(
    go.Scatter(
        x=np.linspace(
            0,
            trace2.shape[0] * (1 / 200),
            trace2.shape[0],
        ),
        y=trace2,
        name=trace2_partic,
        mode="lines",
        line=dict(dash="solid", color=trace2_colour_id, width=2),
        showlegend=False,
    ),
    row=1,
    col=1,
)
paintrace_fig.update_xaxes(title_text="Time (s)", row=1, col=1)
paintrace_fig.update_yaxes(title_text="Pain Intensity (0-10)", row=1, col=1)


# // Middle Panel - Gait Cycle Normalized
for i in all_traces[trace1_partic]["nd"]:
    paintrace_fig.add_trace(
        go.Scatter(
            x=np.linspace(
                0,
                100,
                101,
            ),
            y=i,
            name=trace1_partic,
            mode="lines",
            line=dict(dash="solid", color=grey_individual_colours, width=0.5),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
for i in all_traces[trace2_partic]["nd"]:
    paintrace_fig.add_trace(
        go.Scatter(
            x=np.linspace(
                0,
                100,
                101,
            ),
            y=i,
            name=trace2_partic,
            mode="lines",
            line=dict(dash="solid", color=grey_individual_colours, width=0.5),
            showlegend=False,
        ),
        row=1,
        col=2,
    )


# mean traces with colour bands
trace1_mean = np.mean(np.stack(all_traces[trace1_partic]["nd"]), axis=0)
trace1_std = np.std(np.stack(all_traces[trace1_partic]["nd"]), axis=0)
trace2_mean = np.mean(np.stack(all_traces[trace2_partic]["nd"]), axis=0)
trace2_std = np.std(np.stack(all_traces[trace2_partic]["nd"]), axis=0)
paintrace_fig.add_trace(
    go.Scatter(
        x=np.linspace(
            0,
            100,
            101,
        ),
        y=trace1_mean,
        mode="lines",
        line=dict(color=p_colours[0], width=5),
        showlegend=False,
    ),
    row=1,
    col=2,
)
paintrace_fig.add_trace(
    go.Scatter(
        x=np.linspace(
            0,
            100,
            101,
        ),
        y=trace1_mean + trace1_std,
        mode="lines",
        marker=dict(color=p_colours[0]),
        line=dict(width=0),
        showlegend=False,
    ),
    row=1,
    col=2,
)
paintrace_fig.add_trace(
    go.Scatter(
        x=np.linspace(
            0,
            100,
            101,
        ),
        y=trace1_mean - trace1_std,
        fill="tonexty",
        mode="lines",
        marker=dict(color=p_colours[0]),
        line=dict(width=0),
        fillcolor=p_colours_semi_faded[0],
        showlegend=False,
    ),
    row=1,
    col=2,
)

paintrace_fig.add_trace(
    go.Scatter(
        x=np.linspace(
            0,
            100,
            101,
        ),
        y=trace2_mean,
        mode="lines",
        line=dict(color=p_colours[1], width=5),
        showlegend=False,
    ),
    row=1,
    col=2,
)
paintrace_fig.add_trace(
    go.Scatter(
        x=np.linspace(
            0,
            100,
            101,
        ),
        y=trace2_mean + trace2_std,
        mode="lines",
        marker=dict(color=p_colours[1]),
        line=dict(width=0),
        showlegend=False,
    ),
    row=1,
    col=2,
)
paintrace_fig.add_trace(
    go.Scatter(
        x=np.linspace(
            0,
            100,
            101,
        ),
        y=trace2_mean - trace2_std,
        fill="tonexty",
        mode="lines",
        marker=dict(color=p_colours[1]),
        line=dict(width=0),
        fillcolor=p_colours_semi_faded[1],
        showlegend=False,
    ),
    row=1,
    col=2,
)
paintrace_fig.update_xaxes(title_text="Gait cycle (%)", row=1, col=2)


# // Right Panel - All Participant Ensembles

paintrace_fig.add_trace(
    go.Heatmap(z=mean_traces, showlegend=False, showscale=True, colorscale="Plasma"),
    row=1,
    col=3,
)
paintrace_fig.update_layout(
    coloraxis=dict(
        colorscale="Plasma",
        cmin=1.5,
        cmax=3.5,
        colorbar=dict(
            title="Pain Intensity (0-10)",
            tickfont=dict(family="Arial, sans-serif", size=22, color="black"),
        ),
    )
)

paintrace_fig.update_yaxes(title_text="Strides (n)", title_standoff=0.1, row=1, col=3)
paintrace_fig.update_xaxes(title_text="Gait cycle (%)", row=1, col=3)

paintrace_fig.update_layout(
    width=1200,
    height=600,
    template="simple_white",
    title="Pain Intensity by Gait Cycle",
    font_family="Arial, sans-serif",
    font_size=22,
)
paintrace_fig.update_annotations(font=dict(size=25, family="Arial, sans-serif"))

paintrace_fig.show()
paintrace_fig.write_image(f"{proc_data_path}results/pain_modulation_3panel.svg")

# Inset of modulated participant
trace1_partic = "p08"
trace1_colour_id = "rgba(229,30,57,1)"
trace1_colour_id_alpha = "rgba(229,30,57,0.4)"
trace1_mean = np.mean(np.stack(all_traces[trace1_partic]["nd"]), axis=0)
trace1_std = np.std(np.stack(all_traces[trace1_partic]["nd"]), axis=0)
paintrace_fig_modulator = make_subplots(
    rows=1,
    cols=1,
    shared_yaxes=False,
    horizontal_spacing=0.1,
    vertical_spacing=0.1,
)

paintrace_fig_modulator.update_yaxes(range=[0, 5.5], row=1, col=1)

for i in all_traces[trace1_partic]["nd"]:
    paintrace_fig_modulator.add_trace(
        go.Scatter(
            x=np.linspace(
                0,
                100,
                101,
            ),
            y=i,
            name=trace1_partic,
            mode="lines",
            line=dict(dash="solid", color=grey_individual_colours, width=0.5),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

paintrace_fig_modulator.add_trace(
    go.Scatter(
        x=np.linspace(
            0,
            100,
            101,
        ),
        y=trace1_mean,
        mode="lines",
        line=dict(color=trace1_colour_id, width=5),
        showlegend=False,
    ),
    row=1,
    col=1,
)
paintrace_fig_modulator.add_trace(
    go.Scatter(
        x=np.linspace(
            0,
            100,
            101,
        ),
        y=trace1_mean + trace1_std,
        mode="lines",
        marker=dict(color=trace1_colour_id),
        line=dict(width=0),
        showlegend=False,
    ),
    row=1,
    col=1,
)
paintrace_fig_modulator.add_trace(
    go.Scatter(
        x=np.linspace(
            0,
            100,
            101,
        ),
        y=trace1_mean - trace1_std,
        fill="tonexty",
        mode="lines",
        marker=dict(color=trace1_colour_id),
        line=dict(width=0),
        fillcolor=trace1_colour_id_alpha,
        showlegend=False,
    ),
    row=1,
    col=1,
)

paintrace_fig_modulator.update_layout(
    width=600,
    height=600,
    template="simple_white",
    title="Pain Intensity by Gait Cycle",
    font_family="Arial, sans-serif",
    font_size=40,
)

paintrace_fig_modulator.show()
paintrace_fig_modulator.write_image(f"{proc_data_path}results/pain_modulator_inset.svg")
