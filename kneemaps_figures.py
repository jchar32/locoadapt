# %% 3x2 panel by knee diagram plots for manuscript
import numpy as np
import pickle
import pandas as pd
import plotly.graph_objects as go
import os

proc_data_path = "./experiment_two/"  # // either "experiment_one" or "experiment_two"
with open(f"{proc_data_path}all_maps_frontal.pkl", "rb") as f:
    all_fr_img_stack = pickle.load(f)
with open(f"{proc_data_path}all_maps_transverse.pkl", "rb") as f:
    all_tr_img_stack = pickle.load(f)

moments = pd.read_csv(os.path.join(proc_data_path, "all_maps_moments.csv"))

num_conditions = 3

# %%
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

moment_df["views"] = moment_df["views"].replace({"Page 1": "fr", "Page 2": "tr"})


color_scale_custom = [
    "rgb(255,240,160)",
    "rgb(255,180,0)",
    "rgb(255,110,60)",
    "rgb(255,0,0)",
    "rgb(130,0,0)",
    "rgb(50,0,0)",
    "rgb(0,0,0)",
]

view = ["fr", "tr"]
for v in view:
    for c in range(num_conditions):
        x_pos = moment_df.loc[
            (moment_df["views"] == v) & (moment_df["conds"] == str(c + 1)), "x_pos"
        ]
        y_pos = moment_df.loc[
            (moment_df["views"] == v) & (moment_df["conds"] == str(c + 1)), "y_pos"
        ]
        ar_ = (
            all_fr_img_stack[:, c, :, :] if v == "fr" else all_tr_img_stack[:, c, :, :]
        )
        sum_map = np.sum(ar_, axis=0)

        sum_map = sum_map.astype(float)
        sum_map[sum_map == 0] = np.nan
        masked_map = sum_map.copy()
        heat_map = go.Figure()
        heat_map.add_trace(
            go.Heatmap(
                z=masked_map,
                coloraxis="coloraxis",
                reversescale=True,
                showlegend=False,
                showscale=False,
                opacity=0.7,
            ),
        )
        heat_map.add_trace(
            go.Scatter(
                x=np.array([x_pos.mean()]),
                y=np.array([y_pos.mean()]),
                error_x=dict(
                    type="data",
                    array=np.array([x_pos.std()]),
                    visible=True,
                    thickness=2,
                    width=8,
                    color="black",
                ),
                error_y=dict(
                    type="data",
                    array=np.array([y_pos.std()]),
                    visible=True,
                    thickness=2,
                    width=8,
                    color="black",
                ),
                mode="markers",
                marker=dict(size=15, color="black"),
                showlegend=False,
            )
        )
        if v == "fr":
            ax_names = ["x", "z"]
            ax_dir = [1, 1]
        else:
            ax_names = ["x", "y"]
            ax_dir = [1, -1]
        heat_map.add_trace(
            go.Scatter(
                x=np.array([x_pos.mean()]),
                y=np.array([y_pos.mean() + 200]),
                mode="text",
                text=[
                    f"({ax_names[0]}: {ax_dir[0]*x_pos.mean():.2f}, {ax_names[1]}: {ax_dir[1]*y_pos.mean():.2f})"
                ],
                showlegend=False,
                textfont=dict(family="Arial", size=38, color="black"),
            )
        )
        heat_map.update_xaxes(
            range=[0, 1875],
            showline=False,
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        )
        heat_map.update_yaxes(
            range=[0, 1875],
            showline=False,
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        )
        heat_map.update_layout(
            autosize=True,
            width=600,
            height=600,
            yaxis=dict(autorange="reversed"),
            coloraxis=dict(
                showscale=False,
                colorscale=color_scale_custom,
                cmin=0,
                cmax=17,
                colorbar=dict(
                    tick0=0,
                    ticklabelstep=1,
                    tickmode="array",
                    tickvals=np.arange(0, 17, 1),
                    orientation="h",
                    tickfont=dict(family="Arial, sans-serif", size=20, color="black"),
                ),
            ),
            showlegend=False,
            legend=dict(orientation="v", font=dict(size=18)),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0),
        )

        heat_map.show()

        heat_map.write_image(f"{proc_data_path}results/kneemap_{v}_{c+1}.svg")
