import copy
from operator import itemgetter

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

import networkx as nx


def node_trace(x, y, text: str, size: int):
    return go.Scatter(
        x=(x,), y=(y,),
        mode='markers',
        text=text,
        # hovertext=hover_data,
        hoverinfo='text',
        marker=dict(
            showscale=False,
            # colorscale options
            # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlGnBu',
            reversescale=True,
            color='#B0BEC5',
            size=min(size, 40),
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))


def plot_network(g: nx.DiGraph, title: str = "Topic Evolution Network", width: int = 650, height: int = 650) -> go.Figure:
    nodes_ = list()
    for n_id in g.nodes:
        t_id, t_name, (x, y), t_size, bin_str = itemgetter("ID", "name", "pos", "size", "bin")(g.nodes[n_id])
        nodes_.append([t_id, t_name, t_size, x, y, bin_str])
    nodes_df = pd.DataFrame(nodes_, columns=["ID", "name", "size", "x", "y", "bin"])

    fig1 = px.scatter(nodes_df, x="x", y="y", size="size", size_max=40, template="simple_white", labels={"x": "", "y": "", "bin": "Timerange"},
                      hover_data={"ID": True, "name": True, "size": True, "bin": True, "x": False, "y": False}, color="bin")
    fig1.update_traces(marker=dict(line=dict(width=2, color='DarkSlateGrey')))

    # Update hover order
    fig1.update_traces(hovertemplate="<br>".join(["<b>Topic %{customdata[0]}</b>",
                                                  "%{customdata[1]}",
                                                  "Size: %{customdata[2]}",
                                                  "Time range: %{customdata[3]}"]))

    bin_edge_data = dict()
    init_d = dict(edge_x=list(),
                  edge_y=list(),
                  edge_w=list())
    for u, v in g.out_edges:
        u_node = g.nodes[u]
        v_node = g.nodes[v]
        ux, uy = u_node['pos']
        vx, vy = v_node['pos']
        bin_node = u_node["bin"]
        bin_d = bin_edge_data.setdefault(bin_node, copy.deepcopy(init_d))
        edge_x = bin_d["edge_x"]
        edge_y = bin_d["edge_y"]
        edge_w = bin_d["edge_w"]

        sim = g[u][v]["w"]
        edge_x.append(ux)
        edge_x.append(vx)
        edge_x.append(None)
        edge_y.append(uy)
        edge_y.append(vy)
        edge_y.append(None)
        edge_w.append(sim)

    edge_traces = list()
    for bin_name, data in bin_edge_data.items():
        edge_x, edge_y = data["edge_x"], data["edge_y"]
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            name=bin_name,
            mode='lines',
            legendgroup=bin_name)

        edge_traces.append(edge_trace)

    # fig = go.Figure(data=[*edge_traces, *fig1.data])

    fig1.add_traces(edge_traces)  # , secondary_ys=[True] * len(edge_traces)
    # Stylize layout
    fig1.update_layout(
        title={
            'text': f"{title}",
            'y': .95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        },
        width=width,
        height=height,
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
        xaxis={"visible": False},
        yaxis={"visible": False},
        # sliders=sliders
    )

    return fig1
