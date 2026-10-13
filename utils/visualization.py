import plotly.graph_objects as go


# ============================================================
# 1. VISUALISASI SEBARAN NODE
# ============================================================

def plot_nodes(df, home=0, title="Node Distribution"):
    """
    Membuat visualisasi posisi seluruh node.

    Parameter
    ---------
    df : DataFrame
        Data dengan kolom: node, x, y.

    home : int
        Index node yang digunakan sebagai titik awal.

    title : str
        Judul grafik.

    Return
    ------
    plotly.graph_objects.Figure
    """

    fig = go.Figure()

    # Semua node
    colors = [
        "#B23A48" if i == home else "#4F8FA3"
        for i in range(len(df))
    ]

    fig.add_trace(
        go.Scatter(
            x=df["x"],
            y=df["y"],
            mode="markers+text",
            text=df["node"].astype(str),
            textposition="top center",
            marker=dict(
                size=14,
                color=colors,
                line=dict(
                    color="white",
                    width=2
                )
            ),
            hovertemplate=(
                "Node: %{text}<br>"
                "X: %{x}<br>"
                "Y: %{y}"
                "<extra></extra>"
            )
        )
    )

    fig.update_layout(
        title=title,
        height=450,
        margin=dict(
            l=20,
            r=20,
            t=50,
            b=20
        ),
        xaxis_title="X",
        yaxis_title="Y",
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1
        ),
        showlegend=False
    )

    return fig


# ============================================================
# 2. VISUALISASI RUTE
# ============================================================

def plot_route(
    df,
    tour,
    home=0,
    title="TSP Route",
    show_arrows=True
):
    """
    Membuat visualisasi rute TSP.

    Parameter
    ---------
    df : DataFrame
        Data dengan kolom node, x, y.

    tour : list
        Urutan node yang dikunjungi.
        Contoh: [0, 2, 4, 1, 3, 0]

    home : int
        Index node awal.

    title : str
        Judul grafik.

    show_arrows : bool
        Menampilkan arah perjalanan.

    Return
    ------
    plotly.graph_objects.Figure
    """

    # Ambil koordinat berdasarkan index node
    xs = [df.iloc[i]["x"] for i in tour]
    ys = [df.iloc[i]["y"] for i in tour]

    fig = go.Figure()

    # --------------------------------------------------------
    # Garis rute
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(
                color="#2D120D",
                width=3
            ),
            hoverinfo="skip",
            name="Route"
        )
    )

    # --------------------------------------------------------
    # Arah perjalanan
    # --------------------------------------------------------

    if show_arrows and len(tour) <= 80:

        for i in range(len(tour) - 1):

            x_start = xs[i]
            y_start = ys[i]

            x_end = xs[i + 1]
            y_end = ys[i + 1]

            fig.add_annotation(
                x=x_end,
                y=y_end,
                ax=x_start,
                ay=y_start,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=1,
                arrowwidth=1.5,
                arrowcolor="#6B0B0C",
                standoff=8
            )

    # --------------------------------------------------------
    # Node
    # --------------------------------------------------------

    node_colors = [
        "#B23A48" if i == home else "#4F8FA3"
        for i in range(len(df))
    ]

    fig.add_trace(
        go.Scatter(
            x=df["x"],
            y=df["y"],
            mode="markers+text",
            text=df["node"].astype(str),
            textposition="top center",
            marker=dict(
                size=14,
                color=node_colors,
                line=dict(
                    color="white",
                    width=2
                )
            ),
            hovertemplate=(
                "Node: %{text}<br>"
                "X: %{x}<br>"
                "Y: %{y}"
                "<extra></extra>"
            ),
            name="Node"
        )
    )

    fig.update_layout(
        title=title,
        height=500,
        margin=dict(
            l=20,
            r=20,
            t=50,
            b=20
        ),
        xaxis_title="X",
        yaxis_title="Y",
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


# ============================================================
# 3. VISUALISASI PERBANDINGAN RUTE
# ============================================================

def plot_route_comparison(
    df,
    initial_tour,
    final_tour,
    home=0,
    initial_title="Initial Route",
    final_title="Final Route"
):
    """
    Menghasilkan dua Figure:
    - rute awal
    - rute akhir

    Dipakai oleh app.py untuk ditampilkan berdampingan.
    """

    initial_fig = plot_route(
        df=df,
        tour=initial_tour,
        home=home,
        title=initial_title
    )

    final_fig = plot_route(
        df=df,
        tour=final_tour,
        home=home,
        title=final_title
    )

    return initial_fig, final_fig


# ============================================================
# 4. VISUALISASI HISTORY / ITERASI
# ============================================================

def plot_iteration(
    df,
    tour,
    iteration,
    home=0
):
    """
    Membuat visualisasi rute pada suatu iterasi.
    """

    return plot_route(
        df=df,
        tour=tour,
        home=home,
        title=f"Iteration {iteration}"
    )
