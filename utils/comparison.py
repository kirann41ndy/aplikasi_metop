import pandas as pd
import plotly.graph_objects as go


# Warna grafik
CHART_RED = "#6B0B0C"       # Final Distance & Execution Time
CHART_YELLOW = "#FFFBCA"    # Solution Improvement


# ============================================================
# 1. HITUNG IMPROVEMENT
# ============================================================

def calculate_improvement(initial_distance, final_distance):
    """
    Menghitung persentase perbaikan jarak.

    Improvement (%) =
        ((Initial - Final) / Initial) * 100

    Return None jika metode tidak memiliki initial solution
    (misalnya metode Constructive yang membangun rute dari awal).
    """

    if initial_distance is None:
        return None

    if initial_distance == 0:
        return 0.0

    return (
        (initial_distance - final_distance)
        / initial_distance
    ) * 100


# ============================================================
# 2. MEMBUAT TABEL COMPARISON
# ============================================================

def create_comparison_table(results):
    """
    Membuat DataFrame hasil perbandingan.

    results berupa list dictionary.

    Contoh:
    [
        {
            "method": "Nearest Neighbor",
            "initial_distance": None,      # Constructive: tidak ada initial
            "final_distance": 120,
            "execution_time": 0.002
        },
        {
            "method": "2-opt",
            "initial_distance": 150,
            "final_distance": 120,
            "execution_time": 0.004
        }
    ]

    Initial Distance dan Improvement (%) bernilai kosong (NaN)
    untuk metode yang tidak memiliki initial solution.
    """

    rows = []

    for result in results:

        initial_raw = result.get("initial_distance")

        initial = (
            None
            if initial_raw is None
            else float(initial_raw)
        )

        final = float(
            result.get("final_distance", 0)
        )

        time_value = float(
            result.get("execution_time", 0)
        )

        improvement = calculate_improvement(
            initial,
            final
        )

        rows.append(
            {
                "Method": result.get(
                    "method",
                    "Unknown"
                ),

                "Initial Distance": (
                    None
                    if initial is None
                    else round(initial, 2)
                ),

                "Final Distance": round(
                    final,
                    2
                ),

                "Improvement (%)": (
                    None
                    if improvement is None
                    else round(improvement, 2)
                ),

                "Execution Time (ms)": round(
                    time_value * 1000,
                    3
                )
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 3. SORTING HASIL
# ============================================================

def sort_by_distance(comparison_df):
    """
    Mengurutkan hasil berdasarkan final distance.
    Jarak paling kecil berada di atas.
    """

    return comparison_df.sort_values(
        by="Final Distance",
        ascending=True
    ).reset_index(drop=True)


def sort_by_time(comparison_df):
    """
    Mengurutkan hasil berdasarkan waktu eksekusi.
    Waktu paling cepat berada di atas.
    """

    return comparison_df.sort_values(
        by="Execution Time (ms)",
        ascending=True
    ).reset_index(drop=True)


# ============================================================
# 4. METODE TERBAIK BERDASARKAN DISTANCE
# ============================================================

def get_best_distance(comparison_df):
    """
    Mengambil metode dengan final distance terkecil.
    """

    if comparison_df.empty:
        return None

    idx = comparison_df[
        "Final Distance"
    ].idxmin()

    return comparison_df.loc[idx]


# ============================================================
# 5. METODE TERCEPAT
# ============================================================

def get_fastest_method(comparison_df):
    """
    Mengambil metode dengan waktu eksekusi tercepat.
    """

    if comparison_df.empty:
        return None

    idx = comparison_df[
        "Execution Time (ms)"
    ].idxmin()

    return comparison_df.loc[idx]


# ============================================================
# 6. GRAFIK FINAL DISTANCE
# ============================================================

def plot_distance_comparison(comparison_df):
    """
    Grafik perbandingan final distance.
    Semakin kecil semakin baik.
    """

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=comparison_df["Method"],
            y=comparison_df["Final Distance"],
            text=comparison_df["Final Distance"],
            textposition="auto",
            marker_color=CHART_RED
        )
    )

    fig.update_layout(
        title="Final Distance Comparison",
        xaxis_title="Method",
        yaxis_title="Final Distance",
        height=400,
        margin=dict(
            l=20,
            r=20,
            t=50,
            b=80
        )
    )

    return fig


# ============================================================
# 7. GRAFIK EXECUTION TIME
# ============================================================

def plot_time_comparison(comparison_df):
    """
    Grafik perbandingan waktu eksekusi.
    Semakin kecil semakin cepat.
    """

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=comparison_df["Method"],
            y=comparison_df["Execution Time (ms)"],
            text=comparison_df["Execution Time (ms)"],
            textposition="auto",
            marker_color=CHART_RED
        )
    )

    fig.update_layout(
        title="Execution Time Comparison",
        xaxis_title="Method",
        yaxis_title="Time (ms)",
        height=400,
        margin=dict(
            l=20,
            r=20,
            t=50,
            b=80
        )
    )

    return fig


# ============================================================
# 8. GRAFIK IMPROVEMENT
# ============================================================

def plot_improvement_comparison(comparison_df):
    """
    Grafik persentase improvement.

    Semakin besar improvement,
    semakin besar penurunan jarak dari initial solution.

    Hanya metode yang memiliki initial solution yang ditampilkan.
    """

    data = comparison_df.dropna(
        subset=["Improvement (%)"]
    )

    fig = go.Figure()

    if data.empty:

        fig.add_annotation(
            text=(
                "Tidak ada metode dengan initial solution "
                "(improvement tidak dapat dihitung)."
            ),
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False
        )

    else:

        fig.add_trace(
            go.Bar(
                x=data["Method"],
                y=data["Improvement (%)"],
                text=[
                    f"{x:.2f}%"
                    for x in data["Improvement (%)"]
                ],
                textposition="auto",
                marker_color=CHART_YELLOW
            )
        )

    fig.update_layout(
        title="Solution Improvement",
        xaxis_title="Method",
        yaxis_title="Improvement (%)",
        height=400,
        margin=dict(
            l=20,
            r=20,
            t=50,
            b=80
        )
    )

    return fig
