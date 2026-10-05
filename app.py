import streamlit as st
import pandas as pd
import random
import time

from algorithms.base import (
    build_distance_matrix,
    tour_distance,
    generate_initial_tour
)

from algorithms.constructive import (
    nearest_neighbor,
    nearest_insertion,
    farthest_insertion,
    arbitrary_insertion
)

from algorithms.local_search import (
    two_opt,
    three_opt
)

from algorithms.simulated_annealing import (
    simulated_annealing
)

from algorithms.tabu_search import (
    tabu_search
)

from utils.load_data import load_data

from utils.visualization import (
    plot_nodes,
    plot_route,
    plot_route_comparison
)

from utils.comparison import (
    create_comparison_table,
    sort_by_distance,
    get_best_distance,
    get_fastest_method,
    plot_distance_comparison,
    plot_time_comparison,
    plot_improvement_comparison
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="TSP Learning & Optimization",
    page_icon="🧭",
    layout="wide"
)


# ============================================================
# ALGORITHM CONFIGURATION
# ============================================================

ALGORITHMS = {

    # --------------------------------------------------------
    # CONSTRUCTIVE
    # --------------------------------------------------------

    "Nearest Neighbor": {
        "type": "Constructive",
        "needs_start_node": True,
        "needs_initial_route": False,
        "function": nearest_neighbor,
        "description":
            "Membangun rute dengan memilih node terdekat secara bertahap.",
        "parameters": {}
    },

    "Nearest Insertion": {
        "type": "Constructive",
        "needs_start_node": True,
        "needs_initial_route": False,
        "function": nearest_insertion,
        "description":
            "Membangun rute dengan menyisipkan node pada posisi terbaik.",
        "parameters": {}
    },

    "Farthest Insertion": {
        "type": "Constructive",
        "needs_start_node": True,
        "needs_initial_route": False,
        "function": farthest_insertion,
        "description":
            "Membangun rute dengan memprioritaskan node yang paling jauh.",
        "parameters": {}
    },

    "Arbitrary Insertion": {
        "type": "Constructive",
        "needs_start_node": True,
        "needs_initial_route": False,
        "function": arbitrary_insertion,
        "description":
            "Memilih node secara acak lalu menyisipkannya pada posisi terbaik.",
        "parameters": {
            "seed": {
                "type": "int",
                "default": 42,
                "min": 0,
                "max": 99999,
                "step": 1,
                "help":
                    "Seed untuk mengontrol proses acak."
            }
        }
    },


    # --------------------------------------------------------
    # LOCAL SEARCH
    # --------------------------------------------------------

    "2-opt": {
        "type": "Local Search",
        "needs_start_node": False,
        "needs_initial_route": True,
        "function": two_opt,
        "description":
            "Memperbaiki rute dengan membalik segmen rute.",
        "parameters": {
            "max_iter": {
                "type": "int",
                "default": 100,
                "min": 1,
                "max": 10000,
                "step": 1,
                "help":
                    "Batas maksimum iterasi pencarian."
            },
            "strategy": {
                "type": "select",
                "options": ["best", "first"],
                "default": "best",
                "help":
                    "Best memilih perbaikan terbaik, sedangkan First memilih perbaikan pertama."
            }
        }
    },

    "3-opt": {
        "type": "Local Search",
        "needs_start_node": False,
        "needs_initial_route": True,
        "function": three_opt,
        "description":
            "Memperbaiki rute dengan mengevaluasi perubahan 3-opt.",
        "parameters": {
            "max_iter": {
                "type": "int",
                "default": 100,
                "min": 1,
                "max": 10000,
                "step": 1,
                "help":
                    "Batas maksimum iterasi pencarian."
            },
            "strategy": {
                "type": "select",
                "options": ["best", "first"],
                "default": "best",
                "help":
                    "Best memilih perbaikan terbaik, sedangkan First memilih perbaikan pertama."
            }
        }
    },


    # --------------------------------------------------------
    # METAHEURISTIC
    # --------------------------------------------------------

    "Simulated Annealing": {
        "type": "Metaheuristic",
        "needs_start_node": False,
        "needs_initial_route": True,
        "function": simulated_annealing,
        "description":
            "Mencari solusi dengan menerima beberapa solusi yang lebih buruk secara probabilistik.",
        "parameters": {
            "initial_temp": {
                "type": "float",
                "default": 1000.0,
                "min": 0.01,
                "max": 100000.0,
                "step": 10.0,
                "help":
                    "Suhu awal proses pencarian."
            },
            "cooling_rate": {
                "type": "float",
                "default": 0.95,
                "min": 0.01,
                "max": 0.999,
                "step": 0.01,
                "help":
                    "Seberapa cepat suhu diturunkan."
            },
            "min_temp": {
                "type": "float",
                "default": 0.01,
                "min": 0.0001,
                "max": 100.0,
                "step": 0.01,
                "help":
                    "Suhu minimum sebelum pencarian berhenti."
            },
            "max_iter": {
                "type": "int",
                "default": 10,
                "min": 1,
                "max": 10000,
                "step": 1,
                "help":
                    "Batas maksimum iterasi."
            },
            "seed": {
                "type": "int",
                "default": 42,
                "min": 0,
                "max": 99999,
                "step": 1,
                "help":
                    "Seed untuk menjaga proses acak tetap konsisten."
            },
            "verbose_history": {
                "type": "bool",
                "default": False,
                "help":
                    "Menyimpan history proses pencarian."
            }
        }
    },


    "Tabu Search": {
        "type": "Metaheuristic",
        "needs_start_node": False,
        "needs_initial_route": True,
        "function": tabu_search,
        "description":
            "Mencari solusi dengan menggunakan tabu list untuk menghindari perpindahan yang sama.",
        "parameters": {
            "max_iter": {
                "type": "int",
                "default": 100,
                "min": 1,
                "max": 10000,
                "step": 1,
                "help":
                    "Batas maksimum iterasi."
            },
            "tabu_tenure": {
                "type": "int",
                "default": 3,
                "min": 1,
                "max": 100,
                "step": 1,
                "help":
                    "Lama perpindahan disimpan dalam tabu list."
            },
            "verbose_history": {
                "type": "bool",
                "default": False,
                "help":
                    "Menyimpan history proses pencarian."
            }
        }
    }
}


# ============================================================
# SESSION STATE
# ============================================================

if "df" not in st.session_state:
    st.session_state.df = None

if "independent_results" not in st.session_state:
    st.session_state.independent_results = []


# ============================================================
# HELPER
# ============================================================

def route_to_labels(df, route):

    if route is None:
        return "-"

    return " → ".join(
        str(df.iloc[int(i)]["node"])
        for i in route
    )


def extract_route(result):

    if isinstance(result, dict):

        for key in [
            "route",
            "tour",
            "best_tour",
            "final_tour"
        ]:
            if key in result:
                return result[key]

    if isinstance(result, tuple):

        for value in result:

            if isinstance(value, (list, tuple)):
                return list(value)

    if isinstance(result, (list, tuple)):
        return list(result)

    return None


def extract_history(result):

    if isinstance(result, dict):
        return result.get("history")

    if isinstance(result, tuple):

        for value in result:

            if isinstance(value, list):

                if value and isinstance(value[0], dict):
                    return value

    return None


# ============================================================
# RUN ALGORITHM
# ============================================================

def run_algorithm(
    method,
    dist_matrix,
    start_node=None,
    initial_route=None,
    parameters=None
):

    parameters = parameters or {}

    # --------------------------------------------------------
    # CONSTRUCTIVE
    # --------------------------------------------------------

    if method == "Nearest Neighbor":

        return nearest_neighbor(
            dist_matrix,
            start_node
        )

    elif method == "Nearest Insertion":

        return nearest_insertion(
            dist_matrix,
            start_node
        )

    elif method == "Farthest Insertion":

        return farthest_insertion(
            dist_matrix,
            start_node
        )

    elif method == "Arbitrary Insertion":

        return arbitrary_insertion(
            dist_matrix,
            start_node,
            seed=parameters.get("seed")
        )


    # --------------------------------------------------------
    # LOCAL SEARCH
    # --------------------------------------------------------

    elif method == "2-opt":

        return two_opt(
            initial_route,
            dist_matrix,
            max_iter=parameters.get("max_iter", 100),
            strategy=parameters.get("strategy", "best")
        )

    elif method == "3-opt":

        return three_opt(
            initial_route,
            dist_matrix,
            max_iter=parameters.get("max_iter", 100),
            strategy=parameters.get("strategy", "best")
        )


    # --------------------------------------------------------
    # SIMULATED ANNEALING
    # --------------------------------------------------------

    elif method == "Simulated Annealing":

        return simulated_annealing(
            dist_matrix,
            initial_tour=initial_route,
            initial_temp=parameters.get(
                "initial_temp",
                1000.0
            ),
            cooling_rate=parameters.get(
                "cooling_rate",
                0.95
            ),
            min_temp=parameters.get(
                "min_temp",
                0.01
            ),
            max_iter=parameters.get(
                "max_iter",
                10
            ),
            seed=parameters.get(
                "seed",
                42
            ),
            verbose_history=parameters.get(
                "verbose_history",
                False
            )
        )


    # --------------------------------------------------------
    # TABU SEARCH
    # --------------------------------------------------------

    elif method == "Tabu Search":

        return tabu_search(
            initial_route,
            dist_matrix,
            max_iter=parameters.get(
                "max_iter",
                100
            ),
            tabu_tenure=parameters.get(
                "tabu_tenure",
                3
            ),
            verbose_history=parameters.get(
                "verbose_history",
                False
            )
        )

    else:

        raise ValueError(
            f"Algoritma '{method}' tidak ditemukan."
        )


# ============================================================
# RUN ONE INDEPENDENT ALGORITHM
# ============================================================

def execute_independent_algorithm(
    method,
    df,
    dist_matrix,
    start_node=None,
    initial_route=None,
    parameters=None
):

    start_time = time.perf_counter()

    result = run_algorithm(
        method=method,
        dist_matrix=dist_matrix,
        start_node=start_node,
        initial_route=initial_route,
        parameters=parameters
    )

    execution_time = (
        time.perf_counter()
        - start_time
    )

    route = extract_route(result)

    final_distance = tour_distance(
        route,
        dist_matrix
    )

    # Initial distance hanya relevan
    # untuk Local Search / Metaheuristic.
    if initial_route is not None:

        initial_distance = tour_distance(
            initial_route,
            dist_matrix
        )

    else:

        initial_distance = None

    return {
        "method": method,
        "route": route,
        "initial_route": initial_route,
        "initial_distance": initial_distance,
        "final_distance": final_distance,
        "execution_time": execution_time,
        "history": extract_history(result)
    }


# ============================================================
# HEADER
# ============================================================

st.title("🧭 TSP Learning & Optimization")

st.caption(
    "Pelajari, jalankan, dan bandingkan algoritma "
    "Travelling Salesman Problem."
)


# ============================================================
# INPUT DATA
# ============================================================

st.header("📂 Input Data")

input_type = st.radio(
    "Pilih sumber data:",
    ["Generate", "Upload", "Manual"],
    horizontal=True
)


new_data = None


# ------------------------------------------------------------
# GENERATE
# ------------------------------------------------------------

if input_type == "Generate":

    col1, col2, col3 = st.columns(3)

    with col1:

        n_nodes = st.number_input(
            "Jumlah node",
            min_value=3,
            max_value=500,
            value=10
        )

    with col2:

        data_seed = st.number_input(
            "Seed",
            min_value=0,
            value=42
        )

    with col3:

        coordinate_range = st.number_input(
            "Coordinate Range",
            min_value=1,
            value=100
        )


    if st.button(
        "Generate Dataset",
        type="primary"
    ):

        rng = random.Random(
            int(data_seed)
        )

        new_data = pd.DataFrame({

            "node": [
                chr(65 + i)
                if i < 26
                else f"N{i + 1}"
                for i in range(
                    int(n_nodes)
                )
            ],

            "x": [
                round(
                    rng.uniform(
                        0,
                        coordinate_range
                    ),
                    2
                )
                for _ in range(
                    int(n_nodes)
                )
            ],

            "y": [
                round(
                    rng.uniform(
                        0,
                        coordinate_range
                    ),
                    2
                )
                for _ in range(
                    int(n_nodes)
                )
            ]
        })


# ------------------------------------------------------------
# UPLOAD
# ------------------------------------------------------------

elif input_type == "Upload":

    uploaded_file = st.file_uploader(
        "Upload CSV / Excel",
        type=[
            "csv",
            "xlsx",
            "xls"
        ]
    )

    if uploaded_file is not None:

        if st.button(
            "Gunakan Dataset",
            type="primary"
        ):

            new_data = uploaded_file


# ------------------------------------------------------------
# MANUAL
# ------------------------------------------------------------

else:

    if "manual_data" not in st.session_state:

        st.session_state.manual_data = pd.DataFrame({
            "node": ["A", "B", "C", "D", "E"],
            "x": [10, 50, 90, 70, 30],
            "y": [20, 80, 40, 60, 90]
        })


    edited_data = st.data_editor(
        st.session_state.manual_data,
        num_rows="dynamic",
        use_container_width=True
    )


    if st.button(
        "Gunakan Dataset",
        type="primary"
    ):

        new_data = edited_data


# ============================================================
# LOAD DATA
# ============================================================

if new_data is not None:

    try:

        st.session_state.df = load_data(
            new_data
        )

        st.session_state.independent_results = []

        st.success(
            "Dataset berhasil digunakan."
        )

        st.rerun()

    except Exception as e:

        st.error(
            f"Gagal membaca dataset: {e}"
        )


# Stop kalau belum ada data

if st.session_state.df is None:

    st.info(
        "Masukkan dataset terlebih dahulu."
    )

    st.stop()


df = st.session_state.df


# ============================================================
# VIEW DATA
# ============================================================

st.divider()

st.header("📊 Data & Visualization")

col1, col2 = st.columns(2)


with col1:

    st.subheader("Dataset")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


with col2:

    st.subheader("Node Distribution")

    fig_nodes = plot_nodes(
        df
    )

    st.plotly_chart(
        fig_nodes,
        use_container_width=True
    )


# ============================================================
# DISTANCE METHOD
# ============================================================

st.divider()

st.header("📐 Distance Matrix")

distance_method = st.radio(
    "Distance Method",
    ["Euclidean", "Manhattan"],
    horizontal=True
)


distance_metric = (
    "euclidean"
    if distance_method == "Euclidean"
    else "manhattan"
)


coords = list(
    zip(
        df["x"],
        df["y"]
    )
)


dist_matrix = build_distance_matrix(
    coords,
    metric=distance_metric
)


with st.expander(
    "View Distance Matrix"
):

    st.dataframe(
        pd.DataFrame(
            dist_matrix,
            index=df["node"],
            columns=df["node"]
        ),
        use_container_width=True
    )


# ============================================================
# MAIN MODE
# ============================================================

st.divider()

tab_independent, tab_hybrid = st.tabs([
    "Independent",
    "Hybrid"
])


# ============================================================
# INDEPENDENT SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Independent")

    st.divider()

    # --------------------------------------------------------
    # ALGORITHM
    # --------------------------------------------------------

    st.subheader("Algorithm")

    selected_algorithms = st.multiselect(
        "Pilih satu atau beberapa algoritma:",
        list(ALGORITHMS.keys()),
        key="independent_algorithm_selection"
    )


    # --------------------------------------------------------
    # INITIAL SOLUTION
    # --------------------------------------------------------

    has_constructive = any(
        ALGORITHMS[method]["needs_start_node"]
        for method in selected_algorithms
    )

    has_route_algorithm = any(
        ALGORITHMS[method]["needs_initial_route"]
        for method in selected_algorithms
    )


    if has_constructive or has_route_algorithm:

        st.divider()

        st.subheader(
            "Initial Solution"
        )


    # --------------------------------------------------------
    # START NODE
    # --------------------------------------------------------

    start_node = None

    if has_constructive:

        st.selectbox(
            "Start Node",
            df["node"].tolist(),
            key="independent_start_node",
            help=(
                "Digunakan oleh metode Constructive "
                "untuk menentukan node awal."
            )
        )

        start_node_label = st.session_state[
            "independent_start_node"
        ]

        start_node = int(
            df.index[
                df["node"] == start_node_label
            ][0]
        )


    # --------------------------------------------------------
    # STARTING ROUTE
    # --------------------------------------------------------

    initial_route = None

    if has_route_algorithm:

        st.selectbox(
            "Starting Route",
            [
                "Generate Random",
                "Manual"
            ],
            key="independent_route_type",
            help=(
                "Local Search dan Metaheuristic "
                "membutuhkan satu rute sebagai solusi awal."
            )
        )

        route_type = st.session_state[
            "independent_route_type"
        ]


        # ----------------------------------------------------
        # RANDOM ROUTE
        # ----------------------------------------------------

        if route_type == "Generate Random":

            seed = st.number_input(
                "Seed",
                min_value=0,
                max_value=99999,
                value=42,
                step=1,
                key="independent_initial_seed"
            )

            initial_route = generate_initial_tour(
                len(df),
                home=0,
                seed=int(seed)
            )


        # ----------------------------------------------------
        # MANUAL ROUTE
        # ----------------------------------------------------

        else:

            st.caption(
                "Tentukan urutan node setelah node pertama."
            )

            route_start = st.selectbox(
                "Route Start",
                df["node"].tolist(),
                key="independent_route_start"
            )

            route_start_index = int(
                df.index[
                    df["node"] == route_start
                ][0]
            )

            remaining_nodes = [
                node
                for node in df["node"].tolist()
                if node != route_start
            ]

            route_order = st.multiselect(
                "Node Order",
                remaining_nodes,
                key="independent_route_order"
            )

            if len(route_order) == len(remaining_nodes):

                initial_route = [
                    route_start_index
                ]

                for node in route_order:

                    initial_route.append(
                        int(
                            df.index[
                                df["node"] == node
                            ][0]
                        )
                    )

                initial_route.append(
                    route_start_index
                )


    # --------------------------------------------------------
    # WHEN DIFFERENT ALGORITHMS NEED DIFFERENT INPUT
    # --------------------------------------------------------

    if (
        has_constructive
        and has_route_algorithm
    ):

        st.caption(
            "Start Node digunakan oleh Constructive, "
            "sedangkan Starting Route digunakan oleh "
            "Local Search dan Metaheuristic."
        )


    # --------------------------------------------------------
    # ALGORITHM PARAMETERS
    # --------------------------------------------------------

    algorithms_with_parameters = [
        method
        for method in selected_algorithms
        if ALGORITHMS[method]["parameters"]
    ]


    if algorithms_with_parameters:

        st.divider()

        st.subheader(
            "Algorithm Parameters"
        )


        algorithm_parameters = {}


        for method in algorithms_with_parameters:

            st.markdown(
                f"**{method}**"
            )

            algorithm_parameters[method] = {}


            for parameter, config in (
                ALGORITHMS[method]["parameters"].items()
            ):

                label = parameter.replace(
                    "_",
                    " "
                ).title()


                if config["type"] == "int":

                    value = st.number_input(
                        label,
                        min_value=config["min"],
                        max_value=config["max"],
                        value=config["default"],
                        step=config["step"],
                        key=(
                            f"independent_"
                            f"{method}_"
                            f"{parameter}"
                        ),
                        help=config["help"]
                    )


                elif config["type"] == "float":

                    value = st.number_input(
                        label,
                        min_value=config["min"],
                        max_value=config["max"],
                        value=config["default"],
                        step=config["step"],
                        key=(
                            f"independent_"
                            f"{method}_"
                            f"{parameter}"
                        ),
                        help=config["help"]
                    )


                elif config["type"] == "select":

                    value = st.selectbox(
                        label,
                        config["options"],
                        index=config["options"].index(
                            config["default"]
                        ),
                        key=(
                            f"independent_"
                            f"{method}_"
                            f"{parameter}"
                        ),
                        help=config["help"]
                    )


                elif config["type"] == "bool":

                    value = st.checkbox(
                        label,
                        value=config["default"],
                        key=(
                            f"independent_"
                            f"{method}_"
                            f"{parameter}"
                        ),
                        help=config["help"]
                    )


                algorithm_parameters[method][
                    parameter
                ] = value


# ============================================================
# INDEPENDENT CONTENT
# ============================================================

with tab_independent:

    st.subheader(
        "Independent Algorithms"
    )

    st.caption(
        "Setiap algoritma berjalan secara independen "
        "menggunakan input awalnya masing-masing."
    )


    if not selected_algorithms:

        st.info(
            "Pilih minimal satu algoritma pada sidebar."
        )


    else:

        if st.button(
            "▶ Run Independent",
            type="primary",
            use_container_width=True
        ):

            # ------------------------------------------------
            # VALIDATION
            # ------------------------------------------------

            if (
                has_route_algorithm
                and initial_route is None
            ):

                st.error(
                    "Starting Route belum lengkap."
                )

            else:

                results = []


                # ------------------------------------------------
                # RUN EACH ALGORITHM INDEPENDENTLY
                # ------------------------------------------------

                for method in selected_algorithms:

                    try:

                        # Constructive memakai start_node
                        if ALGORITHMS[method][
                            "needs_start_node"
                        ]:

                            result = (
                                execute_independent_algorithm(
                                    method=method,
                                    df=df,
                                    dist_matrix=dist_matrix,
                                    start_node=start_node,
                                    initial_route=None,
                                    parameters=algorithm_parameters.get(
                                        method,
                                        {}
                                    )
                                )
                            )

                        # Local Search / Metaheuristic
                        else:

                            result = (
                                execute_independent_algorithm(
                                    method=method,
                                    df=df,
                                    dist_matrix=dist_matrix,
                                    start_node=None,
                                    initial_route=initial_route,
                                    parameters=algorithm_parameters.get(
                                        method,
                                        {}
                                    )
                                )
                            )


                        results.append(
                            result
                        )


                    except Exception as e:

                        st.error(
                            f"{method} gagal dijalankan: {e}"
                        )


                st.session_state.independent_results = (
                    results
                )


    # ========================================================
    # RESULTS
    # ========================================================

    results = st.session_state.independent_results


    if results:

        st.divider()

        st.subheader(
            "Results"
        )


        for i, result in enumerate(results):

            method = result["method"]


            with st.expander(
                method,
                expanded=True
            ):

                col1, col2, col3 = st.columns(3)


                with col1:

                    if result["initial_distance"] is not None:

                        st.metric(
                            "Initial Distance",
                            f"{result['initial_distance']:.2f}"
                        )

                    else:

                        st.metric(
                            "Initial Distance",
                            "-"
                        )


                with col2:

                    st.metric(
                        "Final Distance",
                        f"{result['final_distance']:.2f}"
                    )


                with col3:

                    st.metric(
                        "Execution Time",
                        f"{result['execution_time'] * 1000:.3f} ms"
                    )


                st.write(
                    "**Route:**",
                    route_to_labels(
                        df,
                        result["route"]
                    )
                )


                if result["route"] is not None:

                    fig = plot_route(
                        df,
                        result["route"],
                        title=f"{method} — Final Route"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        key=f"independent_route_{i}"
                    )


        # ====================================================
        # COMPARISON
        # ====================================================

        if len(results) >= 2:

            st.divider()

            st.subheader(
                "Comparison"
            )

            comparison_df = create_comparison_table(
                results
            )

            comparison_df = sort_by_distance(
                comparison_df
            )

            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True
            )


            col1, col2 = st.columns(2)


            with col1:

                best = get_best_distance(
                    comparison_df
                )

                if best is not None:

                    st.metric(
                        "Best Distance",
                        best["Method"],
                        f"{best['Final Distance']:.2f}"
                    )


            with col2:

                fastest = get_fastest_method(
                    comparison_df
                )

                if fastest is not None:

                    st.metric(
                        "Fastest",
                        fastest["Method"],
                        f"{fastest['Execution Time (ms)']:.3f} ms"
                    )


            col1, col2 = st.columns(2)


            with col1:

                st.plotly_chart(
                    plot_distance_comparison(
                        comparison_df
                    ),
                    use_container_width=True
                )


            with col2:

                st.plotly_chart(
                    plot_time_comparison(
                        comparison_df
                    ),
                    use_container_width=True
                )


            st.plotly_chart(
                plot_improvement_comparison(
                    comparison_df
                ),
                use_container_width=True
            )


# ============================================================
# HYBRID
# ============================================================

with tab_hybrid:

    st.subheader(
        "Hybrid"
    )

    st.info(
        "Hybrid akan dibuat setelah Independent "
        "sudah berjalan dengan baik."
    )
