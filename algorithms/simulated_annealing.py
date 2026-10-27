import random
import math
import numbers
import itertools
import statistics
from collections import deque

from .base import (
    tour_distance,
    validate_tour,
    generate_initial_tour
)


_EPS = 1e-9

INITIAL_METHODS = ("default", "random", "nearest_neighbor")
MOVE_OPERATORS = ("2-opt", "swap", "insertion")


# ======================================================================
# Helper validasi
# ======================================================================

def _is_real(x):
    return (
        not isinstance(x, bool)
        and isinstance(x, numbers.Real)
        and math.isfinite(x)
    )


def _is_int(x):
    return not isinstance(x, bool) and isinstance(x, numbers.Integral)


def _validate_matrix_and_args(dist_matrix, max_iter, home):
    n = len(dist_matrix)

    if n == 0 or any(len(row) != n for row in dist_matrix):
        raise ValueError("dist_matrix harus berbentuk persegi (n x n).")

    if not _is_int(max_iter) or max_iter < 0:
        raise ValueError("max_iter harus integer >= 0.")

    if home is not None and (not _is_int(home) or not 0 <= home < n):
        raise ValueError(f"home harus berupa indeks dari 0 sampai {n - 1}.")

    return n


def _validate_sa_params(
    initial_temp, cooling_rate, min_temp, iter_per_temp,
    initial_method, move_operator,
    patience, improvement_tol, accept_window, min_accept_rate,
    target_distance,
):
    if not _is_real(initial_temp) or initial_temp <= 0:
        raise ValueError("initial_temp harus berupa angka positif.")

    if not _is_real(cooling_rate) or not 0 < cooling_rate < 1:
        raise ValueError("cooling_rate harus di antara 0 dan 1.")

    if not _is_real(min_temp) or min_temp < 0:
        raise ValueError("min_temp harus berupa angka >= 0.")

    if not _is_int(iter_per_temp) or iter_per_temp < 1:
        raise ValueError("iter_per_temp harus integer >= 1.")

    if initial_method not in INITIAL_METHODS:
        raise ValueError(f"initial_method harus salah satu dari {INITIAL_METHODS}.")

    if move_operator not in MOVE_OPERATORS:
        raise ValueError(f"move_operator harus salah satu dari {MOVE_OPERATORS}.")

    if patience is not None and (not _is_int(patience) or patience < 1):
        raise ValueError("patience harus None atau integer >= 1.")

    if not _is_real(improvement_tol) or improvement_tol < 0:
        raise ValueError("improvement_tol harus berupa angka >= 0.")

    if accept_window is not None and (
        not _is_int(accept_window) or accept_window < 1
    ):
        raise ValueError("accept_window harus None atau integer >= 1.")

    if min_accept_rate is not None and (
        not _is_real(min_accept_rate) or not 0 <= min_accept_rate <= 1
    ):
        raise ValueError("min_accept_rate harus None atau angka 0..1.")

    if (accept_window is None) != (min_accept_rate is None):
        raise ValueError("accept_window dan min_accept_rate harus diisi bersamaan.")

    if target_distance is not None and not _is_real(target_distance):
        raise ValueError("target_distance harus None atau angka.")


# ======================================================================
# Initial route
# ======================================================================

def nearest_neighbor_tour(dist_matrix, home=0):
    """Nearest Neighbor: dari home, selalu pilih kota terdekat yang belum
    dikunjungi, lalu kembali ke home. Format: [home, ..., home]."""

    n = len(dist_matrix)
    unvisited = set(range(n))
    unvisited.discard(home)

    route = [home]
    current = home

    while unvisited:
        nxt = min(unvisited, key=lambda c: dist_matrix[current][c])
        route.append(nxt)
        unvisited.remove(nxt)
        current = nxt

    route.append(home)
    return route


def random_tour(n, home=0, rng=None):
    """Rute acak. Format: [home, ..., home]."""

    rng = rng or random
    others = [c for c in range(n) if c != home]
    rng.shuffle(others)
    return [home] + others + [home]


def _prepare(initial_route, initial_method, dist_matrix, home, rng):
    """Menyiapkan rute awal. Prioritas: initial_route > initial_method."""

    n = len(dist_matrix)

    if initial_route is not None:
        route = list(initial_route)

        if not route:
            raise ValueError("initial_route tidak boleh kosong.")

        if home is None:
            home = route[0]

    else:
        if home is None:
            home = 0

        if initial_method == "nearest_neighbor":
            route = nearest_neighbor_tour(dist_matrix, home)
        elif initial_method == "random":
            route = random_tour(n, home, rng)
        else:
            route = generate_initial_tour(n, home=home)

    validate_tour(route, n, home=home)

    return route, home


# ======================================================================
# Move operator
# ======================================================================
# Posisi 0 dan terakhir (home) tidak pernah diubah.
# Posisi yang boleh diubah: 1 .. len(route) - 2.

def get_neighbors(route):
    """Menghasilkan SEMUA tetangga 2-opt (dipertahankan untuk kompatibilitas)."""

    neighbors = []

    for i in range(1, len(route) - 2):
        for j in range(i + 1, len(route) - 1):

            new_route = list(route)
            new_route[i:j + 1] = reversed(new_route[i:j + 1])

            neighbors.append({"route": new_route, "i": i, "j": j})

    return neighbors


def random_move(route, operator, rng):
    """Membuat SATU tetangga acak dengan operator tertentu.

    Return dict {"route", "i", "j"} atau None jika tidak ada gerakan valid.
    """

    m = len(route) - 2  # jumlah posisi yang boleh diubah

    if m < 2:
        return None

    new_route = list(route)

    if operator == "2-opt":
        i = rng.randint(1, m - 1)
        j = rng.randint(i + 1, m)
        new_route[i:j + 1] = reversed(new_route[i:j + 1])

    elif operator == "swap":
        i = rng.randint(1, m - 1)
        j = rng.randint(i + 1, m)
        new_route[i], new_route[j] = new_route[j], new_route[i]

    else:  # insertion: ambil kota di posisi i, sisipkan ke posisi j
        i = rng.randint(1, m)
        j = rng.randint(1, m)
        while j == i:
            j = rng.randint(1, m)
        node = new_route.pop(i)
        new_route.insert(j, node)

    return {"route": new_route, "i": i, "j": j}


# ======================================================================
# Simulated Annealing
# ======================================================================

def simulated_annealing(
    dist_matrix,
    initial_route=None,
    home=None,
    initial_temp=1000,
    cooling_rate=0.95,
    min_temp=0.01,
    max_iter=300,
    seed=None,
    # --- variasi ---
    initial_method="default",     # "default" | "random" | "nearest_neighbor"
    move_operator="2-opt",        # "2-opt" | "swap" | "insertion"
    iter_per_temp=1,              # jumlah langkah di setiap level suhu
    # --- kriteria konvergensi (None = nonaktif) ---
    patience=None,
    improvement_tol=1e-9,
    accept_window=None,
    min_accept_rate=None,
    target_distance=None,
):
    """Simulated Annealing untuk rute tertutup [home, ..., home].

    Kriteria berhenti (mana yang tercapai lebih dulu):
      "min_temp"        : suhu <= min_temp
      "max_iter"        : jumlah langkah mencapai max_iter
      "no_improvement"  : best_distance tidak membaik > improvement_tol
                          selama `patience` langkah berturut-turut
      "frozen"          : rasio penerimaan di `accept_window` langkah terakhir
                          < `min_accept_rate`
      "target_reached"  : best_distance <= target_distance
      "no_neighbors"    : tidak ada gerakan valid (kota terlalu sedikit)
    """

    _validate_matrix_and_args(dist_matrix, max_iter, home)
    _validate_sa_params(
        initial_temp, cooling_rate, min_temp, iter_per_temp,
        initial_method, move_operator,
        patience, improvement_tol, accept_window, min_accept_rate,
        target_distance,
    )

    rng = random.Random(seed)

    current_route, home = _prepare(
        initial_route, initial_method, dist_matrix, home, rng
    )

    current_distance = tour_distance(current_route, dist_matrix)

    best_route = list(current_route)
    best_distance = current_distance

    temperature = initial_temp
    no_improve_count = 0
    accept_history = (
        deque(maxlen=accept_window) if accept_window is not None else None
    )

    history = [{
        "iteration": 0,
        "route": list(current_route),
        "distance": current_distance,
        "best_distance": best_distance,
        "temperature": temperature,
        "move": None,
        "delta": None,
        "probability": None,
        "accepted": True,
    }]

    def _result(reason, iterations):
        return {
            "route": best_route,
            "distance": best_distance,
            "history": history,
            "stop_reason": reason,
            "iterations": iterations,
            "initial_method": initial_method,
            "move_operator": move_operator,
        }

    if target_distance is not None and best_distance <= target_distance + _EPS:
        return _result("target_reached", 0)

    stop_reason = "max_iter"
    last_iteration = 0

    for iteration in range(1, int(max_iter) + 1):

        if temperature <= min_temp:
            stop_reason = "min_temp"
            break

        candidate = random_move(current_route, move_operator, rng)

        if candidate is None:
            stop_reason = "no_neighbors"
            break

        candidate_route = candidate["route"]
        candidate_distance = tour_distance(candidate_route, dist_matrix)

        delta = candidate_distance - current_distance

        if delta <= 0:
            probability = 1.0
        else:
            probability = math.exp(-delta / temperature)

        accepted = rng.random() < probability

        if accepted:
            current_route = candidate_route
            current_distance = candidate_distance

        if best_distance - current_distance > improvement_tol:
            best_route = list(current_route)
            best_distance = current_distance
            no_improve_count = 0
        else:
            no_improve_count += 1

        if accept_history is not None:
            accept_history.append(1 if accepted else 0)

        history.append({
            "iteration": iteration,
            "route": list(current_route),
            "distance": current_distance,
            "best_distance": best_distance,
            "temperature": temperature,
            "move": {
                "type": move_operator,
                "positions": {"i": candidate["i"], "j": candidate["j"]},
            },
            "delta": delta,
            "probability": probability,
            "accepted": accepted,
        })

        last_iteration = iteration

        # Pendinginan: suhu turun setiap iter_per_temp langkah
        if iteration % iter_per_temp == 0:
            temperature *= cooling_rate

        # --- cek kriteria konvergensi ---
        if target_distance is not None and best_distance <= target_distance + _EPS:
            stop_reason = "target_reached"
            break

        if patience is not None and no_improve_count >= patience:
            stop_reason = "no_improvement"
            break

        if (
            accept_history is not None
            and len(accept_history) == accept_window
            and sum(accept_history) / accept_window < min_accept_rate
        ):
            stop_reason = "frozen"
            break

    return _result(stop_reason, last_iteration)


# ======================================================================
# Eksperimen untuk Analisis Parameter
# ======================================================================

def run_experiments(dist_matrix, param_grid, n_runs=10, base_seed=0, **fixed):
    """Menjalankan SA untuk setiap kombinasi parameter di `param_grid`.

    param_grid : dict {nama_parameter: [nilai1, nilai2, ...]}
    n_runs     : jumlah pengulangan per kombinasi (seed berbeda)
    fixed      : parameter SA lain yang dibuat tetap

    Contoh:
        rows = run_experiments(
            dm,
            {"cooling_rate": [0.90, 0.95, 0.99],
             "initial_temp": [100, 1000]},
            n_runs=10, patience=100,
        )
        print_experiment_table(rows)
    """

    keys = list(param_grid.keys())
    rows = []

    for values in itertools.product(*(param_grid[k] for k in keys)):
        params = dict(zip(keys, values))

        distances, iters, reasons = [], [], {}

        for r in range(n_runs):
            res = simulated_annealing(
                dist_matrix, seed=base_seed + r, **fixed, **params
            )
            distances.append(res["distance"])
            iters.append(res["iterations"])
            reasons[res["stop_reason"]] = reasons.get(res["stop_reason"], 0) + 1

        rows.append({
            "params": params,
            "best": min(distances),
            "mean": statistics.mean(distances),
            "std": statistics.pstdev(distances),
            "worst": max(distances),
            "mean_iter": statistics.mean(iters),
            "stop_reasons": reasons,
        })

    return rows


def print_experiment_table(rows):
    """Mencetak hasil run_experiments dalam bentuk tabel teks."""

    header = f"{'Parameter':<45}{'Best':>10}{'Mean':>10}{'Std':>9}{'Worst':>10}{'Iter':>8}  Stop"
    print(header)
    print("-" * len(header))

    for row in rows:
        p = ", ".join(f"{k}={v}" for k, v in row["params"].items())
        s = ", ".join(f"{k}:{v}" for k, v in row["stop_reasons"].items())
        print(
            f"{p:<45}{row['best']:>10.2f}{row['mean']:>10.2f}"
            f"{row['std']:>9.2f}{row['worst']:>10.2f}{row['mean_iter']:>8.0f}  {s}"
        )
