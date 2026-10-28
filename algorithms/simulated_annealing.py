import math
import random

from .base import tour_distance, validate_tour
from .local_search import (
    EPS_REL,
    apply_2opt, describe_2opt_move,
    apply_3opt, describe_3opt_move,
)
from .tabu_search import apply_swap, describe_swap_move

# ---------------------------------------------------------
# Sampler move acak (SA hanya butuh SATU tetangga acak per iterasi,
# bukan seluruh neighborhood seperti Tabu Search)
# Setiap sampler mengembalikan move tuple yang sama formatnya dengan
# apply_* / describe_* di local_search & tabu_search:
#   swap/2opt : (delta, i, j)
#   3opt      : (delta, i, j, k, variant)
# atau None jika rute terlalu pendek untuk move tersebut.
# ---------------------------------------------------------

def sample_swap(route, d, rng):
    n = len(route) - 1
    if n < 3:
        return None
    i, j = sorted(rng.sample(range(1, n), 2))
    if j == i + 1:
        old = d[route[i - 1]][route[i]] + d[route[j]][route[j + 1]]
        new = d[route[i - 1]][route[j]] + d[route[i]][route[j + 1]]
    else:
        old = (
            d[route[i - 1]][route[i]] + d[route[i]][route[i + 1]]
            + d[route[j - 1]][route[j]] + d[route[j]][route[j + 1]]
        )
        new = (
            d[route[i - 1]][route[j]] + d[route[j]][route[i + 1]]
            + d[route[j - 1]][route[i]] + d[route[i]][route[j + 1]]
        )
    return (new - old, i, j)


def sample_2opt(route, d, rng):
    n = len(route) - 1
    if n < 4:
        return None
    while True:
        i, j = sorted(rng.sample(range(1, n), 2))
        if not (i == 1 and j == n - 1):   # balik seluruh interior = delta 0
            break
    a, b = route[i - 1], route[i]
    c, e = route[j], route[j + 1]
    delta = d[a][c] + d[b][e] - d[a][b] - d[c][e]
    return (delta, i, j)


def sample_3opt(route, d, rng):
    n = len(route) - 1
    if n < 3:
        return None
    # 3 titik potong berbeda i < j < k, dengan 1 <= i, k <= n
    i, j, k = sorted(rng.sample(range(1, n + 1), 3))
    variant = rng.randrange(7)

    a, b1 = route[i - 1], route[i]
    b2, c1 = route[j - 1], route[j]
    c2, e = route[k - 1], route[k]
    removed = d[a][b1] + d[b2][c1] + d[c2][e]
    added = (
        d[a][b2] + d[b1][c1] + d[c2][e],   # 0
        d[a][b1] + d[b2][c2] + d[c1][e],   # 1
        d[a][c2] + d[c1][b2] + d[b1][e],   # 2
        d[a][b2] + d[b1][c2] + d[c1][e],   # 3
        d[a][c1] + d[c2][b1] + d[b2][e],   # 4
        d[a][c1] + d[c2][b2] + d[b1][e],   # 5
        d[a][c2] + d[c1][b1] + d[b2][e],   # 6
    )[variant]
    return (added - removed, i, j, k, variant)


# ---------------------------------------------------------
# Registry: sama gayanya dengan MOVE_TYPES di tabu_search
# ---------------------------------------------------------

MOVE_TYPES = {
    "2opt": {"label": "2-opt", "sample": sample_2opt,
             "apply": apply_2opt, "describe": describe_2opt_move},
    "3opt": {"label": "3-opt", "sample": sample_3opt,
             "apply": apply_3opt, "describe": describe_3opt_move},
    "swap": {"label": "Swap", "sample": sample_swap,
             "apply": apply_swap, "describe": describe_swap_move},
}

MOVE_TYPE_LABELS = {key: spec["label"] for key, spec in MOVE_TYPES.items()}

# Pada fase panas (suhu > rasio x T0) SA sengaja menerima solusi buruk, jadi
# "tidak ada perbaikan" di fase itu bukan tanda konvergen. Penghitung patience
# baru berjalan setelah suhu turun di bawah rasio ini.
CONVERGENCE_TEMP_RATIO = 0.1

STOP_CRITERIA = {
    "max_iter": "Jumlah iterasi maksimum",
    "min_temp": "Suhu minimum tercapai",
    "convergence": "Konvergen (tidak ada perbaikan dalam N iterasi)",
}


# ---------------------------------------------------------
# Helper
# ---------------------------------------------------------

def estimate_initial_temp(route, d, sampler, rng,
                          target_accept=0.8, samples=200):
    """
    Taksir suhu awal T0 supaya move yang memburuk rata-rata diterima dengan
    peluang `target_accept` di awal: T0 = -mean(delta+) / ln(target_accept).
    """
    worse = []
    for _ in range(samples):
        mv = sampler(route, d, rng)
        if mv is not None and mv[0] > 0:
            worse.append(mv[0])
    if not worse:
        return 1.0
    return (sum(worse) / len(worse)) / -math.log(target_accept)


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _validate(initial_route, dist_matrix, max_iter, initial_temp, cooling_rate,
              min_temp, iter_per_temp, move_type, stop_criterion, patience):
    n = len(dist_matrix)
    if any(len(row) != n for row in dist_matrix):
        raise ValueError("dist_matrix harus berbentuk persegi (n x n).")
    if not _is_int(max_iter) or max_iter < 0:
        raise ValueError("max_iter harus integer >= 0.")
    if initial_temp is not None and not initial_temp > 0:
        raise ValueError("initial_temp harus > 0 (atau None untuk otomatis).")
    if not 0 < cooling_rate < 1:
        raise ValueError("cooling_rate harus di antara 0 dan 1 (eksklusif).")
    if not min_temp > 0:
        raise ValueError("min_temp harus > 0.")
    if iter_per_temp is not None and (not _is_int(iter_per_temp) or iter_per_temp < 1):
        raise ValueError("iter_per_temp harus integer >= 1.")
    if move_type not in MOVE_TYPES:
        raise ValueError(f"move_type harus salah satu dari {tuple(MOVE_TYPES)}.")
    if stop_criterion not in STOP_CRITERIA:
        raise ValueError(
            f"stop_criterion harus salah satu dari {tuple(STOP_CRITERIA)}."
        )
    if not _is_int(patience) or patience < 1:
        raise ValueError("patience harus integer >= 1.")
    route = list(initial_route)
    validate_tour(route, n, home=route[0] if route else 0)
    return route


# ---------------------------------------------------------
# Simulated Annealing
# ---------------------------------------------------------

def simulated_annealing(
    # input utama
    initial_route,
    dist_matrix,

    # parameter algoritma
    max_iter=10000,
    initial_temp=None,
    cooling_rate=0.95,
    min_temp=1e-3,
    iter_per_temp=None,
    move_type="2opt",
    stop_criterion="max_iter",
    patience=1000,
    seed=None,
):
    """
    Menjalankan Simulated Annealing untuk TSP (closed tour).

    Step 1 : home city = initial_route[0]; current = best = initial_route.
    Step 2 : suhu awal T0 (diberikan, atau ditaksir otomatis jika None).
    Step 3 : tiap iterasi, ambil SATU tetangga acak sesuai move_type.
    Step 4 : delta = jarak_baru - jarak_lama.
             delta <= 0 -> selalu diterima.
             delta  > 0 -> diterima dengan peluang exp(-delta / T).
    Step 5 : tiap `iter_per_temp` iterasi, suhu didinginkan: T <- T * cooling_rate.
    Step 6 : ulangi sampai kriteria henti terpenuhi. Hasil = best_route
             (solusi terbaik yang pernah dikunjungi, bukan rute terakhir).

    Parameters
    ----------
    initial_route : list
        Rute awal tertutup [home, ..., home] (mis. dari nearest_neighbor).
    dist_matrix : list
        Matriks jarak.
    max_iter : int
        Maksimum iterasi (satu iterasi = satu move yang diusulkan, diterima
        ataupun ditolak). Pada "min_temp"/"convergence" jadi batas pengaman.
    initial_temp : float | None
        Suhu awal. None -> ditaksir otomatis agar move yang memburuk diterima
        ~80% di awal (lihat estimate_initial_temp).
    cooling_rate : float
        Faktor pendinginan geometrik (0 < alpha < 1). Makin dekat 1 makin
        lambat dingin: lebih teliti tapi butuh iterasi lebih banyak.
    min_temp : float
        Batas suhu minimum (dipakai pada stop_criterion="min_temp").
    iter_per_temp : int | None
        Jumlah iterasi pada tiap level suhu sebelum didinginkan. None ->
        jumlah kota (n), praktik umum agar tiap level suhu sempat "mencapai
        kesetimbangan".
    move_type : str
        "2opt" (default), "3opt", atau "swap".
    stop_criterion : str
        "max_iter" | "min_temp" | "convergence".
    patience : int
        Untuk "convergence": berhenti jika best_distance tidak membaik selama
        `patience` iterasi berturut-turut. Hitungan baru dimulai setelah suhu
        <= 10% T0 (CONVERGENCE_TEMP_RATIO), karena fase panas memang eksploratif;
        yang dipantau adalah perbaikan sejak fase dingin dimulai.
    seed : int | None
        Seed RNG agar hasil reproducible.

    Returns
    -------
    dict
        {
            "route": rute terbaik, "distance": jarak terbaik,
            "best_iteration": iterasi saat terbaik ditemukan (0 = awal),
            "move_type", "stop_criterion", "stop_reason", "iterations_run",
            "initial_temp": T0 yang dipakai, "final_temp": suhu akhir,
            "acceptance_rate": proporsi move yang diterima,
            "history": riwayat tiap iterasi
        }
    """

    # =========================
    # 1. Initialization
    # =========================

    d = dist_matrix
    current_route = _validate(
        initial_route, d, max_iter, initial_temp, cooling_rate, min_temp,
        iter_per_temp, move_type, stop_criterion, patience,
    )
    current_distance = tour_distance(current_route, d)

    if iter_per_temp is None:
        iter_per_temp = max(len(d), 1)

    rng = random.Random(seed)
    spec = MOVE_TYPES[move_type]
    sampler = spec["sample"]

    best_route = current_route.copy()
    best_distance = current_distance
    best_iteration = 0
    no_improve = 0
    stop_reason = "max_iter"

    eps = EPS_REL * current_distance / max(len(current_route) - 1, 1)

    if initial_temp is None:
        temp = estimate_initial_temp(current_route, d, sampler, rng)
    else:
        temp = float(initial_temp)
    t0 = temp

    accepted_count = 0
    cold_best = None   # jarak terbaik sejak suhu <= rasio x T0 (untuk convergence)

    history = [{
        "iteration": 0,
        "route": current_route.copy(),
        "distance": current_distance,
        "best_distance": best_distance,
        "temperature": temp,
        "move": None,
        "delta": None,
        "probability": None,
        "random_u": None,
        "accepted": None,
    }]

    # =========================
    # 2. Main Algorithm
    # =========================

    for iteration in range(1, max_iter + 1):

        if stop_criterion == "min_temp" and temp <= min_temp:
            stop_reason = "min_temp"
            break

        move = sampler(current_route, d, rng)
        if move is None:
            stop_reason = "no_candidates"
            break   # rute terlalu pendek untuk move ini

        delta = move[0]

        # Kriteria Metropolis
        if delta <= 0:
            probability, u, accepted = 1.0, None, True
        else:
            probability = math.exp(-delta / temp)
            u = rng.random()
            accepted = u < probability

        detail = None
        if accepted:
            detail = spec["describe"](current_route, d, move)
            detail["distance_before"] = current_distance
            current_route = spec["apply"](current_route, move)
            # hitung ulang dari rute (bukan akumulasi delta) agar tidak drift
            current_distance = tour_distance(current_route, d)
            detail["distance_after"] = current_distance
            accepted_count += 1

        if current_distance < best_distance - eps:
            best_distance = current_distance
            best_route = current_route.copy()
            best_iteration = iteration

        # Patience dipantau pada fase dingin saja, terhadap solusi terbaik
        # SEJAK fase dingin dimulai (bukan best global yang bisa saja masih
        # rute awal, karena fase panas memang menjauh dari rute awal).
        if temp <= CONVERGENCE_TEMP_RATIO * t0:
            if cold_best is None or current_distance < cold_best - eps:
                cold_best = current_distance
                no_improve = 0
            else:
                no_improve += 1

        history.append({
            "iteration": iteration,
            "route": current_route.copy(),
            "distance": current_distance,
            "best_distance": best_distance,
            "temperature": temp,
            "move": detail,            # None jika move ditolak
            "delta": delta,
            "probability": probability,
            "random_u": u,
            "accepted": accepted,
        })

        # Pendinginan geometrik
        if iteration % iter_per_temp == 0:
            temp *= cooling_rate

        if stop_criterion == "convergence" and no_improve >= patience:
            stop_reason = "convergence"
            break

    # =========================
    # 3. Return Result
    # =========================

    iterations_run = len(history) - 1
    return {
        "route": best_route,
        "distance": best_distance,
        "best_iteration": best_iteration,
        "move_type": move_type,
        "history": history,
        "stop_criterion": stop_criterion,
        "stop_reason": stop_reason,
        "iterations_run": iterations_run,
        "initial_temp": t0,
        "final_temp": temp,
        "acceptance_rate": accepted_count / iterations_run if iterations_run else 0.0,
    }
