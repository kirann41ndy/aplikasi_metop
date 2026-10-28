from .base import tour_distance, validate_tour
from .local_search import (
    edges, move_cost, EPS_REL,
    apply_2opt, describe_2opt_move,
    apply_3opt, describe_3opt_move,
)

# ---------------------------------------------------------
# 2-opt
# ---------------------------------------------------------

def get_neighbors_2opt(route, d, current_distance):
    """
    Semua kandidat 2-opt dari rute tertutup [home, ..., home].
    Jarak tiap kandidat dihitung dari delta (O(1)), rute baru TIDAK dibangun
    di sini; cukup dibangun sekali untuk kandidat yang terpilih.

    Untuk n kota ada C(n-1, 2) - 1 kandidat (satu kombinasi dibuang karena
    membalik seluruh segmen interior = tur yang sama dibaca terbalik,
    delta selalu 0). 
    """
    neighbors = []
    n = len(route) - 1
    for i in range(1, n - 1):
        a, b = route[i - 1], route[i]
        for j in range(i + 1, n):
            if i == 1 and j == n - 1:
                continue
            c, e = route[j], route[j + 1]
            delta = d[a][c] + d[b][e] - d[a][b] - d[c][e]
            neighbors.append({
                "i": i,
                "j": j,
                "delta": delta,
                "distance": current_distance + delta,
            })
    return neighbors


def removed_edges_2opt(route, candidate):
    """Edge yang dibuang move (i, j), dihitung dari rute saat ini."""
    i, j = candidate["i"], candidate["j"]
    a, b = route[i - 1], route[i]
    c, e = route[j], route[j + 1]
    return {frozenset((a, b)), frozenset((c, e))}


def added_edges_2opt(route, candidate):
    """Edge baru yang terbentuk move (i, j), dihitung dari rute saat ini."""
    i, j = candidate["i"], candidate["j"]
    a, b = route[i - 1], route[i]
    c, e = route[j], route[j + 1]
    return {frozenset((a, c)), frozenset((b, e))}


def _move_tuple_2opt(c):
    return (c["delta"], c["i"], c["j"])


# ---------------------------------------------------------
# 3-opt
# ---------------------------------------------------------

def get_neighbors_3opt(route, d, current_distance):
    """
    Semua kandidat 3-opt (i, j, k, variant) dari rute tertutup. 7 varian
    penyambungan ulang per (i, j, k) -- varian 0-2 setara 2-opt (cuma
    membalik satu segmen efektif), varian 3-6 murni 3-opt (tukar/balik
    2 segmen sekaligus). Jumlah kandidat jauh lebih banyak dari 2-opt
    (urutan O(n^3)), jadi tiap iterasi lebih berat dibanding 2-opt/swap.
    """
    neighbors = []
    n = len(route) - 1
    for i in range(1, n - 1):
        a, b1 = route[i - 1], route[i]
        da_b1 = d[a][b1]
        for j in range(i + 1, n):
            b2, c1 = route[j - 1], route[j]
            removed_ab = da_b1 + d[b2][c1]
            for k in range(j + 1, n + 1):
                c2, e = route[k - 1], route[k]
                removed = removed_ab + d[c2][e]

                added = (
                    d[a][b2] + d[b1][c1] + d[c2][e],   # 0
                    d[a][b1] + d[b2][c2] + d[c1][e],   # 1
                    d[a][c2] + d[c1][b2] + d[b1][e],   # 2
                    d[a][b2] + d[b1][c2] + d[c1][e],   # 3
                    d[a][c1] + d[c2][b1] + d[b2][e],   # 4
                    d[a][c1] + d[c2][b2] + d[b1][e],   # 5
                    d[a][c2] + d[c1][b1] + d[b2][e],   # 6
                )

                for variant, cost in enumerate(added):
                    delta = cost - removed
                    neighbors.append({
                        "i": i, "j": j, "k": k, "variant": variant,
                        "delta": delta,
                        "distance": current_distance + delta,
                    })
    return neighbors


def removed_edges_3opt(route, candidate):
    i, j, k = candidate["i"], candidate["j"], candidate["k"]
    a, b1 = route[i - 1], route[i]
    b2, c1 = route[j - 1], route[j]
    c2, e = route[k - 1], route[k]
    return {frozenset((a, b1)), frozenset((b2, c1)), frozenset((c2, e))}


def added_edges_3opt(route, candidate):
    """
    Mencerminkan tabel penyambungan ulang 3-opt di local_search.describe_3opt_move
    (RECONNECTIONS_3OPT).
    """
    i, j, k, variant = (
        candidate["i"], candidate["j"], candidate["k"], candidate["variant"]
    )
    a, b1 = route[i - 1], route[i]
    b2, c1 = route[j - 1], route[j]
    c2, e = route[k - 1], route[k]

    added_pairs = (
        [(a, b2), (b1, c1), (c2, e)],   # 0
        [(a, b1), (b2, c2), (c1, e)],   # 1
        [(a, c2), (c1, b2), (b1, e)],   # 2
        [(a, b2), (b1, c2), (c1, e)],   # 3
        [(a, c1), (c2, b1), (b2, e)],   # 4
        [(a, c1), (c2, b2), (b1, e)],   # 5
        [(a, c2), (c1, b1), (b2, e)],   # 6
    )[variant]

    return {frozenset(p) for p in added_pairs}


def _move_tuple_3opt(c):
    return (c["delta"], c["i"], c["j"], c["k"], c["variant"])


# ---------------------------------------------------------
# Swap (tukar posisi 2 kota)
# ---------------------------------------------------------

def get_neighbors_swap(route, d, current_distance):
    """
    Semua kandidat swap: tukar posisi 2 kota interior (i, j), i < j.
    Beda dari 2-opt/3-opt yang membalik segmen, swap cuma menukar 2 kota
    tanpa membalik urutan kota di antaranya.

    Kasus kota bertetangga (j == i+1) ditangani khusus: edge di antara
    keduanya (route[i], route[j]) tidak berubah (cuma arah bacanya yang
    kebalik, jaraknya sama), jadi cuma 2 edge luar yang ikut berubah,
    bukan 4.

    Untuk n kota ada C(n-1, 2) kandidat (satu lebih banyak dari 2-opt,
    karena swap tidak punya kombinasi degenerate yang perlu dibuang).
    """
    neighbors = []
    n = len(route) - 1
    for i in range(1, n - 1):
        for j in range(i + 1, n):
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
            delta = new - old
            neighbors.append({
                "i": i,
                "j": j,
                "delta": delta,
                "distance": current_distance + delta,
            })
    return neighbors


def removed_edges_swap(route, candidate):
    i, j = candidate["i"], candidate["j"]
    if j == i + 1:
        return {frozenset((route[i - 1], route[i])), frozenset((route[j], route[j + 1]))}
    return {
        frozenset((route[i - 1], route[i])), frozenset((route[i], route[i + 1])),
        frozenset((route[j - 1], route[j])), frozenset((route[j], route[j + 1])),
    }


def added_edges_swap(route, candidate):
    i, j = candidate["i"], candidate["j"]
    if j == i + 1:
        return {frozenset((route[i - 1], route[j])), frozenset((route[i], route[j + 1]))}
    return {
        frozenset((route[i - 1], route[j])), frozenset((route[j], route[i + 1])),
        frozenset((route[j - 1], route[i])), frozenset((route[i], route[j + 1])),
    }


def apply_swap(route, move):
    _, i, j = move
    new_route = list(route)
    new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route


def describe_swap_move(route, d, move):
    _, i, j = move
    if j == i + 1:
        removed_pairs = [(route[i - 1], route[i]), (route[j], route[j + 1])]
        added_pairs = [(route[i - 1], route[j]), (route[i], route[j + 1])]
    else:
        removed_pairs = [
            (route[i - 1], route[i]), (route[i], route[i + 1]),
            (route[j - 1], route[j]), (route[j], route[j + 1]),
        ]
        added_pairs = [
            (route[i - 1], route[j]), (route[j], route[i + 1]),
            (route[j - 1], route[i]), (route[i], route[j + 1]),
        ]

    removed = edges(removed_pairs, d)
    added = edges(added_pairs, d)
    removed_cost, added_cost, delta = move_cost(removed, added)

    return {
        "type": "swap",
        "positions": {"i": i, "j": j},
        "swapped_cities": {"posisi_i": route[i], "posisi_j": route[j]},
        "removed_edges": removed,
        "added_edges": added,
        "removed_cost": removed_cost,
        "added_cost": added_cost,
        "delta": delta,
    }


def _move_tuple_swap(c):
    return (c["delta"], c["i"], c["j"])

MOVE_TYPES = {
    "2opt": {
        "label": "2-opt",
        "get_neighbors": get_neighbors_2opt,
        "move_tuple": _move_tuple_2opt,
        "apply": apply_2opt,
        "describe": describe_2opt_move,
        "broken_edges": removed_edges_2opt,
        "formed_edges": added_edges_2opt,
    },
    "3opt": {
        "label": "3-opt",
        "get_neighbors": get_neighbors_3opt,
        "move_tuple": _move_tuple_3opt,
        "apply": apply_3opt,
        "describe": describe_3opt_move,
        "broken_edges": removed_edges_3opt,
        "formed_edges": added_edges_3opt,
    },
    "swap": {
        "label": "Swap",
        "get_neighbors": get_neighbors_swap,
        "move_tuple": _move_tuple_swap,
        "apply": apply_swap,
        "describe": describe_swap_move,
        "broken_edges": removed_edges_swap,
        "formed_edges": added_edges_swap,
    },
}

MOVE_TYPE_LABELS = {key: spec["label"] for key, spec in MOVE_TYPES.items()}

STOP_CRITERIA = {
    "max_iter": "Jumlah iterasi maksimum",
    "convergence": "Konvergen (tidak ada perbaikan dalam N iterasi)",
}

def tabu_remaining(tokens, tabu_list, iteration):
    """
    Total sisa tenure dari edge yang dibuang move pada iterasi ini.
    0 = tidak ada edge tabu (move boleh); >0 = move tabu. Dipakai juga untuk
    memilih move "least tabu" saat semua kandidat tabu.
    """
    return sum(
        max(0, tabu_list.get(tok, -1) - iteration + 1) for tok in tokens
    )


def _validate(initial_route, dist_matrix, max_iter, tabu_tenure, move_type, stop_criterion, patience):
    n = len(dist_matrix)
    if any(len(row) != n for row in dist_matrix):
        raise ValueError("dist_matrix harus berbentuk persegi (n x n).")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 0:
        raise ValueError("max_iter harus integer >= 0.")
    if isinstance(tabu_tenure, bool) or not isinstance(tabu_tenure, int) or tabu_tenure < 1:
        raise ValueError("tabu_tenure harus integer >= 1.")
    if move_type not in MOVE_TYPES:
        raise ValueError(
            f"move_type harus salah satu dari {tuple(MOVE_TYPES)}."
        )
    if stop_criterion not in STOP_CRITERIA:
        raise ValueError(f"stop_criterion harus salah satu dari {tuple(STOP_CRITERIA)}.")
    if isinstance(patience, bool) or not isinstance(patience, int) or patience < 1:
        raise ValueError("patience harus integer >= 1.")
    route = list(initial_route)
    validate_tour(route, n, home=route[0] if route else 0)
    return route


def tabu_search(
    # input utama
    initial_route,
    dist_matrix,

    # parameter algoritma
    max_iter=1000,
    tabu_tenure=5,
    move_type="2opt", 
    stop_criterion ="convergence",
    patience = 30,
    verbose_history=False,
):
    """
    Menjalankan Tabu Search untuk TSP.

    Step 1 : home city = initial_route[0] (= initial_route[-1], closed tour).
    Step 2 : initial_route jadi current solution; jaraknya jadi aspiration
             level awal; tabu_list kosong.
    Step 3 : iterasi i = 1..max_iter.
    Step 4 : bentuk semua kandidat sesuai move_type yang dipilih (2-opt /
             3-opt / swap), hitung total distance tiap kandidat.
    Step 5 : pilih kandidat admissible terbaik jadi current solution (boleh
             lebih buruk dari sebelumnya), lalu update tabu_list.
    Step 6 : ulangi sampai max_iter atau konvergen. Rute terbaik = hasil akhir.

    Larangan tabu dipasang pada edge baru yang terbentuk tiap move dan berlaku
    sampai iterasi (t + tabu_tenure), berlaku sama untuk ketiga move_type.
    Sebuah move tabu jika salah satu edge yang dibuangnya masih tabu; move
    tabu ditolak kecuali aspiration criterion terpenuhi (jarak < solusi
    terbaik sejauh ini). Jika semua kandidat tabu dan tak ada yang lolos
    aspiration, dipakai aspiration by default: pilih move "least tabu"
    (total sisa tenure edge-dibuang terkecil), seri -> jarak terkecil.

    Parameters
    ----------
    initial_route : list
        Rute awal (closed tour: [home, ..., home]).
    dist_matrix : list
        Matriks jarak (diasumsikan simetris).
    max_iter : int
        Maksimum iterasi. Pada stop_criterion="max_iter" ini adalah satu-
        satunya kriteria henti; pada "convergence" ini jadi batas pengaman
        supaya loop tidak berjalan tanpa akhir.
    tabu_tenure : int
        Lama (iterasi) sebuah edge baru dilarang dibongkar.
    move_type : str
        Mekanisme pergerakan/neighborhood: "2opt" (default), "3opt", atau
        "swap". Neighborhood 3-opt jauh lebih besar (O(n^3) vs O(n^2) untuk
        2-opt/swap), jadi tiap iterasi lebih lambat, pertimbangkan
        max_iter lebih kecil kalau n besar.
    verbose_history : bool
        False -> history[k]["candidates"] = [] (ringan).
        True  -> diisi SEMUA kandidat tiap iterasi beserta status tabu,
        sisa tenure, aspiration, admissible, dan penanda "selected"
        (untuk tab Ilustrasi).
    patience : int
        Jumlah iterasi berturut-turut tanpa perbaikan best_distance yang
        ditoleransi sebelum dianggap konvergen. Hanya dipakai jika
        stop_criterion="convergence". Harus integer >= 1; sebaiknya tidak
        lebih kecil dari tabu_tenure agar tabu list sempat "lepas".
    stop_criterion : str
        Kriteria perhentian: "max_iter" (default) berhenti setelah max_iter
        iterasi; "convergence" berhenti lebih awal jika best_distance tidak
        membaik selama `patience` iterasi berturut-turut. Yang dipantau
        adalah solusi terbaik, bukan rute yang sedang dijalani (rute
        sekarang di Tabu Search hampir selalu berpindah).
    Returns
    -------
    dict
        {
            "route": rute terbaik,
            "distance": total jarak terbaik,
            "best_iteration": iterasi saat solusi terbaik ditemukan (0 = awal),
            "move_type": move_type yang dipakai,
            "history": riwayat iterasi
        }
    """

    # =========================
    # 1. Initialization
    # =========================

    d = dist_matrix
    current_route = _validate(initial_route, d, max_iter, tabu_tenure, move_type, stop_criterion, patience)
    current_distance = tour_distance(current_route, d)

    move_spec = MOVE_TYPES[move_type]

    best_route = list(current_route)
    best_distance = current_distance          # aspiration level awal (Step 2)
    best_iteration = 0
    no_improve = 0
    stop_reason = "max_iter"

    # Toleransi relatif agar perbandingan
    # float tidak salah karena selisih ~1e-15.
    eps = EPS_REL * current_distance / max(len(current_route) - 1, 1)

    tabu_list = {}   # edge (frozenset) -> iterasi kadaluarsa (flat, 1 edge = 1 entri)

    history = [{
        "iteration": 0,
        "route": current_route.copy(),
        "distance": current_distance,
        "move": None,
        "tabu_list": tabu_list.copy(),
        "candidates": [],
    }]

    # =========================
    # 2. Main Algorithm
    # =========================

    for iteration in range(1, max_iter + 1):

        neighbors = move_spec["get_neighbors"](current_route, d, current_distance)
        if not neighbors:
            stop_reason = "no_candidates"
            break   # rute terlalu pendek: tidak ada kandidat move

        candidate_log = []
        best_candidate = None
        best_idx = None

        # Step 4-5: evaluasi seluruh neighborhood
        for idx, nb in enumerate(neighbors):

            # tabu jika salah satu edge yang dibuang masih tabu
            removed = move_spec["broken_edges"](current_route, nb)
            remaining = tabu_remaining(removed, tabu_list, iteration)
            nb["tabu_remaining"] = remaining
            is_tabu = remaining > 0
            aspiration_met = nb["distance"] < best_distance - eps
            admissible = (not is_tabu) or aspiration_met

            if verbose_history:
                entry = dict(nb)
                entry.update({
                    "is_tabu": is_tabu,
                    "aspiration_met": aspiration_met,
                    # aspiration hanya bermakna kalau move tabu
                    "aspiration_relevant": is_tabu,
                    "admissible": admissible,
                    "selected": False,
                })
                candidate_log.append(entry)

            if admissible and (best_candidate is None
                               or nb["distance"] < best_candidate["distance"]):
                best_candidate = nb
                best_idx = idx

        # aspiration by default: semua kandidat tabu dan tak ada yang
        # aspiration_met -> bebaskan move "least tabu" (sisa tenure terkecil),
        # kalau seri ambil jarak terkecil.
        by_default = False
        if best_candidate is None:
            by_default = True
            best_idx = min(
                range(len(neighbors)),
                key=lambda k: (neighbors[k]["tabu_remaining"],
                               neighbors[k]["distance"]),
            )
            best_candidate = neighbors[best_idx]
            if verbose_history:
                candidate_log[best_idx]["admissible"] = True

        if verbose_history:
            candidate_log[best_idx]["selected"] = True

        move = move_spec["move_tuple"](best_candidate)

        # attribute tabu dan detail move dihitung dari rute SEBELUM move
        new_attr = move_spec["formed_edges"](current_route, best_candidate)
        detail = move_spec["describe"](current_route, d, move)
        for key in ("i", "j", "k", "variant"):
            if key in best_candidate:
                detail[key] = best_candidate[key]
        detail["distance_before"] = current_distance
        detail["aspiration_by_default"] = by_default

        current_route = move_spec["apply"](current_route, move)
        # hitung ulang dari rute (bukan akumulasi delta) supaya tidak drift
        current_distance = tour_distance(current_route, d)
        detail["distance_after"] = current_distance

        # update tabu list: tambah edge baru, buang yang sudah kadaluarsa
        for tok in new_attr:
            tabu_list[tok] = iteration + tabu_tenure
        tabu_list = {k: v for k, v in tabu_list.items() if v >= iteration}

        if current_distance < best_distance - eps:
            best_distance = current_distance
            best_route = current_route.copy()
            best_iteration = iteration
            no_improve = 0
        else:
            no_improve += 1

        history.append({
            "iteration": iteration,
            "route": current_route.copy(),
            "distance": current_distance,
            "move": detail,
            "tabu_list": tabu_list.copy(),
            "candidates": candidate_log,
        })

        if stop_criterion == "convergence" and no_improve >= patience:
            stop_reason = "convergence"
            break


    # =========================
    # 3. Return Result
    # =========================

    return {
        "route": best_route,
        "distance": best_distance,
        "best_iteration": best_iteration,
        "move_type": move_type,
        "history": history,
        "stop_criterion": stop_criterion,
        "stop_reason": stop_reason,
        "iterations_run": len(history) - 1,
    }
