
from .base import tour_distance, validate_tour
from .local_search import apply_2opt, describe_2opt_move, EPS_REL


def get_neighbors(route, d, current_distance):
    """
    Semua kandidat 2-opt dari rute tertutup [home, ..., home].
    Jarak tiap kandidat dihitung dari delta (O(1)), rute baru TIDAK dibangun
    di sini; cukup dibangun sekali untuk kandidat yang terpilih.

    Untuk n kota ada C(n-1, 2) kandidat, mis. 6 kota -> 9 kandidat.
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


def removed_edges(route, i, j):
    """
    Edge yang dibuang move (i, j), dihitung dari rute saat ini. Dipakai untuk
    cek terhadap tabu: move tabu kalau salah satu edge ini masih tabu
    (admissible hanya jika kedua edge yang dibuang non-tabu).
    """
    e1 = frozenset((route[i - 1], route[i]))
    e2 = frozenset((route[j], route[j + 1]))
    return frozenset((e1, e2))


def added_edges(route, i, j):
    """
    Edge yang terbentuk oleh move (i, j), dihitung dari rute saat ini. Kedua
    edge ini yang dijadikan tabu (dilarang dibuang lagi) selama tabu_tenure.
    """
    e1 = frozenset((route[i - 1], route[j]))
    e2 = frozenset((route[i], route[j + 1]))
    return frozenset((e1, e2))


def tabu_edge_expiry(tabu_list):
    """
    Ratakan tabu_list {pasangan edge baru: expiry} menjadi {edge: expiry}.
    Kalau satu edge muncul di beberapa entri, dipakai expiry yang terbesar.
    """
    edge_expiry = {}
    for pair, expiry in tabu_list.items():
        for edge in pair:
            if expiry > edge_expiry.get(edge, -1):
                edge_expiry[edge] = expiry
    return edge_expiry


def tabu_remaining(removed, edge_expiry, iteration):
    """
    Total sisa tenure dari edge yang dibuang move pada iterasi ini.
    0 = tidak ada edge tabu (move boleh); >0 = move tabu. Dipakai juga untuk
    memilih move "least tabu" saat semua kandidat tabu.
    """
    return sum(
        max(0, edge_expiry.get(edge, -1) - iteration + 1) for edge in removed
    )


def _validate(initial_route, dist_matrix, max_iter, tabu_tenure):
    n = len(dist_matrix)
    if any(len(row) != n for row in dist_matrix):
        raise ValueError("dist_matrix harus berbentuk persegi (n x n).")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 0:
        raise ValueError("max_iter harus integer >= 0.")
    if isinstance(tabu_tenure, bool) or not isinstance(tabu_tenure, int) or tabu_tenure < 1:
        raise ValueError("tabu_tenure harus integer >= 1.")
    route = list(initial_route)
    validate_tour(route, n, home=route[0] if route else 0)
    return route


def tabu_search(
    # input utama
    initial_route,
    dist_matrix,

    # parameter algoritma
    max_iter=100,
    tabu_tenure=3,
    verbose_history=False,
):
    """
    Menjalankan Tabu Search untuk TSP.

    Step 1 : home city = initial_route[0] (= initial_route[-1], closed tour).
    Step 2 : initial_route jadi current solution; jaraknya jadi aspiration
             level awal; tabu_list kosong.
    Step 3 : iterasi i = 1..max_iter.
    Step 4 : bentuk semua kandidat 2-opt, hitung total distance tiap kandidat.
    Step 5 : pilih kandidat admissible terbaik jadi current solution (boleh
             lebih buruk dari sebelumnya), lalu update tabu_list.
    Step 6 : ulangi sampai max_iter. Rute terbaik = hasil akhir.

    Larangan tabu dipasang pada edge baru yang terbentuk tiap move dan berlaku
    sampai iterasi (t + tabu_tenure). Sebuah move tabu jika salah satu edge
    yang dibuangnya masih tabu; move tabu ditolak kecuali aspiration criterion
    terpenuhi (jarak < solusi terbaik sejauh ini). Jika semua kandidat tabu dan
    tak ada yang lolos aspiration, dipakai aspiration by default: pilih move
    "least tabu" (total sisa tenure edge-dibuang terkecil), seri -> jarak
    terkecil.

    Parameters
    ----------
    initial_route : list
        Rute awal (closed tour: [home, ..., home]).
    dist_matrix : list
        Matriks jarak (diasumsikan simetris).
    max_iter : int
        Maksimum iterasi.
    tabu_tenure : int
        Lama (iterasi) sebuah edge baru dilarang dibongkar.
    verbose_history : bool
        False -> history[k]["candidates"] = [] (ringan).
        True  -> diisi SEMUA kandidat tiap iterasi beserta status tabu,
        sisa tenure, aspiration, admissible, dan penanda "selected"
        (untuk tab Ilustrasi).

    Returns
    -------
    dict
        {
            "route": rute terbaik,
            "distance": total jarak terbaik,
            "best_iteration": iterasi saat solusi terbaik ditemukan (0 = awal),
            "history": riwayat iterasi
        }
    """

    # =========================
    # 1. Initialization
    # =========================

    d = dist_matrix
    current_route = _validate(initial_route, d, max_iter, tabu_tenure)
    current_distance = tour_distance(current_route, d)

    best_route = list(current_route)
    best_distance = current_distance          # aspiration level awal (Step 2)
    best_iteration = 0

    # Toleransi relatif (sama gaya dengan local_search) agar perbandingan
    # float tidak salah karena selisih ~1e-15.
    eps = EPS_REL * current_distance / max(len(current_route) - 1, 1)

    tabu_list = {}   # pasangan edge baru (frozenset) -> iterasi kadaluarsa

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

        neighbors = get_neighbors(current_route, d, current_distance)
        if not neighbors:
            break   # rute terlalu pendek (< 4 kota): tidak ada move 2-opt

        edge_expiry = tabu_edge_expiry(tabu_list)

        candidate_log = []
        best_candidate = None
        best_idx = None

        # Step 4-5: evaluasi seluruh neighborhood
        for idx, nb in enumerate(neighbors):
            i, j = nb["i"], nb["j"]

            # tabu jika salah satu edge yang dibuang masih tabu
            removed = removed_edges(current_route, i, j)
            remaining = tabu_remaining(removed, edge_expiry, iteration)
            nb["tabu_remaining"] = remaining
            is_tabu = remaining > 0
            aspiration_met = nb["distance"] < best_distance - eps
            admissible = (not is_tabu) or aspiration_met

            if verbose_history:
                candidate_log.append({
                    "i": i,
                    "j": j,
                    "delta": nb["delta"],
                    "distance": nb["distance"],
                    "is_tabu": is_tabu,
                    "tabu_remaining": remaining,
                    "aspiration_met": aspiration_met,
                    # aspiration hanya bermakna kalau move tabu
                    "aspiration_relevant": is_tabu,
                    "admissible": admissible,
                    "selected": False,
                })

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

        i, j = best_candidate["i"], best_candidate["j"]
        move = (best_candidate["delta"], i, j)

        # attribute tabu dan detail move dihitung dari rute SEBELUM move
        new_attr = added_edges(current_route, i, j)
        detail = describe_2opt_move(current_route, d, move)
        detail["i"], detail["j"] = i, j
        detail["distance_before"] = current_distance
        detail["aspiration_by_default"] = by_default

        current_route = apply_2opt(current_route, move)
        # hitung ulang dari rute (bukan akumulasi delta) supaya tidak drift
        current_distance = tour_distance(current_route, d)
        detail["distance_after"] = current_distance

        # update tabu list: tambah edge baru, buang yang sudah kadaluarsa
        tabu_list[new_attr] = iteration + tabu_tenure
        tabu_list = {k: v for k, v in tabu_list.items() if v >= iteration}

        if current_distance < best_distance - eps:
            best_distance = current_distance
            best_route = current_route.copy()
            best_iteration = iteration

        history.append({
            "iteration": iteration,
            "route": current_route.copy(),
            "distance": current_distance,
            "move": detail,
            "tabu_list": tabu_list.copy(),
            "candidates": candidate_log,
        })

    # =========================
    # 3. Return Result
    # =========================

    return {
        "route": best_route,
        "distance": best_distance,
        "best_iteration": best_iteration,
        "history": history,
    }
