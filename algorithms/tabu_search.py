"""
Nama Algoritma: Tabu Search
Author: Indy
Deskripsi singkat: Tabu Search untuk TSP. Neighborhood dibentuk pakai 2-opt
                    (reverse segmen rute). Tiap iterasi, SEMUA kandidat 2-opt
                    dievaluasi, dipilih yang terbaik di antara yang tidak tabu
                    (atau yang tabu tapi memenuhi aspiration criterion --
                    lebih baik dari solusi terbaik sejauh ini).
"""

from .base import tour_distance


def get_neighbors(route):
    """Semua kandidat 2-opt dari satu rute (closed tour: [home, ..., home])."""
    neighbors = []
    for i in range(1, len(route) - 2):
        for j in range(i + 1, len(route) - 1):
            new_route = list(route)
            new_route[i:j + 1] = reversed(new_route[i:j + 1])
            neighbors.append({"route": new_route, "i": i, "j": j})
    return neighbors


def removed_edges(route, i, j):
    """
    Edge (pasangan kota) yang DIBUANG oleh move (i, j), dihitung dari rute
    SAAT INI (sebelum move dijalankan). Dipakai buat CEK apakah kandidat
    ini boleh dijalankan (dibandingkan dengan tabu_list).
    """
    e1 = frozenset((route[i - 1], route[i]))
    e2 = frozenset((route[j], route[j + 1]))
    return frozenset((e1, e2))


def added_edges(route, i, j):
    """
    Edge (pasangan kota) yang TERBENTUK oleh move (i, j), dihitung dari
    rute SAAT INI (sebelum move dijalankan) -- yaitu edge baru yang akan
    ada setelah segmen [i..j] dibalik.

    Inilah yang DISIMPAN ke tabu_list (bukan removed_edges!). Alasannya:
    "gerakan balik lagi" (undo) ke rute sebelumnya = membongkar PERSIS
    edge yang baru saja ditambahkan ini. Kalau yang disimpan malah edge
    LAMA yang dibuang, larangan itu tidak akan pernah match, karena edge
    lama itu sudah tidak ada lagi di rute manapun -- sehingga algoritma
    bebas bolak-balik ke solusi yang sama tanpa pernah ketangkap tabu.
    """
    e1 = frozenset((route[i - 1], route[j]))
    e2 = frozenset((route[i], route[j + 1]))
    return frozenset((e1, e2))


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
    Menjalankan algoritma Tabu Search untuk TSP.

    Step 1 : home city = initial_route[0] (dan initial_route[-1], closed tour).
    Step 2 : initial_route jadi current solution. Total distance-nya jadi
             aspiration level awal. tabu_list mulai kosong.
    Step 3 : i = 1 (-> loop for iteration di bawah)
    Step 4 : tukar 2 arc jadi 2 arc baru (2-opt, lihat get_neighbors) ->
             hitung total distance tiap kandidat.
    Step 5 : evaluasi SEMUA kandidat di neighborhood, pilih yang admissible
             terbaik jadi current solution, update tabu_list.
    Step 6 : i += 1, ulangi Step 4.
    Step 7 : stop kalau iterasi == max_iter. Rute terbaik = hasil akhir.

    Larangan tabu dipasang di EDGE BARU yang terbentuk tiap move (lihat
    added_edges di atas), supaya gerakan "balik lagi" ke rute sebelumnya
    -- yang berarti membongkar edge baru itu -- bisa ketahuan dan ditolak
    selama tabu_tenure iterasi (kecuali aspiration criterion terpenuhi).

    Parameters
    ----------
    initial_route : list
        Rute awal (closed tour: [home, ..., home]).
    dist_matrix : list
        Matriks jarak.
    max_iter : int
        Maksimum iterasi.
    tabu_tenure : int
        Berapa iterasi sebuah move dilarang diulang sebelum boleh dipakai lagi.
    verbose_history : bool
        False (default) -> history["candidates"] selalu [] (ringan, cukup
        buat tab "Pilihan Metode").
        True  -> history["candidates"] diisi SEMUA kandidat 2-opt yang
        dievaluasi tiap iterasi, lengkap status tabu & aspiration-nya.
        Dipakai buat tab "Ilustrasi" yang butuh nunjukin proses milihnya,
        bukan cuma hasil akhir.

    Returns
    -------
    dict
        {
            "route": rute terbaik,
            "distance": total jarak,
            "history": riwayat iterasi
        }
    """

    # =========================
    # 1. Initialization
    # =========================

    current_route = list(initial_route)
    current_distance = tour_distance(current_route, dist_matrix)

    best_route = list(current_route)
    best_distance = current_distance          # aspiration level awal (Step 2)

    tabu_list = {}   # atribut (added_edges) -> iterasi kadaluarsa

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

        neighbors = get_neighbors(current_route)   # Step 4: semua kandidat 2-opt

        candidate_log = []
        best_candidate = None
        best_candidate_distance = None
        best_candidate_attr = None

        # Step 5: evaluasi seluruh neighborhood, pilih admissible terbaik
        for nb in neighbors:

            # edge yang akan DIBUANG kalau kandidat ini dijalankan --
            # dicek terhadap tabu_list (yang isinya edge BARU dari move
            # sebelumnya). Match -> kandidat ini akan membongkar edge
            # yang baru saja ditambahkan, berarti ini gerakan balik lagi.
            check_attr = removed_edges(current_route, nb["i"], nb["j"])
            # edge yang akan TERBENTUK kalau kandidat ini dijalankan --
            # inilah yang disimpan ke tabu_list kalau kandidat ini kepilih.
            new_attr = added_edges(current_route, nb["i"], nb["j"])

            d = tour_distance(nb["route"], dist_matrix)

            is_tabu = check_attr in tabu_list and tabu_list[check_attr] >= iteration
            aspiration_met = d < best_distance          # aspiration criterion
            admissible = (not is_tabu) or aspiration_met

            if verbose_history:
                candidate_log.append({
                    "i": nb["i"],
                    "j": nb["j"],
                    "distance": d,
                    "is_tabu": is_tabu,
                    "aspiration_met": aspiration_met,
                    "admissible": admissible,
                })

            if admissible and (best_candidate is None or d < best_candidate_distance):
                best_candidate = nb
                best_candidate_distance = d
                best_candidate_attr = new_attr

        # fallback langka: semua kandidat tabu & tak satupun aspiration_met
        if best_candidate is None:
            best_candidate = min(neighbors, key=lambda nb: tour_distance(nb["route"], dist_matrix))
            best_candidate_distance = tour_distance(best_candidate["route"], dist_matrix)
            best_candidate_attr = added_edges(current_route, best_candidate["i"], best_candidate["j"])

        selected_move = {"i": best_candidate["i"], "j": best_candidate["j"]}

        current_route = best_candidate["route"]
        current_distance = best_candidate_distance

        # update tabu list: tambah EDGE BARU yang baru dibentuk, buang yang kadaluarsa
        tabu_list[best_candidate_attr] = iteration + tabu_tenure
        tabu_list = {k: v for k, v in tabu_list.items() if v >= iteration}

        if current_distance < best_distance:
            best_distance = current_distance
            best_route = current_route.copy()

        history.append({
            "iteration": iteration,
            "route": current_route.copy(),
            "distance": current_distance,
            "move": selected_move,
            "tabu_list": tabu_list.copy(),
            "candidates": candidate_log,
        })

    # =========================
    # 3. Return Result
    # =========================

    return {
        "route": best_route,
        "distance": best_distance,
        "history": history
    }
