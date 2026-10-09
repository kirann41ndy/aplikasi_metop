import numbers

from .base import tour_distance, validate_tour

EPS_REL = 1e-9
EPS = 1e-9
STRATEGIES = ("Best", "First")


# Helper bersama
def prepare(initial_route, dist_matrix, max_iter, strategy):
    """Validasi input: kembalikan salinan tur awal (input tidak dimutasi)."""
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy harus salah satu dari {STRATEGIES}.")

    n = len(dist_matrix)
    if any(len(row) != n for row in dist_matrix):
        raise ValueError("dist_matrix harus berbentuk persegi (n x n).")

    if (
        isinstance(max_iter, bool)
        or not isinstance(max_iter, numbers.Integral)
        or max_iter < 0
    ):
        raise ValueError("max_iter harus integer >= 0.")

    route = list(initial_route)
    # start/home = node pertama tur awal, tidak pernah dipindahkan
    validate_tour(route, n, home=route[0] if route else 0)
    return route


def edges(pairs, d):
    """Daftar edge beserta jaraknya: [{"edge": (u, v), "distance": d[u][v]}]."""
    return [{"edge": (u, v), "distance": d[u][v]} for u, v in pairs]


def move_cost(removed, added):
    """Total jarak edge dilepas, total jarak edge baru, dan delta (baru - lama)."""
    removed_cost = sum(item["distance"] for item in removed)
    added_cost = sum(item["distance"] for item in added)
    return removed_cost, added_cost, added_cost - removed_cost


def local_search(
    route, dist_matrix, max_iter, strategy,
    find_move, apply_move, describe_move
):
    first_improvement = strategy == "First"
    best_route = route
    best_distance = tour_distance(best_route, dist_matrix)

    # Toleransi relatif: rata-rata panjang edge tur awal x EPS_REL
    eps = EPS_REL * best_distance / max(len(best_route) - 1, 1)

    history = [{
        "iteration": 0,
        "route": best_route.copy(),
        "distance": best_distance,
        "move": None,
    }]

    for iteration in range(1, int(max_iter) + 1):
        move = find_move(best_route, dist_matrix, first_improvement, eps)
        if move is None:
            break  # local optimum

        # Detail dihitung dari tur SEBELUM move diterapkan
        detail = describe_move(best_route, dist_matrix, move)
        detail["distance_before"] = best_distance

        best_route = apply_move(best_route, move)
        best_distance = tour_distance(best_route, dist_matrix)
        detail["distance_after"] = best_distance

        history.append({
            "iteration": iteration,
            "route": best_route.copy(),
            "distance": best_distance,
            "move": detail,
        })

    return {
        "route": best_route,
        "distance": best_distance,
        "history": history,
    }


# 2-opt
def find_2opt_move(route, d, first_improvement=False, eps=EPS):
    n = len(route) - 1
    best = None
    best_delta = -eps

    for i in range(1, n - 1):
        a, b = route[i - 1], route[i]
        for j in range(i + 1, n):
            c, e = route[j], route[j + 1]
            delta = d[a][c] + d[b][e] - d[a][b] - d[c][e]
            if delta < best_delta:
                best_delta, best = delta, (delta, i, j)
                if first_improvement:
                    return best
    return best


def apply_2opt(route, move):
    _, i, j = move
    return route[:i] + route[i:j + 1][::-1] + route[j + 1:]


def describe_2opt_move(route, d, move):
    _, i, j = move
    a, b = route[i - 1], route[i]
    c, e = route[j], route[j + 1]

    removed = edges([(a, b), (c, e)], d)
    added = edges([(a, c), (b, e)], d)
    removed_cost, added_cost, delta = move_cost(removed, added)

    return {
        "type": "2-opt",
        "positions": {"i": i, "j": j},
        "reversed_segment": route[i:j + 1],  # urutan sebelum dibalik
        "removed_edges": removed,
        "added_edges": added,
        "removed_cost": removed_cost,
        "added_cost": added_cost,
        "delta": delta,
    }


def two_opt(initial_route, dist_matrix, max_iter=100, strategy="Best"):
    route = prepare(initial_route, dist_matrix, max_iter, strategy)
    return local_search(
        route, dist_matrix, max_iter, strategy,
        find_2opt_move, apply_2opt, describe_2opt_move
    )


# 3-opt
def find_3opt_move(route, d, first_improvement=False, eps=EPS):
    n = len(route) - 1
    best = None
    best_delta = -eps

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
                    if delta < best_delta:
                        best_delta = delta
                        best = (delta, i, j, k, variant)
                        if first_improvement:
                            return best
    return best


def apply_3opt(route, move):
    _, i, j, k, variant = move
    A, B, C, D = route[:i], route[i:j], route[j:k], route[k:]
    if variant == 0:
        return A + B[::-1] + C + D
    if variant == 1:
        return A + B + C[::-1] + D
    if variant == 2:
        return A + C[::-1] + B[::-1] + D
    if variant == 3:
        return A + B[::-1] + C[::-1] + D
    if variant == 4:
        return A + C + B + D
    if variant == 5:
        return A + C + B[::-1] + D
    if variant == 6:
        return A + C[::-1] + B + D
    raise ValueError(f"variant tidak dikenal: {variant}")


# Label penyambungan ulang per variant (A, B, C, D = segmen tur; X' = dibalik)
RECONNECTIONS_3OPT = (
    "A B' C D",     # 0 (setara 2-opt)
    "A B C' D",     # 1 (setara 2-opt)
    "A C' B' D",    # 2 (setara 2-opt)
    "A B' C' D",    # 3
    "A C B D",      # 4 (tukar segmen)
    "A C B' D",     # 5
    "A C' B D",     # 6
)


def describe_3opt_move(route, d, move):
    _, i, j, k, variant = move
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

    removed = edges([(a, b1), (b2, c1), (c2, e)], d)
    added = edges(added_pairs, d)
    removed_cost, added_cost, delta = move_cost(removed, added)

    return {
        "type": "3-opt",
        "positions": {"i": i, "j": j, "k": k},
        "variant": variant,
        "reconnection": RECONNECTIONS_3OPT[variant],
        "segments": {"B": route[i:j], "C": route[j:k]},
        "removed_edges": removed,
        "added_edges": added,
        "removed_cost": removed_cost,
        "added_cost": added_cost,
        "delta": delta,
    }


def three_opt(initial_route, dist_matrix, max_iter=100, strategy="Best"):
    route = prepare(initial_route, dist_matrix, max_iter, strategy)
    return local_search(
        route, dist_matrix, max_iter, strategy,
        find_3opt_move, apply_3opt, describe_3opt_move
    )