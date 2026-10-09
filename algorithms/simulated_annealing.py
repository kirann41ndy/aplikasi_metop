
import random
import math
import numbers

from .base import (
    tour_distance,
    validate_tour,
    generate_initial_tour
)


_EPS = 1e-9


def _prepare(initial_route, dist_matrix, max_iter, home):
    """Validasi input dan menyiapkan rute awal."""

    n = len(dist_matrix)

    if n == 0 or any(len(row) != n for row in dist_matrix):
        raise ValueError("dist_matrix harus berbentuk persegi (n x n).")

    if (
        isinstance(max_iter, bool)
        or not isinstance(max_iter, numbers.Integral)
        or max_iter < 0
    ):
        raise ValueError("max_iter harus integer >= 0.")

    if home is not None and (
        isinstance(home, bool)
        or not isinstance(home, numbers.Integral)
        or not 0 <= home < n
    ):
        raise ValueError(f"home harus berupa indeks dari 0 sampai {n - 1}.")

    if initial_route is None:
        if home is None:
            home = 0

        route = generate_initial_tour(n, home=home)

    else:
        route = list(initial_route)

        if not route:
            raise ValueError("initial_route tidak boleh kosong.")

        if home is None:
            home = route[0]

    validate_tour(route, n, home=home)

    return route, home


def get_neighbors(route):
    """Menghasilkan tetangga menggunakan perpindahan 2-opt."""

    neighbors = []

    for i in range(1, len(route) - 2):
        for j in range(i + 1, len(route) - 1):

            new_route = list(route)
            new_route[i:j + 1] = reversed(new_route[i:j + 1])

            neighbors.append({
                "route": new_route,
                "i": i,
                "j": j
            })

    return neighbors


def simulated_annealing(
    dist_matrix,
    initial_route=None,
    home=None,
    initial_temp=1000,
    cooling_rate=0.95,
    min_temp=0.01,
    max_iter=100,
    seed=None,
):
    """Mencari rute terbaik menggunakan Simulated Annealing."""

    if (
        isinstance(initial_temp, bool)
        or not isinstance(initial_temp, numbers.Real)
        or not math.isfinite(initial_temp)
        or initial_temp <= 0
    ):
        raise ValueError("initial_temp harus berupa angka positif.")

    if (
        isinstance(cooling_rate, bool)
        or not isinstance(cooling_rate, numbers.Real)
        or not math.isfinite(cooling_rate)
        or not 0 < cooling_rate < 1
    ):
        raise ValueError("cooling_rate harus di antara 0 dan 1.")

    if (
        isinstance(min_temp, bool)
        or not isinstance(min_temp, numbers.Real)
        or not math.isfinite(min_temp)
        or min_temp < 0
    ):
        raise ValueError("min_temp harus berupa angka >= 0.")

    current_route, home = _prepare(
        initial_route,
        dist_matrix,
        max_iter,
        home
    )

    rng = random.Random(seed)

    current_distance = tour_distance(
        current_route, dist_matrix
    )

    best_route = list(current_route)
    best_distance = current_distance

    temperature = initial_temp

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

    for iteration in range(1, int(max_iter) + 1):

        if temperature <= min_temp:
            break

        neighbors = get_neighbors(current_route)

        if not neighbors:
            break

        candidate = rng.choice(neighbors)
        candidate_route = candidate["route"]

        candidate_distance = tour_distance(
            candidate_route, dist_matrix
        )

        delta = candidate_distance - current_distance

        if delta <= 0:
            probability = 1.0
        else:
            probability = math.exp(-delta / temperature)

        accepted = rng.random() < probability

        if accepted:
            current_route = list(candidate_route)
            current_distance = candidate_distance

        if current_distance < best_distance:
            best_route = list(current_route)
            best_distance = current_distance

        history.append({
            "iteration": iteration,
            "route": list(current_route),
            "distance": current_distance,
            "best_distance": best_distance,
            "temperature": temperature,
            "move": {
                "type": "2-opt",
                "positions": {
                    "i": candidate["i"],
                    "j": candidate["j"],
                },
            },
            "delta": delta,
            "probability": probability,
            "accepted": accepted,
        })

        temperature *= cooling_rate

    return {
        "route": best_route,
        "distance": best_distance,
        "history": history,
    }
    
