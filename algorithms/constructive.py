# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# Import Library

# %% colab={"base_uri": "https://localhost:8080/", "height": 384} id="AgCZkt-3rCx-" outputId="bd2e7d4e-50ce-4433-dc0a-6d7b2ae62eef"
import random
from .base import build_distance_matrix, tour_distance


# Validasi Start Node

def validate_start_node(start_node, n):
    if not isinstance(start_node, int):
        raise TypeError(
            "start_node harus berupa integer."
        )

    if start_node < 0 or start_node >= n:
        raise ValueError(
            f"start_node harus berada pada rentang 0 sampai {n - 1}."
        )


# Nearest Neighbor

def nearest_neighbor(distance_matrix, start_node):
    n = len(distance_matrix)

    validate_start_node(start_node, n)

    current_route = [start_node]
    visited = {start_node}
    current_node = start_node

    history = []

    # Initial state
    history.append({
        "iteration": 0,
        "route": current_route.copy(),
        "distance": 0
    })

    iteration = 0

    while len(visited) < n:

        candidates = [
            node
            for node in range(n)
            if node not in visited
        ]

        # Pilih node terdekat dari current node
        # Jika jarak sama, pilih node dengan index lebih kecil
        next_node = min(
            candidates,
            key=lambda node: (
                distance_matrix[current_node][node],
                node
            )
        )

        current_route.append(next_node)
        visited.add(next_node)
        current_node = next_node

        iteration += 1

        # Hitung distance sementara dengan kembali
        # ke start node
        current_distance = tour_distance(
            current_route + [start_node],
            distance_matrix
        )

        history.append({
            "iteration": iteration,
            "route": current_route.copy(),
            "distance": current_distance
        })

    # Tutup tour
    current_route.append(start_node)

    current_distance = tour_distance(
        current_route,
        distance_matrix
    )

    iteration += 1

    history.append({
        "iteration": iteration,
        "route": current_route.copy(),
        "distance": current_distance
    })

    return {
        "route": current_route,
        "distance": current_distance,
        "history": history
    }


# Nearest Insertion

def nearest_insertion(distance_matrix, start_node):
    n = len(distance_matrix)

    validate_start_node(start_node, n)

    # Pilih node terdekat dari start node
    candidates = [
        node
        for node in range(n)
        if node != start_node
    ]

    nearest_node = min(
        candidates,
        key=lambda node: (
            distance_matrix[start_node][node],
            node
        )
    )

    # Initial closed tour
    current_route = [
        start_node,
        nearest_node,
        start_node
    ]

    unvisited = (
        set(range(n))
        - {start_node, nearest_node}
    )

    history = []

    # Initial state
    current_distance = tour_distance(
        current_route,
        distance_matrix
    )

    history.append({
        "iteration": 0,
        "route": current_route.copy(),
        "distance": current_distance
    })

    iteration = 0

    while unvisited:

        # Pilih node unvisited yang paling dekat
        # dengan current tour
        selected_node = min(
            unvisited,
            key=lambda node: (
                min(
                    distance_matrix[node][tour_node]
                    for tour_node in current_route[:-1]
                ),
                node
            )
        )

        best_position = None
        best_increase = float("inf")

        # Cari posisi insertion dengan
        # minimum tambahan jarak
        for i in range(len(current_route) - 1):

            node_i = current_route[i]
            node_j = current_route[i + 1]

            increase = (
                distance_matrix[node_i][selected_node]
                + distance_matrix[selected_node][node_j]
                - distance_matrix[node_i][node_j]
            )

            if (
                increase < best_increase
                or (
                    increase == best_increase
                    and (
                        best_position is None
                        or i + 1 < best_position
                    )
                )
            ):
                best_increase = increase
                best_position = i + 1

        current_route.insert(
            best_position,
            selected_node
        )

        unvisited.remove(selected_node)

        iteration += 1

        current_distance = tour_distance(
            current_route,
            distance_matrix
        )

        history.append({
            "iteration": iteration,
            "route": current_route.copy(),
            "distance": current_distance
        })

    return {
        "route": current_route,
        "distance": current_distance,
        "history": history
    }


# Farthest Insertion

def farthest_insertion(distance_matrix, start_node):
    n = len(distance_matrix)

    validate_start_node(start_node, n)

    # Pilih node terjauh dari start node
    candidates = [
        node
        for node in range(n)
        if node != start_node
    ]

    farthest_node = max(
        candidates,
        key=lambda node: (
            distance_matrix[start_node][node],
            -node
        )
    )

    # Initial closed tour
    current_route = [
        start_node,
        farthest_node,
        start_node
    ]

    unvisited = (
        set(range(n))
        - {start_node, farthest_node}
    )

    history = []

    # Initial state
    current_distance = tour_distance(
        current_route,
        distance_matrix
    )

    history.append({
        "iteration": 0,
        "route": current_route.copy(),
        "distance": current_distance
    })

    iteration = 0

    while unvisited:

        # Pilih node yang paling jauh
        # dari current tour
        selected_node = max(
            unvisited,
            key=lambda node: (
                min(
                    distance_matrix[node][tour_node]
                    for tour_node in current_route[:-1]
                ),
                -node
            )
        )

        best_position = None
        best_increase = float("inf")

        # Cari posisi insertion dengan
        # minimum tambahan jarak
        for i in range(len(current_route) - 1):

            node_i = current_route[i]
            node_j = current_route[i + 1]

            increase = (
                distance_matrix[node_i][selected_node]
                + distance_matrix[selected_node][node_j]
                - distance_matrix[node_i][node_j]
            )

            if (
                increase < best_increase
                or (
                    increase == best_increase
                    and (
                        best_position is None
                        or i + 1 < best_position
                    )
                )
            ):
                best_increase = increase
                best_position = i + 1

        current_route.insert(
            best_position,
            selected_node
        )

        unvisited.remove(selected_node)

        iteration += 1

        current_distance = tour_distance(
            current_route,
            distance_matrix
        )

        history.append({
            "iteration": iteration,
            "route": current_route.copy(),
            "distance": current_distance
        })

    return {
        "route": current_route,
        "distance": current_distance,
        "history": history
    }


# Arbitrary Insertion
def arbitrary_insertion(dist_matrix, start=0, seed=None):
    n = len(dist_matrix)

    validate_start_node(start, n)

    rng = random.Random(seed)

    # Pilih node kedua secara acak
    candidates = [
        node
        for node in range(n)
        if node != start
    ]

    second_node = rng.choice(candidates)

    # Initial closed tour
    route = [
        start,
        second_node,
        start
    ]

    unvisited = (
        set(range(n))
        - {start, second_node}
    )

    history = []

    # Initial state
    distance = tour_distance(route, dist_matrix)

    history.append({
        "iteration": 0,
        "route": route.copy(),
        "distance": distance
    })

    iteration = 0

    while unvisited:

        # Pilih node unvisited secara acak
        # (sorted agar hasil reproducible dengan seed)
        selected_node = rng.choice(
            sorted(unvisited)
        )

        best_position = None
        best_increase = float("inf")

        # Cari posisi insertion dengan
        # minimum tambahan jarak
        for i in range(len(route) - 1):

            node_i = route[i]
            node_j = route[i + 1]

            increase = (
                dist_matrix[node_i][selected_node]
                + dist_matrix[selected_node][node_j]
                - dist_matrix[node_i][node_j]
            )

            if (
                increase < best_increase
                or (
                    increase == best_increase
                    and (
                        best_position is None
                        or i + 1 < best_position
                    )
                )
            ):
                best_increase = increase
                best_position = i + 1

        route.insert(
            best_position,
            selected_node
        )

        unvisited.remove(selected_node)

        iteration += 1

        distance = tour_distance(route, dist_matrix)

        history.append({
            "iteration": iteration,
            "route": route.copy(),
            "distance": distance
        })

    return {
        "route": route,
        "distance": distance,
        "history": history
    }