
import random
import math

def euclidean_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x1 - x2
    dy = y1 - y2
    return math.sqrt(dx * dx + dy * dy)

def manhattan_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)

def build_distance_matrix(coords, metric="euclidean"):
    n = len(coords)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if metric == "euclidean":
                matrix[i][j] = euclidean_distance(coords[i], coords[j])
            elif metric == "manhattan":
                matrix[i][j] = manhattan_distance(coords[i], coords[j])
            else:
                raise ValueError("metric harus 'euclidean' atau 'manhattan'")
    return matrix

def tour_distance(tour, dist_matrix):
    total = 0.0
    for k in range(len(tour) - 1):
        a = tour[k]
        b = tour[k + 1]
        total += dist_matrix[a][b]
    return total

def generate_initial_tour(n, home=0, seed=None):
    rng = random.Random(seed)
    others = [c for c in range(n) if c != home]
    rng.shuffle(others)
    return [home] + others + [home]

def validate_tour(tour, n, home=0):
    if len(tour) != n + 1:
        raise ValueError("Tur closed harus memiliki n + 1 posisi.")

    if tour[0] != home or tour[-1] != home:
        raise ValueError(
            f"Tur harus dimulai dan berakhir di home (city {home})."
        )

    if sorted(tour[:-1]) != list(range(n)):
        raise ValueError(
            "Tur harus memuat semua kota tepat satu kali, "
            "kecuali home yang muncul kembali di akhir."
        )

    return True
