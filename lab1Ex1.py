import numpy as np
from collections import Counter

RNG = np.random.default_rng()

def one_trial(rng):

    counts = {"R": 3, "B": 4, "K": 2}
    d = rng.integers(1, 7) #zar 1-6

    if d in (2, 3, 5): #nr prim
        counts["K"] += 1
    elif d == 6:
        counts["R"] += 1
    else:   #1 sau 4
        counts["B"] += 1

    total = counts["R"] + counts["B"] + counts["K"]
    probs = [counts["R"]/total, counts["B"]/total, counts["K"]/total]
    draw = rng.choice(["R", "B", "K"], p=probs)
    return draw

#simulare
def simulate_red(n_trials=200_000):
    rng = np.random.default_rng()
    draws = [one_trial(rng) for _ in range(n_trials)]
    freqs = Counter(draws)
    return freqs["R"] / n_trials

#calcul real
def theoretical_red():
    return (3/6)*(3/10) + (1/6)*(4/10) + (2/6)*(3/10)  #formula prob totale
p1 = simulate_red()
p2 = theoretical_red()

print(f"prob simulat = {p1:.6f}")
print(f"prob calculat = {p2:.6f}")
