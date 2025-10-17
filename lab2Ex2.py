import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from math import comb

N_SIM = 10_000
rng = np.random.default_rng()

def play_once(rng):
    S = rng.integers(0, 2)
    n = rng.integers(1, 7)
    p_heads = (4/7) if S == 0 else (1/2)
    m = rng.binomial(2*n, p_heads)
    starter_wins = (n >= m)
    winner = S if starter_wins else (1 - S)
    return winner

wins = np.array([play_once(rng) for _ in range(N_SIM)])
p0_win = np.mean(wins == 0)
p1_win = 1 - p0_win
print(f"[Simulare 10k] P0 win ≈ {p0_win:.4f} | P1 win ≈ {p1_win:.4f} | Castiga mai des: {'P0' if p0_win>p1_win else 'P1'}")

model = DiscreteBayesianNetwork([('S','M'), ('N','M')])
cpd_S = TabularCPD('S', 2, [[0.5], [0.5]])
cpd_N = TabularCPD('N', 6, [[1/6]]*6)

rows_M = 13
values = []
for m in range(rows_M):
    col_probs = []
    for s in (0, 1):
        for n_idx in range(6):
            n_val = n_idx + 1
            flips = 2 * n_val
            p = (4/7) if s == 0 else (1/2)
            if m <= flips:
                col_probs.append(comb(flips, m) * (p**m) * ((1-p)**(flips - m)))
            else:
                col_probs.append(0.0)
    values.append(col_probs)

cpd_M = TabularCPD('M', 13, values, evidence=['S', 'N'], evidence_card=[2, 6])
model.add_cpds(cpd_S, cpd_N, cpd_M)
assert model.check_model()

infer = VariableElimination(model)
post_S_given_M1 = infer.query(variables=['S'], evidence={'M': 1})
p_S0 = float(post_S_given_M1.values[0])
p_S1 = float(post_S_given_M1.values[1])
starter = 'P0' if p_S0 > p_S1 else 'P1'
print(f"[Inferenta BN] P(S=P0 | M=1) = {p_S0:.4f} | P(S=P1 | M=1) = {p_S1:.4f} | Mai probabil starter: {starter}")
