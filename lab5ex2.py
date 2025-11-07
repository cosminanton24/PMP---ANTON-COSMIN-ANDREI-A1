

import numpy as np
from scipy.stats import gamma

T = 10
y_sum = 180

alpha0 = 1e-3
beta0  = 1e-3

#(a)
alpha_post = alpha0 + y_sum
beta_post  = beta0  + T

print(f"(a) Posterior: λ | data ~ Gamma(alpha={alpha_post:.6f}, beta={beta_post:.6f})")

#(b)
def gamma_hdi(alpha, beta, cred_mass=0.94, grid=20001):
    ps = np.linspace(0.0, 1.0 - cred_mass, grid)
    ql = gamma.ppf(ps, a=alpha, scale=1.0/beta)
    qu = gamma.ppf(ps + cred_mass, a=alpha, scale=1.0/beta)
    widths = qu - ql
    i = np.argmin(widths)
    return float(ql[i]), float(qu[i])

l_hdi, u_hdi = gamma_hdi(alpha_post, beta_post, cred_mass=0.94)
print(f"(b) 94% HDI: [{l_hdi:.4f}, {u_hdi:.4f}]")

#(c)
map_lambda = (alpha_post - 1) / beta_post if alpha_post > 1 else 0.0
print(f"(c) MAP (most probable λ): {map_lambda:.4f}")

