import os, sys, typing
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
try:
    import coverage.types as ct
    if not hasattr(ct, "Tracer"):
        class Tracer: ...
        ct.Tracer = Tracer
    if not hasattr(ct, "TShouldTraceFn"):
        ct.TShouldTraceFn = typing.Callable
    if not hasattr(ct, "TShouldStartContextFn"):
        ct.TShouldStartContextFn = typing.Callable
except Exception:
    pass

if "numba" in sys.modules:
    del sys.modules["numba"]

import pymc as pm

az.style.use("arviz-darkgrid")


def load_xy(path):
    data = np.loadtxt(path)
    x = data[:, 0]
    y = data[:, 1]
    return x, y


def make_features(x, order):
    x_p = np.vstack([x**i for i in range(1, order + 1)])
    x_s = (x_p - x_p.mean(axis=1, keepdims=True)) / x_p.std(axis=1, keepdims=True)
    return x_s


def standardize_y(y):
    return (y - y.mean()) / y.std()


def fit_poly_model(x_s, y_s, order, beta_sigma, draws=1500, tune=1500, chains=2, target_accept=0.9, seed=123):
    with pm.Model() as model_p:
        alpha = pm.Normal("alpha", mu=0, sigma=1)
        beta = pm.Normal("beta", mu=0, sigma=beta_sigma, shape=order)
        eps = pm.HalfNormal("eps", 5)
        mu = alpha + pm.math.dot(beta, x_s)
        y_pred = pm.Normal("y_pred", mu=mu, sigma=eps, observed=y_s)
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=seed,
            target_accept=target_accept,
            progressbar=True,
            return_inferencedata=True,
        )
    return model_p, idata


def plot_posterior_mean_curve(x_s, y_s, idata, label, ax=None):
    if ax is None:
        ax = plt.gca()
    alpha_post = idata.posterior["alpha"].mean(("chain", "draw")).values
    beta_post = idata.posterior["beta"].mean(("chain", "draw")).values
    idx = np.argsort(x_s[0])
    y_post = alpha_post + np.dot(beta_post, x_s)
    ax.plot(x_s[0][idx], y_post[idx], label=label)
    ax.scatter(x_s[0], y_s, marker=".", alpha=0.8)
    ax.set_xlabel("x (standardized)")
    ax.set_ylabel("y (standardized)")
    ax.legend()
    return ax


def generate_more_data_from_fit(x, y, n=500, deg=2, seed=123):
    rng = np.random.default_rng(seed)
    coeffs = np.polyfit(x, y, deg=deg)
    y_hat = np.polyval(coeffs, x)
    sigma = np.std(y - y_hat)

    x_new = rng.uniform(x.min(), x.max(), size=n)
    y_new = np.polyval(coeffs, x_new) + rng.normal(0, sigma, size=n)
    return x_new, y_new



# Subpunctul 1
csv_path = "/mnt/data/date.csv"
x_1, y_1 = load_xy(csv_path)

order = 5
x_1s = make_features(x_1, order)
y_1s = standardize_y(y_1)

# 1.a) inference cu model_p si plot curba (order=5)
model_p_10, idata_p_10 = fit_poly_model(x_1s, y_1s, order=order, beta_sigma=10, seed=11)

plt.figure(figsize=(9, 4))
plot_posterior_mean_curve(x_1s, y_1s, idata_p_10, label=f"order={order}, beta sigma=10")
plt.title("1.a) model_p (order=5), beta sigma=10")
plt.show()

# 1.b) repeta cu beta sd=100 si cu sd=np.array([10, 0.1, 0.1, 0.1, 0.1])
model_p_100, idata_p_100 = fit_poly_model(x_1s, y_1s, order=order, beta_sigma=100, seed=12)

beta_sd_vec = np.array([10, 0.1, 0.1, 0.1, 0.1])
model_p_vec, idata_p_vec = fit_poly_model(x_1s, y_1s, order=order, beta_sigma=beta_sd_vec, seed=13)

plt.figure(figsize=(10, 4))
ax = plt.gca()
plot_posterior_mean_curve(x_1s, y_1s, idata_p_10, label="sigma=10", ax=ax)
plot_posterior_mean_curve(x_1s, y_1s, idata_p_100, label="sigma=100", ax=ax)
plot_posterior_mean_curve(x_1s, y_1s, idata_p_vec, label="sigma=[10,0.1,0.1,0.1,0.1]", ax=ax)
plt.title("1.b) comparatie curbe pentru priors pe beta")
plt.show()

# Raspuns (1.b):
# sigma=100 regularizeaza mult mai putin -> coeficientii pot deveni mari -> curba tinde sa fie mai ondulata (mai mult overfitting).
# sigma=[10,0.1,0.1,0.1,0.1] regularizeaza puternic termenii de ordin mai mare -> curba e mai neteda, mai aproape de un model de ordin mic.


# Subpunctul 2
# Repeta exercitiul anterior, dar cu 500 puncte de date
x_500, y_500 = generate_more_data_from_fit(x_1, y_1, n=500, deg=2, seed=21)

order = 5
x_500s = make_features(x_500, order)
y_500s = standardize_y(y_500)

model_p_500, idata_p_500 = fit_poly_model(x_500s, y_500s, order=order, beta_sigma=10, seed=22)

plt.figure(figsize=(9, 4))
plot_posterior_mean_curve(x_500s, y_500s, idata_p_500, label=f"order={order}, n=500, sigma=10")
plt.title("2) model_p (order=5) pe 500 puncte")
plt.show()


# Subpunctul 3
# Inference cu model cubic (order=3), calculeaza WAIC si LOO, ploteaza si compara cu linear si quadratic
def fit_linear_model(x_s1, y_s, beta_sigma=10, draws=1500, tune=1500, chains=2, target_accept=0.9, seed=31):
    with pm.Model() as model_l:
        alpha = pm.Normal("alpha", mu=0, sigma=1)
        beta = pm.Normal("beta", mu=0, sigma=beta_sigma)
        eps = pm.HalfNormal("eps", 5)
        mu = alpha + beta * x_s1
        y_pred = pm.Normal("y_pred", mu=mu, sigma=eps, observed=y_s)
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=seed,
            target_accept=target_accept,
            progressbar=True,
            return_inferencedata=True,
        )
    return model_l, idata


x = x_1
y = y_1
y_s = standardize_y(y)

# linear: folosim doar termenul de grad 1 standardizat
x_s_lin = make_features(x, 1)[0]

model_l, idata_l = fit_linear_model(x_s_lin, y_s, seed=41)

# quadratic (order=2)
x_s2 = make_features(x, 2)
model_q, idata_q = fit_poly_model(x_s2, y_s, order=2, beta_sigma=10, seed=42)

# cubic (order=3)
x_s3 = make_features(x, 3)
model_c, idata_c = fit_poly_model(x_s3, y_s, order=3, beta_sigma=10, seed=43)

# log_likelihood pentru WAIC/LOO
pm.compute_log_likelihood(idata_l, model=model_l)
pm.compute_log_likelihood(idata_q, model=model_q)
pm.compute_log_likelihood(idata_c, model=model_c)

waic_l = az.waic(idata_l, scale="deviance")
waic_q = az.waic(idata_q, scale="deviance")
waic_c = az.waic(idata_c, scale="deviance")

loo_l = az.loo(idata_l, scale="deviance")
loo_q = az.loo(idata_q, scale="deviance")
loo_c = az.loo(idata_c, scale="deviance")

print("WAIC (deviance):")
print("linear   :", waic_l)
print("quadratic:", waic_q)
print("cubic    :", waic_c)

print("\nLOO (deviance):")
print("linear   :", loo_l)
print("quadratic:", loo_q)
print("cubic    :", loo_c)

cmp_waic = az.compare(
    {"linear": idata_l, "quadratic": idata_q, "cubic": idata_c},
    method="BB-pseudo-BMA",
    ic="waic",
    scale="deviance",
)
cmp_loo = az.compare(
    {"linear": idata_l, "quadratic": idata_q, "cubic": idata_c},
    method="BB-pseudo-BMA",
    ic="loo",
    scale="deviance",
)

print("\nCompare WAIC:")
print(cmp_waic)
print("\nCompare LOO:")
print(cmp_loo)

az.plot_compare(cmp_waic)
plt.title("3) Comparatie modele (WAIC)")
plt.show()

az.plot_compare(cmp_loo)
plt.title("3) Comparatie modele (LOO)")
plt.show()

# Plot curbe (posterior mean) pentru linear/quadratic/cubic
alpha_l = idata_l.posterior["alpha"].mean(("chain", "draw")).values
beta_l = idata_l.posterior["beta"].mean(("chain", "draw")).values
y_l_post = alpha_l + beta_l * x_s_lin

alpha_q = idata_q.posterior["alpha"].mean(("chain", "draw")).values
beta_q = idata_q.posterior["beta"].mean(("chain", "draw")).values
y_q_post = alpha_q + np.dot(beta_q, x_s2)

alpha_c = idata_c.posterior["alpha"].mean(("chain", "draw")).values
beta_c = idata_c.posterior["beta"].mean(("chain", "draw")).values
y_c_post = alpha_c + np.dot(beta_c, x_s3)

idx = np.argsort(x_s_lin)
plt.figure(figsize=(10, 4))
plt.scatter(x_s_lin, y_s, marker=".", alpha=0.8, label="data")
plt.plot(x_s_lin[idx], y_l_post[idx], label="linear")
plt.plot(x_s_lin[idx], y_q_post[idx], label="quadratic")
plt.plot(x_s_lin[idx], y_c_post[idx], label="cubic")
plt.xlabel("x (standardized)")
plt.ylabel("y (standardized)")
plt.title("3) Curbe (posterior mean) pentru linear vs quadratic vs cubic")
plt.legend()
plt.show()
