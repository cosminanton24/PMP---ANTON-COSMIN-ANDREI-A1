
import numpy as np
import pymc as pm
import arviz as az

y = np.array([56, 60, 58, 55, 57, 59, 61, 56, 58, 60])

#a)
x = y.mean()  # prior center for mu

with pm.Model() as model_weak_prior:
    # prior for mu
    mu = pm.Normal("mu", mu=x, sigma=10)

    # prior for sigma
    sigma = pm.HalfNormal("sigma", sigma=10)

    X = pm.Normal("X", mu=mu, sigma=sigma, observed=y)

    
    idata_weak = pm.sample(
        2000,
        tune=1000,
        random_seed=42,
        target_accept=0.9,
        cores=1,
        chains=2,
    )

#b)
summary_weak = az.summary(
    idata_weak,
    var_names=["mu", "sigma"],
    hdi_prob=0.95,
    kind="stats",
)
print("\n(b) Posterior summary (weak prior):")
print(summary_weak)

hdi_weak = az.hdi(idata_weak, var_names=["mu", "sigma"], hdi_prob=0.95)
print("\n95% HDI for mu and sigma (weak prior):")
print(hdi_weak)

#c)


mean_freq = y.mean()
std_freq = y.std(ddof=1)


mu_post_mean = idata_weak.posterior["mu"].mean().item()
sigma_post_mean = idata_weak.posterior["sigma"].mean().item()

print("\n(c) Frequentist vs Bayesian (weak prior)")
print(f"Frequentist mean       : {mean_freq:.3f}")
print(f"Bayesian posterior mean: {mu_post_mean:.3f}")
print(f"Frequentist std (ddof=1)       : {std_freq:.3f}")
print(f"Bayesian posterior mean sigma  : {sigma_post_mean:.3f}")

#d)
with pm.Model() as model_strong_prior:
    mu_strong = pm.Normal("mu", mu=50.0, sigma=1.0)
    sigma_strong = pm.HalfNormal("sigma", sigma=10)
    X_strong = pm.Normal("X", mu=mu_strong, sigma=sigma_strong, observed=y)

    idata_strong = pm.sample(
        2000,
        tune=1000,
        random_seed=42,
        target_accept=0.9,
        cores=1,
        chains=2,
    )

summary_strong = az.summary(
    idata_strong,
    var_names=["mu", "sigma"],
    hdi_prob=0.95,
    kind="stats",
)
print("\n(d) Posterior summary (strong prior):")
print(summary_strong)

mu_strong_mean = idata_strong.posterior["mu"].mean().item()
sigma_strong_mean = idata_strong.posterior["sigma"].mean().item()

print("\nPosterior means with strong prior:")
print(f"mu (strong prior)    : {mu_strong_mean:.3f}")
print(f"sigma (strong prior) : {sigma_strong_mean:.3f}")

