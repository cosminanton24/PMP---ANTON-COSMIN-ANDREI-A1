import pandas as pd
import numpy as np
import pymc as pm
import arviz as az

#2a
df = pd.read_csv("bike_daily.csv").dropna()
df_oh = pd.get_dummies(df, columns=["season"], drop_first=True)
y = df_oh["rentals"].values
X_cont = df_oh[["temp_c", "humidity", "wind_kph"]].values
X_cat = df_oh.drop(columns=["rentals", "temp_c", "humidity", "wind_kph"]).values

Xc_mean = X_cont.mean(axis=0)
Xc_std  = X_cont.std(axis=0, ddof=0)
Xc_s = (X_cont - Xc_mean) / Xc_std

y_mean = y.mean()
y_std  = y.std(ddof=0)
y_s = (y - y_mean) / y_std

X = np.hstack([Xc_s, X_cat])

#2b
with pm.Model() as model_lin:
    alpha = pm.Normal("alpha", 0, 10)
    beta  = pm.Normal("beta", 0, 5, shape=X.shape[1])
    sigma = pm.HalfNormal("sigma", 5)

    mu = alpha + pm.math.dot(X, beta)

    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_s)

    idata_lin = pm.sample(
        2000, tune=2000, chains=4,
        target_accept=0.9, return_inferencedata=True
    )

print("\n--- Linear model summary (95% HDI) ---")
print(az.summary(idata_lin, var_names=["alpha", "beta", "sigma"], hdi_prob=0.95))

#2c
temp2 = (Xc_s[:, 0] ** 2)
X_poly = np.hstack([Xc_s, temp2[:, None], X_cat])

with pm.Model() as model_poly:
    alpha = pm.Normal("alpha", 0, 10)
    beta  = pm.Normal("beta", 0, 5, shape=X_poly.shape[1])
    sigma = pm.HalfNormal("sigma", 5)

    mu = alpha + pm.math.dot(X_poly, beta)

    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_s)

    idata_poly = pm.sample(
        2000, tune=2000, chains=4,
        target_accept=0.9, return_inferencedata=True
    )

print("\n--- Polynomial model summary (95% HDI) ---")
print(az.summary(idata_poly, var_names=["alpha", "beta", "sigma"], hdi_prob=0.95))
