
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

#read data
df = pd.read_csv("date_promovare_examen.csv")

#check if data is balanced (number of 0 vs 1)
counts = df["Promovare"].value_counts().sort_index()
props = counts / len(df)
print("Class counts (0/1):")
print(counts)
print("\nClass proportions (0/1):")
print(props)

#prepare X (predictors) and y (target)
X = df[["Ore_Studiu", "Ore_Somn"]].to_numpy()
y = df["Promovare"].to_numpy().astype(int)

#standardize predictors (helps sampling and interpretation of coefficients)
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
Xz = (X - X_mean) / X_std

#model pymc
#    p = sigmoid(alpha + beta1*Ore_Studiu_z + beta2*Ore_Somn_z)
with pm.Model() as logistic_model:
    # priors pentru intercept si coeficienti
    alpha = pm.Normal("alpha", mu=0.0, sigma=5.0)
    beta = pm.Normal("beta", mu=0.0, sigma=5.0, shape=2)

    #prob for promo
    p = pm.math.sigmoid(alpha + pm.math.dot(Xz, beta))

    # binary observations (0/1)
    y_obs = pm.Bernoulli("y_obs", p=p, observed=y)

    # sampling MCMC
    idata = pm.sample(
        draws=2000,
        tune=1000,
        chains=2,
        cores=1,
        random_seed=42,
        target_accept=0.9
    )

# posterior analysis
print("\nPosterior summary (alpha, beta):")
print(az.summary(idata, var_names=["alpha", "beta"], hdi_prob=0.95))
