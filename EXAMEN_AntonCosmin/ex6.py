with pm.Model() as model_log:
    alpha = pm.Normal("alpha", 0, 10)
    beta  = pm.Normal("beta", 0, 5, shape=X_poly.shape[1])

    logit = alpha + pm.math.dot(X_poly, beta)
    p = pm.Deterministic("p", pm.math.sigmoid(logit))

    y_obs = pm.Bernoulli("y_obs", p=p, observed=y_bin)

    idata_log = pm.sample(2000, tune=2000, chains=4, target_accept=0.9, return_inferencedata=True)
