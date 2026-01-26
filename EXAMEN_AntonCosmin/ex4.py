import arviz as az
import pymc as pm
import matplotlib.pyplot as plt

#4a
cmp = az.compare({"linear": idata_lin, "poly": idata_poly}, ic="waic")
print("\n--- WAIC compare ---")
print(cmp)

#4b
with model_lin:
    pm.sample_posterior_predictive(idata_lin, var_names=["y_obs"], extend_inferencedata=True)

with model_poly:
    pm.sample_posterior_predictive(idata_poly, var_names=["y_obs"], extend_inferencedata=True)

print("\n--- PPC plots ---")
az.plot_ppc(idata_lin, num_pp_samples=100)
plt.title("PPC - linear model")
plt.show()

az.plot_ppc(idata_poly, num_pp_samples=100)
plt.title("PPC - polynomial model")
plt.show()
