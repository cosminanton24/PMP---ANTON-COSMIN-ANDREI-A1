az.summary(idata_lin, var_names=["alpha", "beta", "sigma"], hdi_prob=0.95)
az.summary(idata_poly, var_names=["alpha", "beta", "sigma"], hdi_prob=0.95)

az.plot_trace(idata_lin, var_names=["alpha","beta","sigma"])
plt.show()

az.plot_trace(idata_poly, var_names=["alpha","beta","sigma"])
plt.show()
