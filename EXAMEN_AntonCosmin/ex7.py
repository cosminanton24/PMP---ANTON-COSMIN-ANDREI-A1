summ_log = az.summary(idata_log, var_names=["alpha","beta"], hdi_prob=0.95)
print(summ_log)

beta_mean = idata_log.posterior["beta"].mean(("chain","draw")).values

feat_names = ["temp_c", "humidity", "wind_kph", "temp_c2"] + list(df_oh.drop(columns=["rentals","temp_c","humidity","wind_kph"]).columns)

best = feat_names[int(np.argmax(np.abs(beta_mean)))]
print("Most influential:", best)
