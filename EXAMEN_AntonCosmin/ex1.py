import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("bike_daily.csv").dropna()

cols = ["temp_c", "humidity", "wind_kph", "is_holiday", "season", "rentals"]
sns.pairplot(df[cols], hue="season", diag_kind="kde")
plt.show()

sns.scatterplot(x="temp_c", y="rentals", data=df)
plt.show()
