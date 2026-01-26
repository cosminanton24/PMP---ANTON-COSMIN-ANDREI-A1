Q = np.percentile(y, 75)
y_bin = (y >= Q).astype(int)
