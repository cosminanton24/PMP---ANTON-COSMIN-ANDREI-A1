import numpy as np
from hmmlearn import hmm


#a)
start_prob = np.array([0.4, 0.3, 0.3])

trans_mat = np.array([
    [0.6, 0.3, 0.1],   # W
    [0.2, 0.7, 0.1],   #  R
    [0.3, 0.2, 0.5]    # S
])

emission_mat = np.array([
    [0.1, 0.7, 0.2],   # W - l,m,h
    [0.05, 0.25, 0.7], # R - l,m,h
    [0.8, 0.15, 0.05]  # S- l,m,h
])

model = hmm.MultinomialHMM(n_components=3)
model.startprob_ = start_prob
model.transmat_ = trans_mat
model.emissionprob_ = emission_mat


#d)

count = 0
obs = np.array([[1,2,0]]) # M,H,L

for _ in range(10000):
    X, Z = model.sample(3)
    if (X.flatten() == obs).all():
        count += 1

empirical = count / 10000
print("Empirical probability:", empirical)
