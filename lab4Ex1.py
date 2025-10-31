import numpy as np
from hmmlearn.hmm import MultinomialHMM
import networkx as nx
import matplotlib.pyplot as plt

#a)
model = MultinomialHMM(n_components=3, init_params="")

model.startprob_ = np.array([1/3, 1/3, 1/3])
model.transmat_  = np.array([
    [0.0, 0.5, 0.5],   # D -> (M,E)
    [0.5, 0.25, 0.25], # M -> (D,M,E)
    [0.5, 0.25, 0.25], # E -> (D,M,E)
])
model.emissionprob_ = np.array([
    [0.1, 0.2, 0.4, 0.3],   # D: FB,B,S,NS
    [0.15,0.25,0.5, 0.1],   # M
    [0.2, 0.3, 0.4, 0.1],   # E
])

obs_map = {"FB":0, "B":1, "S":2, "NS":3}
O = np.array([[obs_map[x]] for x in ["FB","FB","S","B","B","S","B","B","NS","B","B"]], dtype=int)

#desen
labels = {0:"D", 1:"M", 2:"E"}
G = nx.DiGraph()
G.add_nodes_from([labels[i] for i in range(3)])
for i in range(3):
    for j in range(3):
        p = model.transmat_[i, j]
        if p > 0:
            G.add_edge(labels[i], labels[j], weight=p)
pos = nx.spring_layout(G, seed=7)
plt.figure(figsize=(5,4))
nx.draw(G, pos, with_labels=True, node_size=1600, node_color="#dce6ff", arrowsize=20, width=2)
edge_labels = {(u,v): f"{d['weight']:.2f}" for u,v,d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
plt.title("HMM State Diagram (D,M,E)")
plt.axis("off")
plt.show()

#b)
logP = model.score(O)
print(f"P(O|λ) = {np.exp(logP):.6e}  (logP = {logP:.6f})")

#c)
logP_vit, states = model.decode(O, algorithm="viterbi")
path_labels = [labels[s] for s in states]
print("Viterbi states:", path_labels)
print(f"P*(O, Q*) = {np.exp(logP_vit):.6e}  (logP* = {logP_vit:.6f})")
