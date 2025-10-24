from pgmpy.models import MarkovNetwork
from pgmpy.factors.discrete import DiscreteFactor
import matplotlib.pyplot as plt
import networkx as nx
from itertools import product
import math

vars_ = ["A1","A2","A3","A4","A5"]
idx = {v: i+1 for i,v in enumerate(vars_)}
edges = [("A1","A2"), ("A1","A3"), ("A2","A4"), ("A2","A5"), ("A3","A4"), ("A4","A5")]

model = MarkovNetwork()
model.add_nodes_from(vars_)
model.add_edges_from(edges)

plt.figure(figsize=(6,4))
G = nx.Graph(); G.add_nodes_from(model.nodes()); G.add_edges_from(model.edges())
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color="#d0e1ff", node_size=1200, font_weight='bold', width=2)
plt.title("MRF – Ex.1"); plt.axis("off"); plt.show()

cliques = [("A1","A2"), ("A1","A3"), ("A3","A4"), ("A2","A4","A5")]
print("Cliques (maximale):", cliques)

def state_to_val(state): return -1 if state == 0 else 1

factors = []
for C in cliques:
    card = [2]*len(C)
    table = []
    for assignment in product([0,1], repeat=len(C)):
        s = sum(idx[v] * state_to_val(st) for v, st in zip(C, assignment))
        table.append(math.exp(s))
    factors.append(DiscreteFactor(list(C), card, table))

model.add_factors(*factors)

Z = 0.0
best_prob = -1.0
best_assign = None

for a1,a2,a3,a4,a5 in product([0,1], repeat=5):
    prod_val = 1.0
    assign = {"A1":a1,"A2":a2,"A3":a3,"A4":a4,"A5":a5}
    for fac in factors:
        kw = {v: assign[v] for v in fac.scope()}
        prod_val *= float(fac.get_value(**kw))
    Z += prod_val
    if prod_val > best_prob:
        best_prob = prod_val
        best_assign = (a1,a2,a3,a4,a5)

map_prob = best_prob / Z
map_config_vals = tuple(state_to_val(st) for st in best_assign)

print("Z (constanta de normalizare):", Z)
print("MAP assignment (in valori -1/+1):", map_config_vals)
print("P_MAP (probabilitate normalizata):", map_prob)
