from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = DiscreteBayesianNetwork([('S','O'), ('S','L'), ('S','M'), ('L','M')])

cpd_S = TabularCPD('S', 2, [[0.6], [0.4]])
cpd_O = TabularCPD('O', 2, [[0.9, 0.3], [0.1, 0.7]], evidence=['S'], evidence_card=[2])
cpd_L = TabularCPD('L', 2, [[0.7, 0.2], [0.3, 0.8]], evidence=['S'], evidence_card=[2])
cpd_M = TabularCPD('M', 2, [[0.8, 0.4, 0.5, 0.1], [0.2, 0.6, 0.5, 0.9]],evidence=['S','L'], evidence_card=[2,2])

model.add_cpds(cpd_S, cpd_O, cpd_L, cpd_M)
assert model.check_model()

# a) Independente 
inds = model.local_independencies(['S','O','L','M']).get_assertions()
print("Independente:")
for ind in inds:
    print(ind)

# b) P(S=1 | O,L,M) pentru toate combinatiile
infer = VariableElimination(model)  
print("\n P(S=1 | O,L,M):")
for O in (0,1):
    for L in (0,1):
        for M in (0,1):
            post = infer.query(variables=['S'], evidence={'O': O, 'L': L, 'M': M})
            p_s1 = float(post.values[1]) 
            print(f"O={O}, L={L}, M={M} -> {p_s1:.4f}")

