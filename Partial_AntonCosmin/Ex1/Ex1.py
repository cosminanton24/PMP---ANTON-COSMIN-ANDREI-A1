from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


#a)
model = BayesianNetwork([
    ('O', 'H'),   
    ('O', 'W'),  
    ('H', 'R'),  
    ('W', 'R'),   
    ('H', 'E'),   
    ('R', 'C')    
])

# P(O)
cpd_O = TabularCPD(
    variable='O', variable_card=2,
    values=[[0.3], [0.7]],
    state_names={'O': ['cold', 'mild']}
)

# P(H|O)
cpd_H = TabularCPD(
    variable='H', variable_card=2,
    values=[[0.9, 0.2],      # H=yes
            [0.1, 0.8]],     # =no
    evidence=['O'], evidence_card=[2],
    state_names={'H': ['yes', 'no'], 'O': ['cold', 'mild']}
)

# P(W|O)
cpd_W = TabularCPD(
    variable='W', variable_card=2,
    values=[[0.1, 0.6],      # W=yes
            [0.9, 0.4]],     # W=no
    evidence=['O'], evidence_card=[2],
    state_names={'W': ['yes', 'no'], 'O': ['cold', 'mild']}
)

# P(R|H,W)
#(H=yes,W=yes), (yes,no), (no,yes), (no,no)
cpd_R = TabularCPD(
    variable='R', variable_card=2,
    values=[[0.6, 0.9, 0.3, 0.5],   #R=warm
            [0.4, 0.1, 0.7, 0.5]],  #R=cool
    evidence=['H', 'W'], evidence_card=[2, 2],
    state_names={'R': ['warm', 'cool'],
                 'H': ['yes', 'no'],
                 'W': ['yes', 'no']}
)

# P(E|H)
cpd_E = TabularCPD(
    variable='E', variable_card=2,
    values=[[0.8, 0.2],      # E=high
            [0.2, 0.8]],     # E=low
    evidence=['H'], evidence_card=[2],
    state_names={'E': ['high', 'low'], 'H': ['yes', 'no']}
)

# P(C|R)
cpd_C = TabularCPD(
    variable='C', variable_card=2,
    values=[[0.85, 0.40],    # C=comf
            [0.15, 0.60]],   # C = uncomf
    evidence=['R'], evidence_card=[2],
    state_names={'C': ['comfortable', 'uncomfortable'],
                 'R': ['warm', 'cool']}
)

model.add_cpds(cpd_O, cpd_H, cpd_W, cpd_R, cpd_E, cpd_C)
assert model.check_model()

infer = VariableElimination(model)

#b)
# P(H=yes|C=comf)
phi_H = infer.query(variables=['H'], evidence={'C': 'comfortable'})
print(phi_H)

# P(E=high|C=comf)
phi_E = infer.query(variables=['E'], evidence={'C': 'comfortable'})
print(phi_E)

# MAP(H, W),C=comf
map_HW = infer.map_query(variables=['H', 'W'], evidence={'C': 'comfortable'})
print(map_HW)
