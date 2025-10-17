import numpy as np
import matplotlib.pyplot as plt

def pois_samples(lam, n, rng):
    return rng.poisson(lam, size=n)#generator pt esantioane

def mixture_poisson_samples(lambdas, probs, n, rng):#aleg random un lambda
    chosen_lams = rng.choice(lambdas, size=n, replace=True, p=probs)
    return np.array([rng.poisson(l) for l in chosen_lams])#esantion

def describe(x):
    mu = x.mean()#media
    var = x.var(ddof=1)#variatia 
    return mu, var, var/mu if mu > 0 else np.nan

def plot_hist(x, title):
    #contructie grafic pt fiecare test
    plt.figure()
    
    bmax = int(x.max()) + 2
    plt.hist(x, bins=range(0, max(bmax, 12)), density=True, edgecolor="black")
    plt.xlabel("k")
    plt.ylabel("")
    plt.title(title)
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    rng = np.random.default_rng()
    n = 1000
    lambdas = [1, 2, 5, 10]

    #1) 4 seturi Poisson cu lambda fix
    datasets = {}
    for lam in lambdas:
        s = pois_samples(lam, n, rng)
        datasets[f"Pois({lam})"] = s

    #2) mix lambda{1,2,5,10}
    mix_uniform = mixture_poisson_samples(lambdas, probs=[0.25, 0.25, 0.25, 0.25], n=n, rng=rng)
    datasets["Mixture (uniform lambda={1,2,5,10})"] = mix_uniform

    #lambda = 5 mai probabil
    mix_biased = mixture_poisson_samples(lambdas, probs=[0.1, 0.1, 0.6, 0.2], n=n, rng=rng)
    datasets["Mixture (0.1,0.1,0.6,0.2)"] = mix_biased

    #afis histograme
    print("Dataset                            Mean      Var        Var/Mean")
    print("-"*65)
    for name, arr in datasets.items():
        mu, var, vm = describe(arr)
        print(f"{name:32s}  {mu:8.3f}  {var:10.3f}   {vm:9.3f}")
        plot_hist(arr, f"Histograma – {name}")
