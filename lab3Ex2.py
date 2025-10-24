import numpy as np

g = np.array([
    [+1, +1, +1, -1, -1],
    [+1, +1, -1, -1, -1],
    [+1, -1, -1, -1, +1],
    [+1, -1, -1, +1, +1],
    [-1, -1, +1, +1, +1],
], dtype=int)

lam = 1.0                     
H, W = g.shape

f = g.copy()

nb_offsets = [(1,0), (-1,0), (0,1), (0,-1)]

def local_energy(label, i, j, f_curr):
    e_nb = 0.0
    for di, dj in nb_offsets:
        ni, nj = i+di, j+dj
        if 0 <= ni < H and 0 <= nj < W:
            e_nb += (label - f_curr[ni, nj])**2
    e_data = lam * (label - g[i, j])**2
    return e_nb + e_data

max_iter = 20
for _ in range(max_iter):
    changes = 0
    for i in range(H):
        for j in range(W):
            e_minus = local_energy(-1, i, j, f)
            e_plus  = local_energy(+1, i, j, f)
            new_label = -1 if e_minus < e_plus else +1
            if new_label != f[i, j]:
                f[i, j] = new_label
                changes += 1
    if changes == 0:
        break

def total_energy(fimg):
    e_nb = 0.0
    for i in range(H):
        for j in range(W):
            if i+1 < H: e_nb += (fimg[i,j]-fimg[i+1,j])**2
            if j+1 < W: e_nb += (fimg[i,j]-fimg[i,j+1])**2
    e_data = lam * np.sum((fimg - g)**2)
    return e_nb + e_data

U = total_energy(f)

print("f (etichetarea finala, in {-1,+1}):")
print(f)
print("Energie U(f) =", float(U))
