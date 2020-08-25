# Imports necessários
import numpy as np
from scipy.io.wavfile import read, write

# Carregar sinais esferizados
fs1, whiten1 = read('projeto/data/whiten/whiten1.wav')
fs2, whiten2 = read('projeto/data/whiten/whiten2.wav')

# Garantir frequências idênticas
assert fs1 == fs2
fs = fs1

# Garantir normalização (z-score) dos esferizados
sphere = np.c_[whiten1, whiten2].T
sphere -= sphere.mean(axis=1)[:, np.newaxis]
sphere /= sphere.std(axis=1)[:, np.newaxis]


random_state = np.random.default_rng(2001)

# Escolha de uma função para FastICA
g = np.tanh
h = lambda x: 1 - (np.tanh(x))**2


W = []
# Para cada fonte:
for _ in range(0, 2):
    # Inicializa um vetor de norma 1
    w = random_state.random(len(sphere))
    w /= np.linalg.norm(w)

    # Enquanto não converge
    error = 1
    while error > 1e-8:
        # Encontra novo vetor
        r = (sphere*g(w@sphere)).mean(axis=1) - h(w@sphere).mean()*w

        # Ortogonaliza
        if W:
            r -= r@W[0]*W[0]
        r /= np.linalg.norm(r)

        # Avalia o erro - Quão longe de serem os mesmos
        error = abs(abs(r@w) - 1)

        # Atualiza o vetor
        w = r
    W.append(w)
W = np.r_[W]

# Sinal estimado
estimation1, estimation2 = W.T@sphere

# Salva sinais estimados
write('projeto/data/estimates/estimate1.wav', fs, estimation1)
write('projeto/data/estimates/estimate2.wav', fs, estimation2)
