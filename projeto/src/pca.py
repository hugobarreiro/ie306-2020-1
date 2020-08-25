# Imports necessários
import numpy as np
from scipy.io.wavfile import read, write

# Carregar misturas
fs1, mic1 = read('projeto/data/mixtures/mic1.wav')
fs2, mic2 = read('projeto/data/mixtures/mic2.wav')

# Garantir frequências idênticas
assert fs1 == fs2
fs = fs1

# Normalizar (z-score) misturas
X = np.c_[mic1, mic2].T
X -= X.mean(axis=1)[:, np.newaxis]
X /= X.std(axis=1)[:, np.newaxis]

# Esferização
u, s, _ = np.linalg.svd(X@X.T)
T = np.diag(s**(-1/2))@u.T
whiten = T@X

# Foi usado um ganho de escala para tornar audível o sinal esferizado
whiten1, whiten2 = np.sqrt(len(X.T))*whiten

write('projeto/data/whiten/whiten1.wav', fs, whiten1)
write('projeto/data/whiten/whiten2.wav', fs, whiten2)
