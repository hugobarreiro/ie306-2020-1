# Imports necessários
import numpy as np
from scipy.io import loadmat
from scipy.io.wavfile import write

# Carregar dados originais
male = loadmat('projeto/data/raw/Male_speech.mat')
female = loadmat('projeto/data/raw/Female_speech.mat')

# Garantir que as frequências são iguais
assert male['Fs'] == female['Fs']
fs = male['Fs'].item()

male = male['y'].ravel()
female = female['y'].ravel()

S = np.array(list(zip(female, male))).T

# Misturar fontes
A_mix = np.array([[1, 1.3], [0.6, 0.45]])
mic1, mic2 = A_mix.T@S

# Salvar arquivos de saída
write('projeto/data/sources/male.wav', fs, male)
write('projeto/data/sources/female.wav', fs, female)
write('projeto/data/mixtures/mic1.wav', fs, mic1)
write('projeto/data/mixtures/mic2.wav', fs, mic2)
