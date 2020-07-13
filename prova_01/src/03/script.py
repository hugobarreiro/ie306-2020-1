import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import sympy
from scipy import linalg, signal


def sanitize_path(path):
    return str.replace(path, '.', '_')


def wiener(a, random_state):

    s = random_state.choice([1., -1.], size=(1000,))
    channel = signal.TransferFunction([1, a], [1, 0], dt=1)

    t, x = signal.dlsim(channel, s)

    R = linalg.toeplitz(np.r_[[1+a**2, a], np.zeros(7)])
    p = np.r_[[1], np.zeros(8)]
    w_wiener = np.linalg.inv(R)@p
    with open(f'prova_01/tex/03/{sanitize_path(f"Wiener_{a}")}.tex', mode='w') as tex_file:
        tex_file.write(sympy.latex(sympy.Matrix(w_wiener).n(3)))

    equalizer = signal.TransferFunction(w_wiener[:2], [1, 0], dt=1)

    t, s_est = signal.dlsim(equalizer, x, t=t)

    HW = signal.TransferFunction(
        np.convolve(channel.num, equalizer.num),
        np.convolve(channel.den, equalizer.den),
        dt=1
    )

    w, mag, phase = signal.dbode(HW)

    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].set_ylabel('Magnitude')
    axes[0].semilogx(w, mag)
    axes[1].set_ylabel('Fase')
    axes[1].set_xlabel('$\omega$')
    axes[1].semilogx(w, phase)
    plt.tight_layout()
    fig.savefig(f'prova_01/img/03/{sanitize_path(f"Wiener_Bode_{a}")}.png')


def robinson(a, random_state):
    s = random_state.choice([1., -1.], size=(1000,))
    channel = signal.TransferFunction([1, a], [1, 0], dt=1)
    t, x = signal.dlsim(channel, s)

    r = sm.tsa.acf(x, fft=True)


if __name__ == '__main__':
    random_state = np.random.default_rng(42)
    wiener(0.625, random_state)
    wiener(1.6, random_state)
