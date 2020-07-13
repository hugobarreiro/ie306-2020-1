import matplotlib.pyplot as plt
import numpy as np
import sympy
from scipy import signal
from statsmodels.api import tsa

z = sympy.symbols('z', complex=True)
omega = sympy.symbols('ω')


def simulate(b, a, xlims, random_gen):
    noise_size = 1000
    noise = random_gen.normal(loc=0, scale=1, size=(noise_size,))
    arma_signal = signal.lfilter(b, a, noise)
    acf = tsa.acf(arma_signal, nlags=500, fft=False)

    psd = (np.abs(np.fft.fft(acf))**2)
    freqs = 2*np.pi*np.fft.fftfreq(501)

    fig, ax = plt.subplots()
    ax.plot(np.fft.fftshift(freqs), np.fft.fftshift(psd))
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim(xlims)
    ax.set_xlabel(r'$\omega$', position=(1, 0))
    ax.set_ylabel(r'$\phi(\omega)$', position=(0, 1))
    plt.tight_layout()
    return fig


def ar(random_gen):

    A = 1 + 0.8987*z**-1 + 0.9018*z**-2

    phi = 1 / (A * A.conjugate().subs({z: 1 / z.conjugate()}))

    with open('prova_01/tex/02/PSD_AR.tex', mode='w') as tex_file:
        tex_file.write(sympy.latex(phi.n(3)))

    plot = sympy.plot(
        phi.subs({z: sympy.E ** (sympy.I * omega)}),
        (omega, -3, 3),
        ylabel=r'$\phi(\omega)$',
        depth=10,
        show=False
    )
    plot.save('prova_01/img/02/PSD_AR.png')

    fig = simulate([1.], [1., 0.8987, 0.9018], (-3, 3), random_gen)
    fig.savefig('prova_01/img/02/PSD_AR_simulated.png')


def arma(random_gen):
    z = sympy.symbols('z', complex=True)
    omega = sympy.symbols('ω')

    A = 1 - 1.9368*z**-1 + 0.9519*z**-2
    B = 1 - 1.8894*z**-1 + 1*z**-2

    phi = (B * B.conjugate().subs({z: 1 / z.conjugate()})) / (A * A.conjugate().subs({z: 1 / z.conjugate()}))

    with open('prova_01/tex/02/PSD_ARMA.tex', mode='w') as tex_file:
        tex_file.write(sympy.latex(phi.n(3)))

    plot = sympy.plot(
        phi.subs({z: sympy.E ** (sympy.I * omega)}),
        (omega, -0.5, 0.5),
        ylabel=r'$\phi(\omega)$',
        depth=10,
        show=False,
    )
    plot.save('prova_01/img/02/PSD_ARMA.png')

    fig = simulate([1., -1.8894, 1.], [1., -1.9368, 0.9519], (-0.5, 0.5), random_gen)
    fig.savefig('prova_01/img/02/PSD_ARMA_simulated.png')


if __name__ == '__main__':

    random_gen = np.random.default_rng(42)
    ar(random_gen)
    arma(random_gen)
