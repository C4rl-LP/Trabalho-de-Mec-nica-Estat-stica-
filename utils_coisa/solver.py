"""Solvers for the project's Langevin dynamics."""

import numpy as np


class langevin:
    """Euler–Maruyama solver for an overdamped Langevin equation.

    ``vderivada`` is the derivative of the potential, ``dV/dx``. The position
    update is ``x <- x - dV/dx * f(t) * dt + g * dW`` with
    ``dW ~ Normal(0, dt)``. ``funcao_resposta`` receives the integer step.
    """

    def __init__(
        self,
        dt: float,
        tau_resposta: float,
        g: float,
        N_pontos: int,
        vderivada,
        funcao_resposta,
        rng=None,
    ):
        self.dt = dt
        self.tau_resposta = tau_resposta
        self.g = g
        self.N_pontos = N_pontos
        self.funcao_resposta = funcao_resposta
        self.vderivada = vderivada
        self.rng = rng

    def V_derivada(self, x: np.ndarray) -> np.ndarray:
        return self.vderivada(x)

    def Funcao_resposta(self, n: int) -> float:
        return self.funcao_resposta(n)

    def winner_process(self, dt: float, N_pontos: int) -> np.ndarray:
        if self.rng is None:
            return np.random.normal(0, np.sqrt(dt), N_pontos)
        return self.rng.normal(0, np.sqrt(dt), N_pontos)

    def resolver_ate_n_estatisticas(self, N: int, x0: float):
        """Return post-step times, ensemble mean, and ensemble variance."""
        x_atual = np.full(self.N_pontos, x0, dtype=float)
        tempos = (np.arange(N) + 1) * self.dt
        media = np.empty(N, dtype=float)
        variancia = np.empty(N, dtype=float)

        for i in range(N):
            resposta = self.Funcao_resposta(i)
            dW = self.winner_process(self.dt, self.N_pontos)
            x_atual = (
                x_atual
                - self.V_derivada(x_atual) * resposta * self.dt
                + self.g * dW
            )
            media[i] = np.mean(x_atual)
            variancia[i] = np.var(x_atual)

        return tempos, media, variancia

    def resolver_ate_n(
        self,
        N: int,
        x0: float,
        quando_salvar_passo: int,
        bins: int = 100,
        range_hist: tuple = None,
        nome_arquivo: str = "dados.h5",
    ):
        """Save sampled ensemble histograms to an HDF5 file."""
        import h5py

        x_atual = np.full(self.N_pontos, x0, dtype=float)
        n_salvos = (N - 1) // quando_salvar_passo + 1
        with h5py.File(nome_arquivo, "w") as arquivo:
            histogramas = arquivo.create_dataset(
                "histogramas", shape=(n_salvos, bins), dtype=np.float64
            )
            tempos = arquivo.create_dataset("tempo", shape=(n_salvos,), dtype=np.float64)
            bins_dataset = arquivo.create_dataset("bins", shape=(bins,), dtype=np.float64)
            indice_histograma = 0

            for i in range(N):
                resposta = self.Funcao_resposta(i)
                dW = self.winner_process(self.dt, self.N_pontos)
                x_atual = (
                    x_atual
                    - self.V_derivada(x_atual) * resposta * self.dt
                    + self.g * dW
                )

                if i % quando_salvar_passo == 0:
                    hist, edges = np.histogram(
                        x_atual,
                        bins=bins,
                        range=range_hist,
                        density=True,
                    )
                    histogramas[indice_histograma] = hist
                    tempos[indice_histograma] = (i + 1) * self.dt
                    bins_dataset[:] = (edges[:-1] + edges[1:]) / 2
                    indice_histograma += 1


def resposta_periodica(n: int) -> float:
    """Default square response, retained for compatibility with old notebooks."""
    dt = 0.001
    periodo = 2.0
    fase = (n * dt) % periodo
    return 1.0 if fase < periodo / 2 else 0.0
