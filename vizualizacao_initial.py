import h5py
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

class langevin:
    def __init__(self, dt: float, tau_resposta: float, g: float,N_pontos: int, vderivada, funcao_resposta):
        self.dt = dt
        self.tau_resposta = tau_resposta
        self.g = g
        self.N_pontos = N_pontos
        self.funcao_resposta = funcao_resposta
        self.vderivada = vderivada
    def V_derivada(self, x: np.ndarray) -> np.ndarray:
        return self.vderivada(x)
    def Funcao_resposta(self, n:int) -> float:
        # n aqui não é o número de pontos e sim em que passo estamos
        return self.funcao_resposta(n)
    def winner_process(self, dt: float, N_pontos: int) -> np.ndarray:
        return np.random.normal(0, np.sqrt(dt), N_pontos)  
    def resolver_ate_n(self, N:int, x0:float, quando_salvar_passo: int, bins: int = 100,range_hist: tuple = None, nome_arquivo: str = "dados.h5"):
        N_p = self.N_pontos
        x_atual = np.full(N_p, x0)
        dt = self.dt
        g = self.g
        N_salvos = (N - 1) // quando_salvar_passo + 1
        with h5py.File(nome_arquivo, "w") as f:
            histogramas = f.create_dataset("histogramas", shape=(N_salvos, bins), dtype=np.float64)
            tempos = f.create_dataset("tempo", shape = (N_salvos, ), dtype = np.float64)
            bins_dataset = f.create_dataset("bins",shape=(bins,),dtype=np.float64)
            indice_histograma = 0
            
            for i in range(N):
                f = self.Funcao_resposta(i)
                dw = self.winner_process(dt=dt, N_pontos=N_p)
                dV = self.V_derivada(x_atual)
                x_novo = x_atual - dV*f*dt + dw*g
                
                if i % quando_salvar_passo == 0:
                    hist, edges = np.histogram(
                        x_novo,
                        bins=bins,
                        range=range_hist,
                        density=True
                        )
                    centros = (edges[:-1] + edges[1:]) / 2
                    histogramas[indice_histograma] = hist
                    tempos[indice_histograma] = i
                    bins_dataset[:] = centros
                    indice_histograma += 1
                x_atual = x_novo
    def resolver_ate_n_estatisticas(self, N:int, x0:float):
        N_p = self.N_pontos
        x_atual = np.full(N_p, x0)
        dt = self.dt
        g = self.g
        times = np.arange(N)*dt
        media = np.zeros(N)
        var = np.zeros(N)


        for i in range(N):
            f = self.Funcao_resposta(i)
            dw = self.winner_process(dt=dt, N_pontos=N_p)
            dV = self.V_derivada(x_atual)
            x_novo = x_atual - dV*f*dt + dw*g
            x_atual = x_novo
            media[i] = np.mean(x_atual)
            var[i] = np.var(x_atual)

        return times, media, var

def resposta_periodica(n):
    dt = 0.001
    T = 2.0

    t = n * dt
    fase = t % T

    if fase < T / 2:
        return 1.0
    else:
        return 0.0

def visualizar(self, nome_arquivo: str, V, F, x_min, x_max, N_forca=10):
    """
    Visualização interativa dos histogramas salvos.

    Parâmetros
    ----------
    nome_arquivo : str
        Arquivo .h5 contendo os histogramas.

    V : função
        Potencial V(x).

    F : função
        Força F(x, N), onde N é o número de termos de Fourier.

    x_min, x_max : float
        Intervalo espacial da visualização.

    N_forca : int
        Número de termos da série de Fourier utilizados na força.
    """

    # ==================================================
    # Carrega os dados
    # ==================================================

    with h5py.File(nome_arquivo, "r") as arquivo:
        P = arquivo["histogramas"][:]
        tempos = arquivo["tempo"][:]
        x_hist = arquivo["bins"][:]

    N_tempos = len(tempos)

    # Domínio para potencial e força
    x = np.linspace(x_min, x_max, 2000)

    potencial = V(x)
    forca = F(x, N_forca)

    # ==================================================
    # Figura
    # ==================================================

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True
    )

    plt.subplots_adjust(
        bottom=0.25,
        hspace=0.35
    )

    # ==================================================
    # Gráfico da distribuição
    # ==================================================

    linha_hist, = ax1.plot(
        x_hist,
        P[0],
        linewidth=2,
        label="P(x,t)"
    )

    ax1.set_ylabel(r"$P(x,t)$")
    ax1.set_title(
        f"Distribuição em t = {tempos[0]:.3f}"
    )

    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(0, np.max(P) * 1.1)

    ax1.grid(alpha=0.3)
    ax1.legend()

    # ==================================================
    # Potencial e força
    # ==================================================

    linha_V, = ax2.plot(
        x,
        potencial,
        linewidth=2,
        label=r"$V(x)$"
    )

    linha_F, = ax2.plot(
        x,
        forca,
        linewidth=2,
        label=r"$F(x)$"
    )

    ax2.set_xlabel("$x$")
    ax2.set_ylabel("V / F")
    ax2.set_xlim(x_min, x_max)

    ax2.grid(alpha=0.3)
    ax2.legend()

    # ==================================================
    # Slider
    # ==================================================

    ax_slider = plt.axes(
        [0.15, 0.10, 0.65, 0.04]
    )

    slider = Slider(
        ax=ax_slider,
        label="Tempo",
        valmin=0,
        valmax=N_tempos - 1,
        valinit=0,
        valstep=1
    )

    # ==================================================
    # Botão liga/desliga
    # ==================================================

    ax_check = plt.axes(
        [0.83, 0.07, 0.12, 0.10]
    )

    check = CheckButtons(
        ax_check,
        ["Potencial"],
        [True]
    )

    # ==================================================
    # Atualização do tempo
    # ==================================================

    def atualizar_tempo(valor):

        indice = int(slider.val)

        linha_hist.set_ydata(P[indice])

        ax1.set_title(
            f"Distribuição em t = {tempos[indice]:.3f}"
        )

        fig.canvas.draw_idle()

    slider.on_changed(atualizar_tempo)

    # ==================================================
    # Liga/desliga potencial
    # ==================================================

    def ligar_desligar(label):

        estado = check.get_status()[0]

        linha_V.set_visible(estado)
        linha_F.set_visible(estado)

        fig.canvas.draw_idle()

    check.on_clicked(ligar_desligar)

    plt.show()

