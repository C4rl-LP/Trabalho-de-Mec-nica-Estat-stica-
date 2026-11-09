import numpy as np
import matplotlib.pyplot as plt
from utils_coisa.solver import langevin, resposta_periodica

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

    import h5py
    from matplotlib.widgets import Slider, CheckButtons

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

