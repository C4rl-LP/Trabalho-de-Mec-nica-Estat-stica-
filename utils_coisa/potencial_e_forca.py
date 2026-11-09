import numpy as np 


def alpha_gaussiana(n: int, sigma: float, L: float):
    if sigma < 0:
        raise ValueError("'sigma' deve ser não negativo.")
    if L <= 0:
        raise ValueError("'L' deve ser positivo.")
    return np.exp(-0.5 * (2 * np.pi * n * sigma / L) ** 2)


class Fourie_atenuacao_da_serra:
    def __init__(self, alpha: callable, a: float, L: float, V_0):
        self.alpha = alpha # funcao de atenuacao
        self.a = a # tamanho da rampa em que ela é descrescente 
        self.b = L - a # tamanho da rampa em que ela é crescente 
        self.L = L # périodo do potencial 
        self.V_0 = V_0


    def alpha_n_atenuacao(self,n:int):
        """Fator de atenuação aplicado ao harmônico de índice ``n``."""
        return self.alpha(n)

    def sigma_n(self, n: int):
        """Alias para o fator de atenuação do harmônico ``n``."""
        return self.alpha_n_atenuacao(n)
    
    def V_original(self, x):
        x_mod = np.mod(x, self.L) 
        return np.where(
            x_mod < self.b,
            self.V_0 * x_mod / self.b,
            -self.V_0 / self.a * (x_mod - self.b) + self.V_0,
        )

    def forca_orginial(self, x):
        """Força ``-dV/dx`` do potencial em rampa, por partes."""
        x_mod = np.mod(x, self.L)
        return np.where(x_mod < self.b, -self.V_0 / self.b, self.V_0 / self.a)



    def an(self,n: int, atenuado: bool = True):
        if n == 0:
            return self.V_0 / 2
        a = self.a
        b = self.b
        L = self.L
        coeficiente = (
            -self.V_0 * L**2
            / (np.pi**2 * a * b * n**2)
            * np.sin(np.pi * b * n / L)**2
        )
        if atenuado:
            coeficiente *= self.alpha_n_atenuacao(n)
        return coeficiente


    def bn(self,n: int, atenuado: bool = True):
        if n == 0:
            return 0.0
        a = self.a
        b= self.b
        L = self.L
        V0 = self.V_0
        coeficiente = (
            V0 * L**2
            / (2 * np.pi**2 * a * b * n**2)
            * np.sin(2 * np.pi * b * n / L)
        )
        if atenuado:
            coeficiente *= self.alpha_n_atenuacao(n)
        return coeficiente

    def V_fourier_calc(
        self,
        x: float,
        parada: str = "tol_erro",
        N_stop: int = 5,
        tol_erro: float = 0.1,
        atenuado: bool = True
    ):
        """Avalia a série até ``N_stop`` ou até o próximo harmônico ser pequeno.

        Em ``tol_erro``, ``tol_erro`` é um limiar para o módulo de cada
        coeficiente atenuado do próximo harmônico; não é uma estimativa rigorosa
        do erro total da série.
        """
        if parada not in {"tol_erro", "N_stop"}:
            raise ValueError(
                "'parada' deve ser 'tol_erro' ou 'N_stop'. "
                f"Recebido: {parada!r}."
            )
        if not isinstance(N_stop, (int, np.integer)) or N_stop < 0:
            raise ValueError("'N_stop' deve ser um inteiro não negativo.")
        if tol_erro <= 0:
            raise ValueError("'tol_erro' deve ser positivo.")

        # A busca do índice de corte é sequencial; a soma sobre x e n é NumPy.
        if parada == "N_stop":
            ultimo_n = N_stop
        else:
            ultimo_n = 0
            n = 1
            while max(abs(self.an(n,atenuado=atenuado)), abs(self.bn(n, atenuado=atenuado))) > tol_erro:
                ultimo_n = n
                n += 1

        x_array = np.asarray(x)
        harmonicos = np.arange(1, ultimo_n + 1)
        if ultimo_n == 0:
            return np.full_like(x_array, self.an(0), dtype=float).item() if x_array.ndim == 0 else np.full_like(x_array, self.an(0), dtype=float)

        a_n = np.array([self.an(int(n), atenuado=atenuado) for n in harmonicos])
        b_n = np.array([self.bn(int(n), atenuado=atenuado) for n in harmonicos])
        forma_n = (ultimo_n,) + (1,) * x_array.ndim
        fase = 2 * np.pi * harmonicos.reshape(forma_n) * x_array / self.L
        termos = a_n.reshape(forma_n) * np.cos(fase) + b_n.reshape(forma_n) * np.sin(fase)
        return self.an(0) + np.sum(termos, axis=0)

    def forca_fourrier_calc(
        self,
        x: float,
        parada: str = "tol_erro",
        N_stop: int = 5,
        tol_erro: float = 0.1,
        atenuado: bool = True,
    ):
        """Avalia ``-dV/dx`` pela derivada da série de Fourier truncada.

        No modo ``tol_erro``, o limiar se aplica aos coeficientes de força
        ``(2*pi*n/L)*an`` e ``(2*pi*n/L)*bn`` do próximo harmônico.
        """
        if parada not in {"tol_erro", "N_stop"}:
            raise ValueError(
                "'parada' deve ser 'tol_erro' ou 'N_stop'. "
                f"Recebido: {parada!r}."
            )
        if not isinstance(N_stop, (int, np.integer)) or N_stop < 0:
            raise ValueError("'N_stop' deve ser um inteiro não negativo.")
        if tol_erro <= 0:
            raise ValueError("'tol_erro' deve ser positivo.")

        def coeficientes_forca(n):
            fator = 2 * np.pi * n / self.L
            return (
                fator * self.an(n, atenuado=atenuado),
                fator * self.bn(n, atenuado=atenuado),
            )

        if parada == "N_stop":
            ultimo_n = N_stop
        else:
            ultimo_n = 0
            n = 1
            while max(abs(c) for c in coeficientes_forca(n)) > tol_erro:
                ultimo_n = n
                n += 1

        x_array = np.asarray(x)
        if ultimo_n == 0:
            return np.zeros_like(x_array, dtype=float).item() if x_array.ndim == 0 else np.zeros_like(x_array, dtype=float)

        harmonicos = np.arange(1, ultimo_n + 1)
        coef_sin = np.array([coeficientes_forca(int(n))[0] for n in harmonicos])
        coef_cos = np.array([coeficientes_forca(int(n))[1] for n in harmonicos])
        forma_n = (ultimo_n,) + (1,) * x_array.ndim
        fase = 2 * np.pi * harmonicos.reshape(forma_n) * x_array / self.L
        termos = coef_sin.reshape(forma_n) * np.sin(fase) - coef_cos.reshape(forma_n) * np.cos(fase)
        return np.sum(termos, axis=0)

    def preparar_forca(self, parada: str = "N_stop",
        N_stop: int = 5,
        tol_erro: float = 0.1,
        atenuado: bool = True,
    ):
        if parada not in {"tol_erro", "N_stop"}:
            raise ValueError(
                "'parada' deve ser 'tol_erro' ou 'N_stop'. "
                f"Recebido: {parada!r}."
            )
        if not isinstance(N_stop, (int, np.integer)) or N_stop < 0:
            raise ValueError("'N_stop' deve ser um inteiro não negativo.")
        if tol_erro <= 0:
            raise ValueError("'tol_erro' deve ser positivo.")

        def coeficientes_forca(n):
            fator = 2 * np.pi * n / self.L
            return (
                fator * self.an(n, atenuado=atenuado),
                fator * self.bn(n, atenuado=atenuado),
            )

        if parada == "N_stop":
            ultimo_n = N_stop
        else:
            ultimo_n = 0
            n = 1
            while max(abs(c) for c in coeficientes_forca(n)) > tol_erro:
                ultimo_n = n
                n += 1
        harmonicos = np.arange(1, ultimo_n+1)
        forca_an = np.array([coeficientes_forca(int(n))[0] for n in harmonicos])
        forca_bn = np.array([coeficientes_forca(int(n))[1] for n in harmonicos])
        return harmonicos, forca_an, forca_bn
    def forca_funcao(self, x, harmonicos, forca_an, forca_bn):
        x_array = np.asarray(x)

        ultimo_n = len(harmonicos)
        if ultimo_n == 0:
            return np.zeros_like(x_array, dtype=float).item() if x_array.ndim == 0 else np.zeros_like(x_array, dtype=float)

        forma_n = (ultimo_n,) + (1,) * x_array.ndim # Forma de (numero de hamonico, x_arrat.ndim) no caso que x tem shape (1000, )
        fase = 2 * np.pi * harmonicos.reshape(forma_n) * x_array / self.L # Forma matriz (n_harmonico, n_pontos) 
        termos = forca_an.reshape(forma_n) * np.sin(fase) - forca_bn.reshape(forma_n) * np.cos(fase)
        return np.sum(termos, axis=0)



            
def funcao_resposta(t, tau_resposta):
    """Começa desligada e alterna ligada/desligada por intervalos iguais."""
    fase = np.mod(t, 2 * tau_resposta)
    if isinstance(fase, np.ndarray):
        return (fase >= tau_resposta).astype(float)
    return float(fase >= tau_resposta)