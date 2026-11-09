# Estado atual

Verificação em 2026-11-07.

## Escopo e inspeção

O README descreve simulação da trajetória de uma partícula pela equação estocástica de Langevin. Inspecionei a árvore, o script `vizualizacao_initial.py` e as células/documentação dos notebooks `teste_básico.ipynb`, `teste_potencial.ipynb` e `Foker_plank.ipynb`.

## Observado

- O notebook básico descreve a atualização de Langevin com ruído gaussiano de variância `dt` e registra histogramas em HDF5.
- O notebook de potencial estuda uma forma periódica por série de Fourier.
- O notebook Fokker–Planck implementa uma evolução chamada Crank–Nicolson e compara um caso com expressão analítica.
- `utils_coisa/solver.py` contém o solver de Langevin; `vizualizacao_initial.py` reexporta a classe para manter compatibilidade. O notebook `simulacao_langevin_serra_gaussiana.ipynb` usa o solver para a simulação e para estimar a inclinação da média em função de `tau_resposta`. A classe agora também aceita um gerador aleatório opcional e retorna os tempos após cada passo. O notebook ainda não foi executado.
- `vizualizacao_initial.py` mantém a visualização de histogramas com imports locais de HDF5 e widgets; importar o solver não exige essas dependências opcionais.
- O repositório contém uma subpasta espelhada. Alguns arquivos conferidos são idênticos entre as duas cópias; não determinei qual localização deve ser mantida como fonte principal.

## Limites e pendências

- Não executei notebooks, testes ou simulações. As figuras/saídas embutidas não foram revalidadas.
- Não inspecionei datasets HDF5 porque `h5py` não está instalado no ambiente.
- Não há manifesto de dependências, arquitetura modular ou decisões científicas formalizadas.
- Antes de alterar código, definir qual cópia é canônica e inspecionar as células relevantes, parâmetros, dados e saídas.
