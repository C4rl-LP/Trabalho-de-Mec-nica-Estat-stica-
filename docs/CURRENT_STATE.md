# Estado atual

Verificação em 2026-11-07.

## Escopo e inspeção

O README descreve simulação da trajetória de uma partícula pela equação estocástica de Langevin. Inspecionei a árvore, o script `vizualizacao_initial.py` e as células/documentação dos notebooks `teste_básico.ipynb`, `teste_potencial.ipynb` e `Foker_plank.ipynb`.

## Observado

- O notebook básico descreve a atualização de Langevin com ruído gaussiano de variância `dt` e registra histogramas em HDF5.
- O notebook de potencial estuda uma forma periódica por série de Fourier.
- O notebook Fokker–Planck implementa uma evolução chamada Crank–Nicolson e compara um caso com expressão analítica.
- O script auxiliar contém código de simulação e visualização; a função de visualização referencia `Slider` e `CheckButtons`, cujos imports não aparecem no trecho inspecionado.
- O repositório contém uma subpasta espelhada. Alguns arquivos conferidos são idênticos entre as duas cópias; não determinei qual localização deve ser mantida como fonte principal.

## Limites e pendências

- Não executei notebooks, testes ou simulações. As figuras/saídas embutidas não foram revalidadas.
- Não inspecionei datasets HDF5 porque `h5py` não está instalado no ambiente.
- Não há manifesto de dependências, arquitetura modular ou decisões científicas formalizadas.
- Antes de alterar código, definir qual cópia é canônica e inspecionar as células relevantes, parâmetros, dados e saídas.
