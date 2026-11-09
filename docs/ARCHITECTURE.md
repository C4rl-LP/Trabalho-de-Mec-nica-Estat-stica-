# Arquitetura e mapa dos materiais

O repositório é organizado em notebooks de experimento, um módulo Python auxiliar e dados HDF5. Não há pacote Python ou fluxo de execução automatizado identificado.

- `teste_básico.ipynb` define a classe `langevin`: sorteia incrementos gaussianos, avança as partículas com o termo de força e salva histogramas/estatísticas.
- `teste_potencial.ipynb` define potencial periódico e aproximações por séries de Fourier para explorar forças não nulas.
- `Foker_plank.ipynb` define `flock_plank_crank`, que monta e resolve sistemas tridiagonais para a evolução da densidade, incluindo comparação com uma solução analítica.
- `vizualizacao_initial.py` reexporta o solver de Langevin e contém a visualização interativa de histogramas salvos.
- `utils_coisa/potencial_e_forca.py` contém o potencial serra e suas séries de Fourier atenuadas; `utils_coisa/solver.py` contém o solver de Langevin; `simulacao_langevin_serra_gaussiana.ipynb` usa ambos para a simulação e a varredura da velocidade de drift.
- Arquivos `.h5` guardam dados de simulação; esquema e conteúdo não foram inspecionados neste ambiente.

