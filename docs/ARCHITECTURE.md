# Arquitetura e mapa dos materiais

O repositório é organizado em notebooks de experimento, um módulo Python auxiliar e dados HDF5. Não há pacote Python ou fluxo de execução automatizado identificado.

- `teste_básico.ipynb` define a classe `langevin`: sorteia incrementos gaussianos, avança as partículas com o termo de força e salva histogramas/estatísticas.
- `teste_potencial.ipynb` define potencial periódico e aproximações por séries de Fourier para explorar forças não nulas.
- `Foker_plank.ipynb` define `flock_plank_crank`, que monta e resolve sistemas tridiagonais para a evolução da densidade, incluindo comparação com uma solução analítica.
- `vizualizacao_initial.py` contém a classe `langevin` e função de visualização de histogramas salvos.
- Arquivos `.h5` guardam dados de simulação; esquema e conteúdo não foram inspecionados neste ambiente.

Há uma cópia da estrutura dentro de `Trabalho-de-Mec-nica-Estat-stica-/`, incluindo README, notebooks, script e dados. Os arquivos do script e de Fokker–Planck conferidos são idênticos entre raiz e subpasta.
