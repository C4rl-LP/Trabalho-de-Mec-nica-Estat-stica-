# Próximos passos: periodicidade e maximização do drift

Este roteiro dá continuidade à [construção do operador de evolução](operador_evolucao_fokker_planck.md). As tarefas abaixo são propostas de implementação e investigação; não representam resultados já obtidos.

## 1. Ajustar o propagador para condições periódicas

- Usar uma célula do potencial, de comprimento `L`, com malha uniforme e `endpoint=False`, sem duplicar a extremidade.
- Calcular a corrente nas `N` interfaces, incluindo aquela que conecta o último ponto ao primeiro.
- Montar o gerador espacial com índices módulo `N`. As matrizes de Crank–Nicolson passam a ser tridiagonais cíclicas, com acoplamentos nos dois cantos.
- Resolver `A @ p_novo = B @ p_atual` com um solver que aceite esses acoplamentos, inicialmente `scipy.sparse.linalg.spsolve`. O algoritmo de Thomas atual não os contempla.
- Aplicar efetivamente o chaveamento à força, retornando `resposta * forca`, e corrigir a escrita do histórico quando seu armazenamento estiver desativado.
- Tratar cada etapa ligada ou desligada separadamente, fazendo os passos terminarem nos instantes de chaveamento. Dentro de cada etapa, usar o operador correspondente, sem misturar os dois lados da descontinuidade.

Validar a conservação da massa discreta `dx * sum(p)`, a ausência de densidades negativas relevantes e a convergência ao refinar espaço e tempo. Usar como referências a difusão livre e o potencial estático, que deve ter corrente estacionária nula na ausência de inclinação externa. Se surgirem oscilações ou densidades negativas, rever o esquema antes de otimizar.

**Resultado esperado:** evolução periódica espacialmente consistente, com corrente calculável e erro numérico controlado.

## 2. Definir e medir a velocidade de drift

Em uma célula periódica, a média da coordenada local não registra as voltas acumuladas. Usar a corrente para medir a velocidade na reta:

$$
v(t)\approx\Delta x\sum_{i=0}^{N-1}J_{i+1/2}(t).
$$

Após descartar o transiente, calcular a média sobre ciclos completos de período $2\tau$:

$$
\widehat v_B(\theta)=\frac{1}{2K\tau}
\int_{t_0}^{t_0+2K\tau}v(t;\theta)\,dt.
$$

Aqui, $\theta$ reúne os parâmetros ajustáveis e $K$ é o número de ciclos de medição. Integrar a corrente com uma quadratura temporal consistente com o avanço da densidade. Verificar a convergência comparando densidades na mesma fase de ciclos sucessivos e as velocidades médias por ciclo.

Construir uma função determinística que receba os parâmetros e retorne a velocidade, acompanhada de diagnósticos de massa, positividade e convergência. Estados inválidos devem ser identificados; não corrigir artificialmente a densidade para aumentar o objetivo.

## 3. Escolher os parâmetros e os limites da busca

Começar fixando `gamma`, `L` e `g` e investigando `tau`. Depois ampliar a busca para a assimetria `a/L`, a amplitude `V_0` e a suavização `sigma/L`. Manter `b = L - a`, pois ele não é independente.

Definir intervalos finitos antes da otimização, respeitando `0 < a/L < 1`, suavização positiva, amplitude e tempo positivos. Variar livremente todas as escalas físicas não define uma comparação única. Os limites e os parâmetros mantidos fixos devem acompanhar os resultados.

Para comparar regimes, usar também os grupos adimensionais

$$
\frac{a}{L},\qquad\frac{\sigma}{L},\qquad
\frac{V_0}{\gamma D},\qquad\frac{D\tau}{L^2},
\qquad \widetilde v_B=\frac{Lv_B}{D}.
$$

Como $\gamma D=k_BT$, a terceira razão compara a barreira à energia térmica. Inicialmente, manter a rampa desligada; se ela for incluída depois, sua inclinação passa a ser outro parâmetro físico.

## 4. Comparar estratégias de otimização

Os algoritmos de minimização podem receber o objetivo $f(\theta)=-\widehat v_B(\theta)$, para maximizar o drift na direção positiva escolhida.

| Estratégia | Uso proposto | Limitação a avaliar |
|---|---|---|
| Varredura inicial | Explorar `tau` em escala logarítmica e alguns valores de assimetria; visualizar regiões promissoras | O custo cresce rapidamente com o número de parâmetros |
| `scipy.optimize.differential_evolution` | Busca global estocástica com limites, sem exigir gradientes | Pode exigir muitas simulações; não comprova um máximo global |
| `scipy.optimize.minimize(method="Powell")` | Refinamento local sem derivadas, partindo de bons candidatos | Depende da região inicial e pode terminar em ótimos locais |
| `scipy.optimize.minimize(method="L-BFGS-B")` | Refinamento local com limites e gradientes por diferenças finitas ou fornecidos pelo usuário | Requer um objetivo numericamente suave e derivadas confiáveis |
| Diferenciação automática com JAX | Calcular gradientes através de uma implementação diferenciável da evolução e usá-los no ajuste | Exige adaptar o solver; o código NumPy/SciPy atual não se torna diferenciável apenas ao chamar `jax.grad` |

As capacidades dos métodos constam nas documentações de [differential evolution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html), [minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) e [diferenciação automática do JAX](https://docs.jax.dev/en/latest/automatic-differentiation.html). A escolha e a ordem de aplicação acima são propostas para este projeto.

### Cuidado especial com gradientes em relação a `tau`

Usar `floor`, módulo ou uma comparação booleana para chavear a força em uma malha temporal fixa pode tornar o objetivo discretizado descontínuo ou localmente constante em `tau`. Isso prejudica tanto diferenças finitas quanto diferenciação automática.

Uma alternativa é separar as duas etapas e usar um número fixo de passos por meia fase, com `dt = tau / M`. Assim, variar `tau` altera a duração dos passos sem mudar abruptamente a sequência ligada/desligada. É necessário escolher `M` e validar a precisão em todo o intervalo pesquisado. Para gradientes, também convém trabalhar com um número fixo e suficientemente grande de ciclos, verificando separadamente o erro do transiente.

Conferir qualquer gradiente automático com diferenças finitas em alguns pontos. Como alternativa futura ao JAX, pode-se diferenciar os sistemas lineares do propagador para obter sensibilidades dos parâmetros.

## 5. Sequência proposta e validação do melhor candidato

1. Concluir e validar o propagador periódico e a medição da corrente.
2. Fazer uma varredura de `tau` para obter a curva de velocidade e identificar regiões promissoras.
3. Expandir para poucos parâmetros e aplicar `differential_evolution`, com orçamento de avaliações e sementes registrados.
4. Refinar os melhores candidatos com Powell; comparar com L-BFGS-B se o objetivo tiver suavidade suficiente.
5. Avaliar JAX apenas se o custo das simulações e a qualidade dos gradientes justificarem adaptar o solver.
6. Reavaliar os melhores candidatos com malha mais fina, mais passos temporais, mais modos de Fourier e maior tempo de relaxação. Comparar também com parâmetros vizinhos para medir a sensibilidade do resultado.
7. Comparar a velocidade obtida com simulações independentes de Euler–Maruyama, incluindo a incerteza estatística dessas trajetórias.

**Entregáveis:** solver periódico validado, curva de drift em função de `tau`, comparação do custo e dos resultados dos métodos, e conjunto de parâmetros com a maior velocidade encontrada dentro dos limites investigados. Relatar o erro numérico e evitar interpretar a convergência de um otimizador como prova de máximo global.
