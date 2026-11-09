# Construção do operador de evolução em `Foker_plank.ipynb`

Este texto explica a construção implementada na classe `flock_plank_crank` do [notebook](../Foker_plank.ipynb), partindo da equação de Langevin do [enunciado](../TrabMecEst2026-2.pdf), seção I, equações (1)–(4). A discretização do interior da malha está montada; o tratamento físico das condições de contorno ainda está em aberto. As expressões abaixo distinguem essas duas partes.

## 1. Da equação de Langevin à equação de Fokker–Planck

A equação (1) do enunciado é

$$
\gamma\frac{dx}{dt}=-V'(x)f(t)+g\eta(t),
$$

com ruído branco de média nula e correlação

$$
\langle\eta(t)\eta(t')\rangle=\delta(t-t'),
\qquad g=\sqrt{2\gamma k_B T}.
$$

A função $f(t)$ alterna entre zero e um a cada intervalo $\tau$, começando com o potencial desligado. Escrevendo a equação em termos de um processo de Wiener $W_t$,

$$
dx=u(x,t)\,dt+\frac{g}{\gamma}\,dW_t,
\qquad u(x,t)=-\frac{V'(x)}{\gamma}f(t).
$$

Aqui, $u$ é a velocidade de deriva. Para obter a equação da densidade $p(x,t)$, podemos aplicar a fórmula de Itô a uma função teste suave $\varphi(x)$:

$$
\frac{d}{dt}\langle\varphi(x)\rangle
=\left\langle u\varphi'+\frac{g^2}{2\gamma^2}\varphi''\right\rangle.
$$

Escrevendo as médias como integrais contra $p$ e integrando por partes, inicialmente para funções teste de suporte compacto no interior do domínio, obtemos

$$
\boxed{\frac{\partial p}{\partial t}
=-\frac{\partial}{\partial x}(up)
+D\frac{\partial^2p}{\partial x^2}},
\qquad D=\frac{g^2}{2\gamma^2}=\frac{k_B T}{\gamma}.
$$

Essa derivação local não escolhe as condições de contorno: elas ainda precisam completar o problema no domínio adotado.

O notebook usa a simplificação $\gamma=1$, permitida na seção II do enunciado. Por isso, `self.D = g**2 / 2`, e a função chamada `F(x,t)` no código corresponde a $u(x,t)$. Nessa convenção, força e velocidade de deriva têm o mesmo valor numérico. Para trabalhar com $\gamma\ne1$, seria necessário dividir a força física por $\gamma$ e usar $D=g^2/(2\gamma^2)$.

## 2. Forma conservativa e corrente de probabilidade

Definimos a corrente

$$
J(x,t)=F(x,t)p(x,t)-D\frac{\partial p}{\partial x}.
$$

Então a equação assume a forma de continuidade:

$$
\frac{\partial p}{\partial t}=-\frac{\partial J}{\partial x}.
$$

O primeiro termo da corrente transporta probabilidade pela força; o segundo descreve a difusão. Essa forma é a base da montagem no notebook e evidencia por que as bordas importam:

$$
\frac{d}{dt}\int_{x_{\min}}^{x_{\max}}p(x,t)\,dx
=J(x_{\min},t)-J(x_{\max},t).
$$

Assim, a conservação da probabilidade depende do fluxo nas extremidades, além da discretização no interior.

## 3. Discretização espacial da corrente

Considere uma malha uniforme com $N$ pontos,

$$
x_i=x_{\min}+i\Delta x,\quad i=0,\ldots,N-1,
\qquad t_n=n\Delta t,
\qquad p_i^n\approx p(x_i,t_n).
$$

O método `gerar_matrix` verifica a uniformidade da malha. O método `F_half` calcula a força nas interfaces,

$$
x_{i+1/2}=\frac{x_i+x_{i+1}}2,
\qquad F_{i+1/2}^n=F(x_{i+1/2},t_n).
$$

Aproximamos a densidade na interface pela média dos pontos vizinhos e sua derivada por uma diferença centrada:

$$
J_{i+1/2}^n
=F_{i+1/2}^n\frac{p_i^n+p_{i+1}^n}{2}
-D\frac{p_{i+1}^n-p_i^n}{\Delta x}.
$$

Agrupando os coeficientes,

$$
J_{i+1/2}^n=a_{i+1/2}^np_i^n+b_{i+1/2}^np_{i+1}^n,
$$

$$
\boxed{a_{i+1/2}^n=\frac{F_{i+1/2}^n}{2}+\frac{D}{\Delta x}},
\qquad
\boxed{b_{i+1/2}^n=\frac{F_{i+1/2}^n}{2}-\frac{D}{\Delta x}}.
$$

Essas são as variáveis `a_n` e `b_n`. Os arrays `a_np1` e `b_np1` repetem o cálculo em $t_{n+1}$.

Para os pontos internos, $i=1,\ldots,N-2$,

$$
\frac{dp_i}{dt}=-\frac{J_{i+1/2}-J_{i-1/2}}{\Delta x}.
$$

Isso define o operador espacial discreto $L(t)$, cujas linhas internas são

$$
(L_np)_i=
\frac{a_{i-1/2}^n}{\Delta x}p_{i-1}
+\frac{b_{i-1/2}^n-a_{i+1/2}^n}{\Delta x}p_i
-\frac{b_{i+1/2}^n}{\Delta x}p_{i+1}.
$$

Cada ponto se acopla apenas aos dois vizinhos, dando origem à estrutura tridiagonal. A aproximação centrada tem ordem espacial dois para funções suficientemente suaves no interior.

## 4. Discretização temporal por Crank–Nicolson

O método combina os operadores no início e no fim do passo:

$$
\frac{p^{n+1}-p^n}{\Delta t}
=\frac12\left(L_{n+1}p^{n+1}+L_np^n\right).
$$

Reorganizando,

$$
\boxed{A_{n+1}p^{n+1}=B_np^n},
\qquad
A_{n+1}=I-\frac{\Delta t}{2}L_{n+1},
\qquad B_n=I+\frac{\Delta t}{2}L_n.
$$

Essas identidades descrevem as linhas internas; as linhas das bordas precisam ser especificadas separadamente. Os elementos internos de $A$ são

$$
A_{i,i-1}=-\frac{\Delta t}{2\Delta x}a_{i-1/2}^{n+1},
$$
$$
A_{i,i}=1+\frac{\Delta t}{2\Delta x}
\left(a_{i+1/2}^{n+1}-b_{i-1/2}^{n+1}\right),
$$
$$
A_{i,i+1}=+\frac{\Delta t}{2\Delta x}b_{i+1/2}^{n+1}.
$$

Os elementos internos de $B$ são

$$
B_{i,i-1}=+\frac{\Delta t}{2\Delta x}a_{i-1/2}^{n},
$$
$$
B_{i,i}=1-\frac{\Delta t}{2\Delta x}
\left(a_{i+1/2}^{n}-b_{i-1/2}^{n}\right),
$$
$$
B_{i,i+1}=-\frac{\Delta t}{2\Delta x}b_{i+1/2}^{n}.
$$

No código, o índice `j` de `a_n[j]` ou `b_n[j]` representa a interface $j+1/2$. Assim, `a_np1[1:] - b_np1[:-1]` fornece exatamente o termo da diagonal principal interna de $A$. As diagonais inferiores e superiores são passadas a `scipy.sparse.diags` com deslocamentos `-1` e `+1`.

## 5. O operador de evolução de um passo

Depois de completar as condições de contorno e supondo $A_{n+1}$ invertível, o propagador discreto é

$$
\boxed{U_n=A_{n+1}^{-1}B_n},
\qquad p^{n+1}=U_np^n.
$$

É útil distinguir $L_n$, o gerador espacial da evolução, de $U_n$, que avança a densidade por um intervalo finito $\Delta t$. As matrizes $A$ e $B$ são os fatores usados para aplicar esse avanço.

O método `evolucao` não calcula a inversa explicitamente: primeiro forma `Bpn = B @ p_atual`, depois resolve $Ap^{n+1}=Bpn$ com `resolver_thomas`. O algoritmo de Thomas explora a estrutura tridiagonal e tem custo linear em $N$ por passo, desde que a eliminação não encontre pivôs nulos.

Após vários passos,

$$
p^m=U_{m-1}\cdots U_1U_0p^0.
$$

Para força independente do tempo e contornos fixos, $L_n=L$ e $U=(I-\Delta t L/2)^{-1}(I+\Delta t L/2)$ aproxima $e^{\Delta t L}$. Para o potencial chaveado, os operadores dependem do tempo. A ordem temporal dois do método pressupõe regularidade durante o passo; as descontinuidades de $f(t)$ exigem separar os intervalos ligados e desligados e tratar cada passo com o operador da respectiva etapa, evitando uma média artificial entre os dois lados do salto.

## 6. O que as bordas fazem atualmente

O notebook atribui valor um às diagonais principais de $A$ e $B$ nas duas extremidades e zera os acoplamentos das respectivas linhas. Portanto, as equações dessas linhas são

$$
p_0^{n+1}=p_0^n,
\qquad p_{N-1}^{n+1}=p_{N-1}^n.
$$

Isso mantém os valores iniciais da densidade nas bordas. É um fechamento algébrico de valores fixos, mas ainda não representa uma escolha física justificada para o problema. Se os valores iniciais forem zero, equivale a impor densidade nula nas extremidades; para a gaussiana inicial, eles são numericamente muito pequenos. Densidade nula e corrente nula são condições diferentes.

As opções a considerar incluem:

- **Domínio amplo representando a reta:** afastar as bordas até que a densidade e os fluxos nelas sejam desprezíveis durante todo o intervalo estudado. É necessário verificar a convergência ao ampliar o domínio, pois o motor desloca a distribuição.
- **Bordas refletoras:** impor $J=0$ nas extremidades. Isso conserva probabilidade em um intervalo fechado, mas o confinamento altera o transporte em tempos longos.
- **Bordas absorventes:** impor $p=0$ nas extremidades. A massa no domínio pode diminuir; a solução passa a descrever a probabilidade de a partícula ainda não ter sido absorvida.
- **Bordas periódicas:** identificar as extremidades de uma célula espacial. A matriz passa a ter acoplamentos entre o primeiro e o último ponto, e o algoritmo de Thomas atual precisa ser adaptado para um sistema cíclico. Em uma célula periódica, a posição média local não acompanha o deslocamento acumulado na reta; o transporte deve ser obtido pela corrente ou por uma variável que registre as travessias entre células.

O potencial ser periódico não obriga a densidade na reta a ser periódica: o enunciado parte de uma distribuição localizada que se espalha e se desloca. A escolha do domínio deve acompanhar o observável que se deseja calcular.

Até essa escolha ser feita, não se deve afirmar que a evolução implementada conserva a normalização ou reproduz o regime assintótico do motor. Na forma discreta conservativa, os fluxos internos se cancelam ao somar as equações, mas restam os termos de fronteira; a quadratura da massa também precisa ser consistente com a representação usada nas bordas.

## 7. Correspondência com o exemplo e pendências do notebook

O exemplo usa $x\in[-5,5]$, $N=401$, $\Delta x=0{,}025$, $\Delta t=0{,}001$ e $g=1$, portanto $D=0{,}5$. A condição inicial é uma gaussiana normalizada numericamente, de desvio padrão $0{,}1$, que aproxima a delta inicial do enunciado com largura finita.

Além dos contornos, a versão inspecionada apresenta os seguintes pontos:

- Em `F(x,t)`, `resposta = funcao_resposta(t, tau_resposta)` é calculada, mas o retorno é apenas `forca`. Para representar a equação (1), nessa convenção com $\gamma=1$, a força deve incorporar o fator `resposta`. Atualmente, o exemplo monta um operador para potencial estático.
- `evolucao` tem `salvar_distribuicao=False` como padrão, mas escreve em `p_completa[n+1]` incondicionalmente. Como a chamada do exemplo omite esse argumento, a execução a partir do estado atual encontra uma tentativa de escrita em `None`. Isso é separado da derivação das matrizes.
- `calcular_estatistica` calcula a normalização, mas não a usa nem a retorna. As expressões de média e variância ali pressupõem uma densidade normalizada. A massa deve ser monitorada antes de interpretar esses resultados; renormalizar uma solução com perda física de massa mudaria sua interpretação.
- A curva marcada como `Analítica` na visualização é novamente `p[0]`, não uma solução analítica evoluída no tempo.

Como verificações futuras, após definir os contornos e ajustar a execução, convém acompanhar a massa, a menor densidade e a convergência com os passos espacial e temporal. Crank–Nicolson não garante por si só a positividade da densidade; a discretização centrada da deriva também pode gerar oscilações. Um caso de referência é a difusão livre na reta: para a gaussiana inicial, enquanto as bordas forem desprezíveis, a média permanece zero e a variância deve seguir $\sigma^2(t)=\sigma_0^2+2Dt$.

Este documento descreve a montagem existente e suas limitações; não modifica o notebook nem resolve a escolha das condições de contorno.
