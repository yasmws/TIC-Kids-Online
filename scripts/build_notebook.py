"""Gera o notebook do estudo TIC Kids Online 2024.

Uso (a partir da pasta PROJ 2):
    python scripts/build_notebook.py notebooks/02_tic_kids_wilcoxon_pareado.ipynb
    jupyter nbconvert --to notebook --execute --inplace notebooks/02_tic_kids_wilcoxon_pareado.ipynb
"""
import sys
from pathlib import Path

import nbformat as nbf

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "notebooks/02_tic_kids_wilcoxon_pareado.ipynb")
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip()))


# =============================================================================
# CAPA
# =============================================================================
md(r"""
# Mediação parental, habilidades digitais e experiências negativas na Internet
### Um estudo não paramétrico com os microdados da TIC Kids Online Brasil 2024

**Base de dados:** TIC Kids Online Brasil 2024, microdados v1.2, Cetic.br/NIC.br.
**Foco metodológico:** testes de hipóteses não paramétricos (Wilcoxon pareado, McNemar, Kruskal-Wallis com teste de Dunn e qui-quadrado de independência), complementados por uma análise que respeita o plano amostral complexo.

| Seção | Conteúdo |
|---|---|
| 0 | Configuração do ambiente e funções auxiliares |
| 1 | Introdução e contextualização |
| 2 | Compreensão e documentação do dataset |
| 3 | Preparação e qualidade dos dados |
| 4 | Análise exploratória de dados (EDA) |
| 5 | Formulação das hipóteses estatísticas |
| 6 | Seleção e justificativa dos testes |
| 7 | Execução dos testes de hipóteses |
| 8 | Visualização e comunicação dos resultados |
| 9 | Discussão |
| 10 | Conclusão |
| 11 | Referências e reprodutibilidade |

> **Como ler este notebook.** Cada etapa segue a sequência *O que vamos fazer? → Por que isso é necessário? → Como vamos fazer? → O que encontramos? → Como interpretar?*. Os números citados nos textos são gerados pelas próprias células, que imprimem os resultados da execução. Os textos fixos descrevem apenas padrões que foram conferidos na execução.
""")

# =============================================================================
# 0. CONFIGURAÇÃO
# =============================================================================
md(r"""
## 0. Configuração do ambiente

**O que vamos fazer?** Importar as bibliotecas, fixar a semente aleatória, localizar os arquivos e definir funções auxiliares reutilizáveis.

**Por que isso é necessário?** Centralizar parâmetros e funções evita código duplicado e garante que o notebook rode do início ao fim, sem depender de estados ocultos.
""")
code(r'''
# 0.1 Bibliotecas e parâmetros globais
import platform
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import plotly.io as pio
import scipy
import statsmodels
from IPython.display import Markdown, display
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.stats.contingency_tables import SquareTable, mcnemar
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option("display.max_colwidth", 110)
pd.set_option("display.width", 160)

SEED = 2024          # semente única para todos os procedimentos aleatórios (bootstrap)
ALPHA = 0.05         # nível de significância adotado em todos os testes
B_BOOT = 2000        # reamostragens bootstrap para intervalos de confiança de efeitos
''')
code(r'''
# 0.2 Localização dos arquivos (baixa a base oficial se ela não estiver presente)
def localizar_raiz():
    """Retorna a pasta do projeto: a que contém data/tic_kids_online_2024 (ou a pai de notebooks/)."""
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / "data" / "tic_kids_online_2024").exists():
            return p
    return Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()


RAIZ = localizar_raiz()
PASTA_DADOS = RAIZ / "data" / "tic_kids_online_2024"
ARQ_DADOS = PASTA_DADOS / "tic_kids_online_brasil_2024_microdados_v1.2.csv"
ARQ_DICIO = PASTA_DADOS / "tic_kids_online_brasil_2024_dicionario_de_variaveis_v1.1.xlsx"
URLS = {
    ARQ_DADOS: "https://cetic.br/media/microdados/844/tic_kids_online_brasil_2024_base_de_microdados_v1.2.csv",
    ARQ_DICIO: "https://cetic.br/media/microdados/842/tic_kids_online_brasil_2024_dicionario_de_variaveis_v1.1.xlsx",
}
for arquivo, url in URLS.items():
    if not arquivo.exists():
        import urllib.request
        PASTA_DADOS.mkdir(parents=True, exist_ok=True)
        print(f"Baixando {arquivo.name} ...")
        urllib.request.urlretrieve(url, arquivo)
print("Pasta do projeto:", RAIZ)
''')
code(r'''
# 0.3 Formatação de números (padrão brasileiro) e identidade visual dos gráficos
def fmt(x, casas=2):
    """1234.5 -> '1.234,50'."""
    return f"{x:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_pct(x, casas=1):
    return fmt(100 * x, casas) + "%"


def fmt_p(p):
    """p-valor em texto: 'p < 0,001' ou 'p = 0,043'."""
    return "p < 0,001" if p < 0.001 else f"p = {fmt(p, 3)}"


def mostrar(texto):
    display(Markdown(texto))


COR = {"resp": "#1f6f8b", "cri": "#e08e45", "neutro": "#9aa5b1", "destaque": "#c0392b",
       "seq": ["#c6dbef", "#6baed6", "#2171b5", "#08306b"]}
pio.templates["tic"] = go.layout.Template(layout=dict(
    font=dict(family="Arial, Helvetica, sans-serif", size=14, color="#222"),
    title=dict(x=0.01, xanchor="left", font=dict(size=17)),
    paper_bgcolor="white", plot_bgcolor="white",
    xaxis=dict(showgrid=False, linecolor="#555", ticks="outside", title_standoff=8),
    yaxis=dict(gridcolor="#e6e6e6", linecolor="#555", zeroline=False),
    colorway=[COR["resp"], COR["cri"], "#4daf4a", "#984ea3", "#ff7f00", "#a65628"],
    legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
    margin=dict(t=90, l=70, r=30, b=60), hoverlabel=dict(font_size=13),
))
pio.templates.default = "simple_white+tic"
''')
code(r'''
# 0.4 Funções estatísticas reutilizáveis
ESPECIAIS = {88: "Não entendeu", 97: "Não sabe", 98: "Não respondeu", 99: "Não se aplica"}


def sem_especiais(s):
    """Converte os códigos especiais (88, 97, 98, 99) em ausente (NaN)."""
    return s.where(~s.isin(list(ESPECIAIS)))


def para_frequencia(s):
    """Escala 1=Sempre ... 5=Nunca -> 5=Sempre ... 1=Nunca (maior = mais frequente), sem códigos especiais."""
    return 6 - sem_especiais(s)


def correlacao_bisserial_pareada(d):
    """Correlação bisserial de postos para dados pareados (Kerby, 2014): (W+ - W-)/(W+ + W-)."""
    d = np.asarray(d)
    d = d[d != 0]
    postos = stats.rankdata(np.abs(d))
    return (postos[d > 0].sum() - postos[d < 0].sum()) / postos.sum()


def wilcoxon_pareado(x, y):
    """Teste dos postos sinalizados de Wilcoxon, implementado passo a passo e conferido com o SciPy.

    Empates (d = 0) são descartados (procedimento de Wilcoxon); postos médios para |d| repetidos;
    variância com correção de empates; aproximação normal sem correção de continuidade.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    d_todos = x - y
    d = d_todos[d_todos != 0]
    n = len(d)
    postos = stats.rankdata(np.abs(d))
    w_pos, w_neg = postos[d > 0].sum(), postos[d < 0].sum()
    _, t = np.unique(np.abs(d), return_counts=True)
    media = n * (n + 1) / 4
    var = n * (n + 1) * (2 * n + 1) / 24 - (t**3 - t).sum() / 48
    z = (w_pos - media) / np.sqrt(var)
    p = 2 * stats.norm.sf(abs(z))
    ref = stats.wilcoxon(x, y, zero_method="wilcox", correction=False, method="approx")
    assert np.isclose(ref.pvalue, p, rtol=1e-6), "Implementação manual diverge do SciPy"
    assert np.isclose(ref.statistic, min(w_pos, w_neg)), "Estatística manual diverge do SciPy"
    return dict(n_pares=len(d_todos), n_empates=int((d_todos == 0).sum()), n_util=n,
                W_pos=w_pos, W_neg=w_neg, media_W=media, var_W=var, z=z, p=p,
                r=z / np.sqrt(len(d_todos)), rbc=(w_pos - w_neg) / (w_pos + w_neg))


def ic_bootstrap(estatistica, *arrays, B=B_BOOT, seed=SEED, nivel=0.95):
    """IC percentil por bootstrap não paramétrico (reamostra linhas com reposição)."""
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    est = np.array([estatistica(*(a[idx] for a in arrays))
                    for idx in (rng.integers(0, n, n) for _ in range(B))])
    a = (1 - nivel) / 2
    return np.quantile(est, [a, 1 - a])


def epsilon_quadrado(H, n_total):
    """Tamanho de efeito do Kruskal-Wallis: epsilon² = H / (n - 1) (Tomczak & Tomczak, 2014)."""
    return H / (n_total - 1)


def teste_dunn(valores, grupos):
    """Teste de Dunn (1964): postos conjuntos, correção de empates, p-valores ajustados por Holm."""
    dados = pd.DataFrame({"v": np.asarray(valores), "g": np.asarray(grupos)})
    dados["posto"] = stats.rankdata(dados.v)
    n = len(dados)
    _, t = np.unique(dados.v, return_counts=True)
    correcao = (t**3 - t).sum() / (12 * (n - 1))
    resumo = dados.groupby("g").posto.agg(["mean", "size"])
    linhas = []
    nomes = list(resumo.index)
    for i in range(len(nomes)):
        for j in range(i + 1, len(nomes)):
            a, b = resumo.loc[nomes[i]], resumo.loc[nomes[j]]
            ep = np.sqrt((n * (n + 1) / 12 - correcao) * (1 / a["size"] + 1 / b["size"]))
            z = (a["mean"] - b["mean"]) / ep
            linhas.append(dict(comparacao=f"{nomes[i]} × {nomes[j]}", posto_medio_1=a["mean"],
                               posto_medio_2=b["mean"], z=z, p=2 * stats.norm.sf(abs(z))))
    saida = pd.DataFrame(linhas)
    saida["p_holm"] = multipletests(saida.p, method="holm")[1]
    return saida
''')
code(r'''
# 0.5 Estimação com o plano amostral: pesos replicados bootstrap (Rao-Wu), como no pacote survey do R
def estimativa_com_replicas(estimador, base, nivel=0.95):
    """Aplica `estimador(pesos)` ao PESO e às 200 réplicas.

    Variância bootstrap: V = 1/(R-1) * soma_r (theta_r - média(theta_r))².
    Retorna estimativa pontual, erro-padrão, IC normal e estatística z (H0: theta = 0).
    """
    theta = estimador(base["PESO"].to_numpy())
    reps = np.array([estimador(base[c].to_numpy()) for c in COLS_REPLICAS])
    ep = reps.std(ddof=1)
    zc = stats.norm.isf((1 - nivel) / 2)
    return dict(estimativa=theta, ep=ep, ic_inf=theta - zc * ep, ic_sup=theta + zc * ep,
                z=theta / ep if ep > 0 else np.nan,
                p=2 * stats.norm.sf(abs(theta / ep)) if ep > 0 else np.nan)


def media_ponderada(valores, pesos, mascara=None):
    valores, pesos = np.asarray(valores, float), np.asarray(pesos, float)
    if mascara is not None:
        valores, pesos = valores[mascara], pesos[mascara]
    return np.sum(valores * pesos) / np.sum(pesos)
''')

# =============================================================================
# 1. INTRODUÇÃO
# =============================================================================
md(r"""
## 1. Introdução e contextualização do estudo

### 1.1. Contexto
Crianças e adolescentes brasileiros começam a usar a Internet cada vez mais cedo, e a **mediação parental**, ou seja, o conjunto de práticas com que os responsáveis acompanham, orientam e restringem esse uso, é considerada um fator central para ampliar oportunidades e reduzir riscos online (Livingstone & Helsper, 2008). Boa parte do que se sabe sobre a mediação vem do relato de **um único informante**, em geral o responsável. Se responsáveis e crianças descrevem a mesma prática de formas diferentes, indicadores baseados apenas no relato dos adultos podem retratar a mediação de forma distorcida.

A TIC Kids Online Brasil entrevista, **no mesmo domicílio**, o responsável e a criança sorteada, e faz a ambos perguntas equivalentes sobre mediação. Esse desenho pareado permite comparar diretamente as duas perspectivas.

### 1.2. Motivação e relevância
* **Metodológica:** saber se o relato do responsável pode substituir o da criança na mensuração da mediação.
* **Prática:** orientar políticas e campanhas de educação digital, que dependem de saber se a orientação de fato chega à criança.
* **Didática:** a base tem variáveis ordinais, binárias, pareadas e independentes, o que a torna adequada para demonstrar diferentes testes não paramétricos.

### 1.3. Objetivos
**Objetivo geral:** investigar, com testes não paramétricos, a concordância entre responsáveis e crianças sobre a mediação parental e examinar dois fatores associados ao uso da Internet: idade e sexo.

**Objetivos específicos:**
1. Comparar, no mesmo domicílio, a frequência de conversa sobre a Internet relatada pelo responsável e pela criança (H1 e H2).
2. Verificar se o padrão se mantém nos 14 itens de mediação perguntados aos dois informantes (análise de robustez).
3. Comparar o nível de habilidades digitais entre faixas etárias de 11 a 17 anos (H3).
4. Verificar se a proporção de crianças tratadas de forma ofensiva na Internet difere entre meninos e meninas (H4).
5. Avaliar se as conclusões mudam quando se incorpora o plano amostral complexo (pesos replicados).

### 1.4. Perguntas de pesquisa
* **P1.** Responsáveis e crianças relatam a mesma frequência de conversa sobre o que a criança faz na Internet?
* **P2.** As habilidades digitais autodeclaradas diferem entre as faixas etárias de 11–12, 13–14 e 15–17 anos?
* **P3.** A proporção de crianças que foram tratadas de forma ofensiva na Internet nos últimos 12 meses difere entre meninos e meninas?

### 1.5. Caracterização inicial das hipóteses
* **H1** (ordinal, pareada): a distribuição das diferenças *responsável − criança* não é simétrica em torno de zero.
* **H2** (binária, pareada): a proporção de "Sempre" difere entre o relato do responsável e o da criança.
* **H3** (ordinal/escore, 3 grupos independentes): ao menos uma faixa etária tende a ter escores de habilidade diferentes das demais.
* **H4** (binária, 2 grupos independentes): a proporção de "foi tratado de forma ofensiva" depende do sexo.

A formulação formal está na Seção 5.

### 1.6. Metodologia em resumo
Análise exploratória, seguida de testes não paramétricos clássicos (Wilcoxon pareado, McNemar, Kruskal-Wallis + Dunn, qui-quadrado/Fisher), com tamanhos de efeito e intervalos de confiança, correção de Holm para múltiplas comparações e uma **camada de inferência populacional** com os 200 pesos replicados da pesquisa.

### 1.7. Contribuição e limitações antecipadas
A contribuição é mostrar, com dados oficiais recentes, se a perspectiva do responsável e a da criança coincidem, e quanto. As limitações principais, discutidas na Seção 9, são três. O estudo é observacional e transversal, então não permite conclusões causais. Os testes clássicos supõem amostragem aleatória simples, o que não é o caso. E as hipóteses foram definidas depois de uma exploração preliminar desta mesma base, sem pré-registro.
""")

# =============================================================================
# 2. DATASET
# =============================================================================
md(r"""
## 2. Compreensão e documentação do dataset

**O que vamos fazer?** Documentar a origem, o desenho amostral e as variáveis usadas, consultando o **dicionário oficial** e os relatórios metodológico e de coleta da pesquisa.

**Por que isso é necessário?** O significado de uma variável não pode ser presumido pelo nome. Por exemplo, o código 99 não é um valor alto da escala, e sim "Não se aplica". O plano amostral também define o tipo de inferência que é válido.

### 2.1. Ficha técnica

| Item | Descrição |
|---|---|
| Nome | TIC Kids Online Brasil 2024, base de microdados v1.2 |
| Instituição | Centro Regional de Estudos para o Desenvolvimento da Sociedade da Informação (Cetic.br), do Núcleo de Informação e Coordenação do Ponto BR (NIC.br) |
| Período de referência | Coleta de março a agosto de 2024. Algumas perguntas se referem aos últimos 3 ou 12 meses |
| População-alvo | Crianças e adolescentes de 9 a 17 anos residentes em domicílios particulares permanentes no Brasil |
| Unidade de observação | A criança ou adolescente sorteado no domicílio (uma por domicílio). O responsável é um respondente associado, não uma amostra representativa de pais |
| Coleta | Entrevista face a face com tablet (CAPI), com seções de autopreenchimento (CASI) |
| Plano amostral | Amostragem estratificada de conglomerados em 3 estágios (UPA → domicílio → morador), com 88 estratos (capitais, regiões metropolitanas e UF × situação urbana/rural). UPA selecionadas com probabilidade proporcional à raiz quadrada do número de domicílios |
| Pesos | `PESO` (corrigido para não resposta e calibrado por IPU) e 200 pesos replicados `rep1`–`rep200`, gerados por bootstrap com o mesmo desenho (`as.svrepdesign` do pacote *survey* do R) |
| Fontes | Dicionário de variáveis v1.1, relatório metodológico e relatório de coleta de dados (cetic.br) |
""")
code(r'''
# 2.2 Carga dos dados. `dados_brutos` é preservado intacto; todas as transformações usam cópias.
dados_brutos = pd.read_csv(ARQ_DADOS, sep=";", decimal=",", low_memory=False).copy()
COLS_REPLICAS = [f"rep{i}" for i in range(1, 201)]

dicionario = pd.read_excel(ARQ_DICIO, header=1)
dicionario.columns = ["tamanho", "variavel", "descricao", "tipo_arquivo", "codigos"]
dicionario = dicionario.dropna(subset=["variavel"]).set_index("variavel")

print(f"Registros: {dados_brutos.shape[0]:,} | Variáveis: {dados_brutos.shape[1]}".replace(",", "."))
print(f"Variáveis descritas no dicionário: {len(dicionario)} linhas (as réplicas rep1–rep200 aparecem numa linha só)")
print(f"Soma do PESO (população estimada de 9 a 17 anos): {fmt(dados_brutos.PESO.sum(), 0)}")
''')
md(r"""
### 2.3. Variáveis utilizadas

A tabela abaixo é montada **a partir do dicionário oficial**. A descrição e os códigos vêm do arquivo do Cetic.br; o tipo estatístico e a utilização são a classificação adotada neste estudo.
""")
code(r'''
def codigos_do_dicionario(var):
    """Extrai os códigos numéricos válidos ('1 = "..."') da coluna de rótulos do dicionário."""
    texto = dicionario.loc[var, "codigos"]
    if pd.isna(texto):
        return None
    return sorted({int(c) for c in pd.Series(str(texto)).str.findall(r"(-?\d+)\s*=").iloc[0]})


ITENS_HAB = [c for c in dados_brutos.columns if c.startswith(("R2A_", "R4_", "R5_"))]
VARIAVEIS = [
    ("E9_G", "Ordinal (5 níveis)", "H1, H2 (relato do responsável)"),
    ("U6_G", "Ordinal (5 níveis)", "H1, H2 (relato da criança)"),
    (ITENS_HAB[0], "Ordinal (4 níveis)", f"H3: um dos {len(ITENS_HAB)} itens do escore de habilidades"),
    ("T1A", "Binária", "H4 (desfecho)"),
    ("SEXO_KIDS", "Nominal binária", "H4 (grupo), EDA"),
    ("FAIXA_ETARIA", "Ordinal (4 níveis)", "H3 (grupo), EDA"),
    ("IDADE_KIDS", "Quantitativa discreta", "EDA, Spearman"),
    ("CLASSE_2015_REC", "Ordinal (3 níveis)", "EDA (possível confundidor)"),
    ("COD_REGIAO_2", "Nominal (5 níveis)", "EDA (representatividade)"),
    ("RESP_SEXO", "Nominal binária", "EDA (moderador exploratório)"),
    ("D1", "Binária", "Regra de salto (relato do responsável)"),
    ("D2", "Nominal", "Regra de salto (relato do responsável)"),
    ("M3", "Nominal", "Regra de salto (relato da criança)"),
    ("IDADE_INTERNET", "Binária (filtro)", "Elegibilidade para H3"),
    ("PESO", "Quantitativa contínua", "Inferência populacional"),
]
tabela_variaveis = pd.DataFrame([{
    "Variável": v, "Descrição (dicionário)": dicionario.loc[v, "descricao"], "Tipo": tipo,
    "Valores possíveis": (str(dicionario.loc[v, "codigos"]).replace("\n", "; ")
                          if pd.notna(dicionario.loc[v, "codigos"]) else
                          f"{fmt(dados_brutos[v].min(), 0)} a {fmt(dados_brutos[v].max(), 0)}"),
    "Utilização": uso} for v, tipo, uso in VARIAVEIS])
tabela_variaveis
''')
md(r"""
Além dessas, a análise de robustez usa os 14 pares de itens de mediação perguntados aos dois informantes: `E10_A`–`E10_G` ↔ `U5_A`–`U5_G` e `E9_A`–`E9_G` ↔ `U6_A`–`U6_G`.

**Classificação das variáveis por tipo:**
* **Ordinais:** as escalas de frequência (E9, E10, U5, U6), os itens de habilidade (R2A, R4, R5), faixa etária e classe social.
* **Nominais:** região, sexo da criança e sexo do responsável.
* **Binárias:** T1A, D1 e IDADE_INTERNET.
* **Quantitativas:** idade (discreta) e pesos (contínua).

**Códigos especiais** (88 = Não entendeu, 97 = Não sabe, 98 = Não respondeu, 99 = Não se aplica) não pertencem às escalas e são tratados na Seção 3.
""")
md(r"""
### 2.4. Critérios de inclusão e exclusão

| Análise | População analisada | Inclusão | Exclusão |
|---|---|---|---|
| H1, H2 | Crianças usuárias de Internet segundo **os dois** informantes | `E9_G` e `U6_G` com resposta na escala 1–5 | Não usuários (código 99) e "Não sabe"/"Não respondeu" (97/98) em qualquer um dos dois |
| H3 | Usuários de Internet de 11 a 17 anos (`IDADE_INTERNET = 1`) | Pelo menos 80% dos 17 itens de habilidade válidos | Crianças de 9–10 anos (não recebem o bloco) e respostas 88/97/98 acima do limite |
| H4 | Crianças usuárias de Internet (`M3 = 1`) | `T1A` igual a 0 ou 1 | 97/98 (investigados na Seção 3) e 99 |

### 2.5. Limitações conhecidas da base
* A amostra **não é autoponderada**: há sobreamostragem deliberada de UF menores e de áreas rurais. Sem pesos, as proporções descrevem a amostra, não o Brasil.
* O microdado **não traz identificadores de estrato nem de UPA**. O plano amostral só pode ser incorporado pelos pesos replicados.
* Os responsáveis **não representam** a população de pais (relatório metodológico).
* As perguntas do responsável e da criança são equivalentes, mas **não idênticas** na redação.
* Os dados são autodeclarados e estão sujeitos a viés de memória e de desejabilidade social.
""")

# =============================================================================
# 3. PREPARAÇÃO
# =============================================================================
md(r"""
## 3. Preparação e qualidade dos dados

**O que vamos fazer?** Auditar a integridade da base e entender as ausências antes de excluir qualquer observação.

**Por que isso é necessário?** Excluir casos sem investigar o motivo pode introduzir viés de seleção. Códigos especiais tratados como números distorceriam os postos dos testes.

### 3.1. Integridade: duplicados, ausentes "técnicos" e codificação
""")
code(r'''
VARS_CATEGORICAS = ["E9_G", "U6_G", "T1A", "SEXO_KIDS", "FAIXA_ETARIA", "CLASSE_2015_REC", "COD_REGIAO_2",
                    "RESP_SEXO", "D1", "D2", "M3", "IDADE_INTERNET"] + ITENS_HAB
PARES_MEDIACAO = {**{f"E10_{l}": f"U5_{l}" for l in "ABCDEFG"}, **{f"E9_{l}": f"U6_{l}" for l in "ABCDEFG"}}
VARS_CATEGORICAS += list(PARES_MEDIACAO) + list(PARES_MEDIACAO.values())

fora_do_dicionario = {v: sorted(set(dados_brutos[v].unique()) - set(codigos_do_dicionario(v)))
                      for v in VARS_CATEGORICAS if not set(dados_brutos[v].unique()) <= set(codigos_do_dicionario(v))}
auditoria = pd.DataFrame({
    "verificação": ["Linhas duplicadas", "Domicílios repetidos (ID_DOMICILIO)", "Células vazias (NaN) no arquivo",
                    "Colunas não numéricas", "Variáveis categóricas com código fora do dicionário"],
    "resultado": [
        int(dados_brutos.duplicated().sum()),
        int(dados_brutos.ID_DOMICILIO.duplicated().sum()),
        int(dados_brutos.isna().sum().sum()),
        len(dados_brutos.select_dtypes(exclude="number").columns),
        len(fora_do_dicionario),
    ]})
display(auditoria)
print("Códigos observados que o dicionário não documenta:", fora_do_dicionario)
assert auditoria.resultado.iloc[:4].eq(0).all(), "Problema de integridade: investigar antes de prosseguir"
# única exceção conhecida: o filtro IDADE_INTERNET tem o valor 0, não documentado (ver interpretação e 3.3)
assert fora_do_dicionario == {"IDADE_INTERNET": [0]}, "Código não documentado inesperado: investigar"
''')
md(r"""
**Interpretação.** O arquivo não tem duplicatas nem células vazias. Há **uma inconsistência de codificação**: o dicionário documenta apenas o código 1 da variável-filtro `IDADE_INTERNET` ("Usuários de Internet de 11 a 17 anos"), mas a base também contém o valor **0**, sem rótulo. A Seção 3.3 mostra que o 0 corresponde exatamente a "não pertence ao filtro" (não usuário ou menor de 11 anos), então o valor é tratado assim. Todos os demais códigos observados existem no dicionário. Na TIC Kids Online, a ausência **não aparece como célula vazia**: ela vem codificada (97, 98, 99, 88). Por isso, a investigação de ausências precisa olhar esses códigos.

### 3.2. Frequência dos códigos especiais
""")
code(r'''
def tabela_especiais(variaveis):
    linhas = []
    for v in variaveis:
        cont = dados_brutos[v].value_counts()
        linha = {"variável": v, "válidos": int(cont.drop(list(ESPECIAIS), errors="ignore").sum())}
        linha.update({f"{k} ({r})": int(cont.get(k, 0)) for k, r in ESPECIAIS.items()})
        linhas.append(linha)
    t = pd.DataFrame(linhas).set_index("variável")
    t["% especiais"] = (1 - t["válidos"] / len(dados_brutos)).map(fmt_pct)
    return t


tabela_especiais(["E9_G", "U6_G", "T1A", ITENS_HAB[0], ITENS_HAB[-1]])
''')
md(r"""
### 3.3. Regras de elegibilidade e saltos do questionário

**O que vamos fazer?** Testar se o código 99 ("Não se aplica") corresponde exatamente às regras de salto do questionário.

**Por que?** Se a correspondência for exata, o 99 é uma **ausência estrutural**: a pergunta não se aplica àquela criança, e não se trata de uma recusa. Excluir esses casos então não é um viés, e sim a definição da população de referência.
""")
code(r'''
b = dados_brutos
nao_usuario_seg_resp = (b.D1 == 0) | (b.D2 == 2)       # responsável: nunca usou, ou não usou nos últimos 3 meses
nao_usuario_seg_cri = b.M3 != 1                         # criança: não usou nos últimos 3 meses (ou nunca)
regras = pd.DataFrame({
    "regra": ["E9_G = 99  ⇔  responsável diz que a criança não usou a Internet nos últimos 3 meses",
              "U6_G = 99  ⇔  criança diz que não usou a Internet nos últimos 3 meses",
              "T1A = 99   ⇔  criança diz que não usou a Internet nos últimos 3 meses",
              "IDADE_INTERNET = 1  ⇔  usuária (M3 = 1) e 11 anos ou mais",
              f"Itens de habilidade = 99  ⇔  IDADE_INTERNET = 0 (todos os {len(ITENS_HAB)} itens)"],
    "vale para todos os registros?": [
        ((b.E9_G == 99) == nao_usuario_seg_resp).all(),
        ((b.U6_G == 99) == nao_usuario_seg_cri).all(),
        ((b.T1A == 99) == nao_usuario_seg_cri).all(),
        ((b.IDADE_INTERNET == 1) == ((b.M3 == 1) & (b.IDADE_KIDS >= 11))).all(),
        all(((b[c] == 99) == (b.IDADE_INTERNET == 0)).all() for c in ITENS_HAB)],
})
display(regras)
assert regras.iloc[:, 1].all()

divergentes = int((nao_usuario_seg_resp & ~nao_usuario_seg_cri).sum())
print(f"Domicílios em que o responsável diz que a criança NÃO usa a Internet, mas a criança diz que usa: {divergentes}")
print(f"Domicílios em que o responsável diz que usa, mas a criança diz que NÃO usa: "
      f"{int((~nao_usuario_seg_resp & nao_usuario_seg_cri).sum())}")
''')
md(r"""
**Interpretação.** As cinco regras valem para **todos** os registros: o código 99 é ausência estrutural, gerada pelo próprio questionário. Há uma pequena divergência entre informantes já na pergunta "a criança usa a Internet?". Esses casos ficam fora da análise pareada, porque um dos lados não respondeu ao item de mediação. Isso antecipa o tema do estudo: responsável e criança nem sempre descrevem a mesma realidade.

### 3.4. A ausência por "Não sabe"/"Não respondeu" é aleatória?

**O que vamos fazer?** Comparar quem entra na análise pareada com quem fica fora, por faixa etária, classe e área.

**Por que?** Se os excluídos têm um perfil muito diferente, as conclusões valem apenas para o subgrupo analisado. É isso que precisamos deixar explícito.
""")
code(r'''
incluido_par = sem_especiais(b.E9_G).notna() & sem_especiais(b.U6_G).notna()


def perfil(mascara, rotulo):
    return pd.concat({
        "Faixa etária": b.loc[mascara, "FAIXA_ETARIA"].map({1: "9–10", 2: "11–12", 3: "13–14", 4: "15–17"}).value_counts(normalize=True),
        "Classe": b.loc[mascara, "CLASSE_2015_REC"].map({1: "AB", 2: "C", 3: "DE"}).value_counts(normalize=True),
        "Área": b.loc[mascara, "AREA"].map({1: "Urbana", 2: "Rural"}).value_counts(normalize=True),
    }).rename(rotulo)


comparacao = pd.concat([perfil(incluido_par, f"Incluídos (n={incluido_par.sum()})"),
                        perfil(~incluido_par, f"Excluídos (n={(~incluido_par).sum()})")], axis=1)
display((comparacao * 100).round(1))

so_9798 = (~incluido_par) & ~(b.E9_G.eq(99) | b.U6_G.eq(99))
print(f"Excluídos só por 97/98 (Não sabe / Não respondeu): {so_9798.sum()} domicílios "
      f"({fmt_pct(so_9798.mean())} da base). Os demais excluídos são não usuários (99).")
''')
md(r"""
**Interpretação.** Quase todos os excluídos são **não usuários** de Internet, que estão concentrados entre os mais novos (9–10 anos) e na classe DE; a distribuição por área quase não muda. Os excluídos apenas por "Não sabe"/"Não respondeu" são poucos e não mudariam os resultados. **Consequência:** as conclusões de H1 e H2 valem para **crianças usuárias de Internet**, não para todas as crianças de 9 a 17 anos.

### 3.5. Valores extremos
As variáveis analisadas são ordinais ou binárias, com domínio fechado, então não existem "outliers" no sentido usual. O único componente contínuo é o **peso amostral**. Pesos muito desiguais aumentam a variância das estimativas ponderadas, efeito que medimos pelo **efeito de desenho de Kish** devido à ponderação: $\text{deff}_w = 1 + CV_w^2$.
""")
code(r'''
cv_peso = b.PESO.std(ddof=0) / b.PESO.mean()
deff_kish = 1 + cv_peso**2
print(f"PESO: mínimo {fmt(b.PESO.min(), 1)} | mediana {fmt(b.PESO.median(), 1)} | máximo {fmt(b.PESO.max(), 1)}")
print(f"Razão máximo/mínimo: {fmt(b.PESO.max() / b.PESO.min(), 0)}")
print(f"Efeito de desenho de Kish (ponderação): {fmt(deff_kish, 2)}  ->  tamanho efetivo ≈ {fmt(len(b) / deff_kish, 0)} de {len(b)}")
''')
md(r"""
**Interpretação.** Os pesos são muito desiguais. Para estimativas populacionais, a amostra de 2.686 crianças "vale" bem menos observações independentes, como mostra o tamanho efetivo acima. Isso justifica a camada de inferência com pesos replicados (Seção 7.6). Os pesos **não são removidos nem truncados**, porque fazem parte do desenho oficial.

### 3.6. Recodificações e bases derivadas

| Transformação | Justificativa |
|---|---|
| 88/97/98/99 → ausente | Não são pontos da escala e distorceriam os postos |
| Escalas E9, E10, U5 e U6: `6 − código` (5 = Sempre … 1 = Nunca) | Para que *maior = mais frequente*. Uma diferença positiva passa a significar que o responsável relata mais |
| Escore de habilidades = média dos itens R2A, R4 e R5 válidos (1 a 4), exigindo ≥ 80% de itens válidos | Resume 17 itens ordinais num escore com muitos níveis, reduzindo empates. O limite de 80% evita escores baseados em poucos itens |
| T1A: mantém apenas 0/1 | Desfecho binário de H4 |
""")
code(r'''
ROT_FREQ = {5: "Sempre", 4: "Quase sempre", 3: "Às vezes", 2: "Quase nunca", 1: "Nunca"}
ROT_FAIXA = {1: "9–10 anos", 2: "11–12 anos", 3: "13–14 anos", 4: "15–17 anos"}
ROT_SEXO = {1: "Masculino", 2: "Feminino"}
ROT_CLASSE = {1: "AB", 2: "C", 3: "DE"}
ROT_REGIAO = {1: "Sudeste", 2: "Nordeste", 3: "Sul", 4: "Norte", 5: "Centro-Oeste"}

base = dados_brutos[["ID_DOMICILIO", "IDADE_KIDS", "FAIXA_ETARIA", "SEXO_KIDS", "RESP_SEXO", "CLASSE_2015_REC",
                     "COD_REGIAO_2", "AREA", "IDADE_INTERNET", "PESO", *COLS_REPLICAS]].copy()
base["faixa"] = base.FAIXA_ETARIA.map(ROT_FAIXA)
base["sexo"] = base.SEXO_KIDS.map(ROT_SEXO)
base["classe"] = base.CLASSE_2015_REC.map(ROT_CLASSE)
base["regiao"] = base.COD_REGIAO_2.map(ROT_REGIAO)
for e, u in PARES_MEDIACAO.items():
    base[e], base[u] = para_frequencia(dados_brutos[e]), para_frequencia(dados_brutos[u])

itens_hab = dados_brutos[ITENS_HAB].apply(sem_especiais)
base["n_itens_hab"] = itens_hab.notna().sum(axis=1)
base["escore_hab"] = itens_hab.mean(axis=1).where(base.n_itens_hab >= 0.8 * len(ITENS_HAB))
base["T1A"] = dados_brutos.T1A.where(dados_brutos.T1A.isin([0, 1]))

# Bases analíticas (subconjuntos rastreáveis da base tratada)
pares = base.dropna(subset=["E9_G", "U6_G"]).copy()
pares["d"] = pares.E9_G - pares.U6_G
habil = base[(base.IDADE_INTERNET == 1) & base.escore_hab.notna()].copy()
ofensa = base[base.T1A.notna()].copy()

# validações
assert pares.E9_G.between(1, 5).all() and pares.U6_G.between(1, 5).all()
assert habil.escore_hab.between(1, 4).all() and set(habil.FAIXA_ETARIA) == {2, 3, 4}
assert set(ofensa.T1A.unique()) == {0, 1}
''')
code(r'''
fluxo = pd.DataFrame([
    ("Base original", len(dados_brutos), "—"),
    ("H1/H2 · pares responsável × criança", len(pares),
     "ambos usuários e com resposta válida em E9_G e U6_G"),
    ("H3 · escore de habilidades (11–17 anos)", len(habil),
     f"IDADE_INTERNET = 1 ({int((base.IDADE_INTERNET == 1).sum())}) e ≥ 80% dos itens válidos"),
    ("H4 · T1A válido", len(ofensa), "usuários com T1A = 0/1 (exclui 97/98/99)"),
], columns=["Etapa", "n", "Critério"])
fluxo["% da base"] = (fluxo.n / len(dados_brutos)).map(fmt_pct)
display(fluxo)

fig = go.Figure(go.Bar(x=fluxo.n, y=fluxo.Etapa, orientation="h", marker_color=[COR["neutro"], COR["resp"], COR["resp"], COR["resp"]],
                       text=[f"{n:,}".replace(",", ".") + f" ({p})" for n, p in zip(fluxo.n, fluxo["% da base"])],
                       textposition="auto", customdata=fluxo.Critério,
                       hovertemplate="<b>%{y}</b><br>n = %{x}<br>%{customdata}<extra></extra>"))
fig.update_layout(title="Fluxo amostral: da base original às amostras analíticas", height=330,
                  xaxis_title="Número de crianças/domicílios", yaxis=dict(autorange="reversed", title=""))
fig.show()
''')
md(r"""
**Resumo da preparação.** A base original foi preservada (`dados_brutos`). As três amostras analíticas são subconjuntos documentados da base tratada (`base`). Toda exclusão tem uma causa identificada: não uso da Internet (ausência estrutural), idade fora do bloco de perguntas ou "Não sabe"/"Não respondeu" em pequena proporção.
""")

# =============================================================================
# 4. EDA
# =============================================================================
md(r"""
## 4. Análise exploratória de dados (EDA)

> As análises desta seção são **descritivas**. Diferenças observadas aqui ainda não são evidência estatística; isso fica para a Seção 7.

### 4.1. Análise univariada

#### 4.1.1. Composição da amostra e representatividade
* **Objetivo:** ver como a amostra se distribui e quanto ela difere da população estimada.
* **Metodologia:** proporções sem peso (amostra) e com `PESO` (população), por região.
""")
code(r'''
def proporcoes(col, rotulos_ordem):
    sem = base[col].value_counts(normalize=True).reindex(rotulos_ordem)
    com = base.groupby(col).PESO.sum().reindex(rotulos_ordem) / base.PESO.sum()
    return pd.DataFrame({"Amostra (sem peso)": sem, "População estimada (com peso)": com})


comp_regiao = proporcoes("regiao", list(ROT_REGIAO.values()))
display((comp_regiao * 100).round(1))

fig = go.Figure()
for nome, cor in zip(comp_regiao.columns, [COR["neutro"], COR["resp"]]):
    fig.add_bar(name=nome, x=comp_regiao.index, y=comp_regiao[nome], marker_color=cor,
                text=comp_regiao[nome].map(fmt_pct), textposition="outside",
                hovertemplate="%{x}<br>" + nome + ": %{y:.1%}<extra></extra>")
fig.update_layout(title="Distribuição por região: amostra × população estimada", barmode="group",
                  yaxis=dict(tickformat=".0%", title="Proporção"), xaxis_title="Região", height=420)
fig.show()
''')
code(r'''
mostrar(f"""
* **Resultado:** o Sudeste tem {fmt_pct(comp_regiao.loc['Sudeste', 'Amostra (sem peso)'])} da amostra,
  mas {fmt_pct(comp_regiao.loc['Sudeste', 'População estimada (com peso)'])} da população estimada.
  O Norte tem {fmt_pct(comp_regiao.loc['Norte', 'Amostra (sem peso)'])} da amostra e
  {fmt_pct(comp_regiao.loc['Norte', 'População estimada (com peso)'])} da população.
* **Interpretação:** a amostra foi **desenhada** para não ser proporcional, com sobreamostragem de UF menores. Esse é o principal motivo para comparar resultados sem e com pesos.
* **Limitações:** a comparação mostra apenas uma dimensão; os pesos também corrigem idade, sexo, área e não resposta.
""")
''')
md(r"""
#### 4.1.2. As duas variáveis centrais: E9_G (responsável) e U6_G (criança)
* **Objetivo:** descrever cada relato separadamente.
* **Metodologia:** frequências absolutas e relativas, mediana, quartis e moda, na amostra pareada.
""")
code(r'''
def resumo_ordinal(serie, rotulos):
    cont = serie.value_counts().reindex(sorted(rotulos, reverse=True), fill_value=0)
    t = pd.DataFrame({"n": cont, "%": (cont / cont.sum() * 100).round(1)})
    t.index = [rotulos[k] for k in t.index]
    q1, med, q3 = serie.quantile([0.25, 0.5, 0.75])
    est = {"mediana": rotulos[int(med)], "Q1–Q3": f"{rotulos[int(q1)]} – {rotulos[int(q3)]}",
           "moda": rotulos[int(serie.mode()[0])], "assimetria": fmt(stats.skew(serie), 2)}
    return t, est


t_resp, e_resp = resumo_ordinal(pares.E9_G, ROT_FREQ)
t_cri, e_cri = resumo_ordinal(pares.U6_G, ROT_FREQ)
display(pd.concat({"Responsável (E9_G)": t_resp, "Criança (U6_G)": t_cri}, axis=1))
display(pd.DataFrame({"Responsável (E9_G)": e_resp, "Criança (U6_G)": e_cri}))

fig = go.Figure()
for col, nome, cor in [("E9_G", "Responsável (E9_G)", COR["resp"]), ("U6_G", "Criança (U6_G)", COR["cri"])]:
    prop = pares[col].value_counts(normalize=True).reindex([5, 4, 3, 2, 1])
    cont = pares[col].value_counts().reindex([5, 4, 3, 2, 1])
    fig.add_bar(name=nome, x=[ROT_FREQ[k] for k in prop.index], y=prop, marker_color=cor,
                text=prop.map(fmt_pct), textposition="outside", customdata=cont,
                hovertemplate="%{x}<br>" + nome + ": %{y:.1%} (n = %{customdata})<extra></extra>")
fig.update_layout(title=f"Com que frequência o responsável conversa sobre o que a criança faz na Internet? (n = {len(pares)} pares)",
                  barmode="group", yaxis=dict(tickformat=".0%", title="Proporção dos pares", range=[0, 0.75]),
                  xaxis_title="Resposta", height=450)
fig.show()
''')
code(r'''
mostrar(f"""
* **Resultado:** "Sempre" é a resposta de {fmt_pct((pares.E9_G == 5).mean())} dos responsáveis e de
  {fmt_pct((pares.U6_G == 5).mean())} das crianças. A mediana é **{e_resp['mediana']}** no relato do responsável e
  **{e_cri['mediana']}** no da criança. A distribuição do responsável é fortemente concentrada no topo
  (assimetria {e_resp['assimetria']}); a da criança é mais espalhada.
* **Interpretação:** à primeira vista, o responsável descreve uma mediação mais frequente.
* **Limitações:** as distribuições **marginais** não dizem se, *dentro do mesmo domicílio*, os dois discordam. Isso exige olhar os pares (4.2).
""")
''')
md(r"""
#### 4.1.3. Escore de habilidades digitais (11–17 anos) e T1A
* **Objetivo:** descrever o desfecho de H3 e o de H4.
* **Metodologia:** medidas de posição e dispersão do escore; frequência de T1A.
""")
code(r'''
desc_hab = habil.escore_hab.describe()
print("Escore de habilidades (1 a 4):")
print(f"  n = {int(desc_hab['count'])} | média = {fmt(desc_hab['mean'])} | DP = {fmt(desc_hab['std'])} | "
      f"mediana = {fmt(desc_hab['50%'])} | Q1–Q3 = {fmt(desc_hab['25%'])}–{fmt(desc_hab['75%'])} | "
      f"assimetria = {fmt(stats.skew(habil.escore_hab))}")
print(f"  valores distintos do escore: {habil.escore_hab.nunique()}")
print(f"\nT1A (tratado de forma ofensiva na Internet, 12 meses): n = {len(ofensa)} | Sim = {int(ofensa.T1A.sum())} "
      f"({fmt_pct(ofensa.T1A.mean())})")

fig = go.Figure(go.Histogram(x=habil.escore_hab, xbins=dict(start=1, end=4.01, size=0.125), marker_color=COR["resp"],
                             hovertemplate="Escore %{x}<br>%{y} adolescentes<extra></extra>"))
fig.add_vline(x=habil.escore_hab.median(), line_dash="dash", line_color=COR["destaque"],
              annotation_text=f"mediana = {fmt(habil.escore_hab.median())}", annotation_position="top left")
fig.update_layout(title="Distribuição do escore de habilidades digitais (11–17 anos)", height=400,
                  xaxis_title="Escore médio (1 = Não é verdade … 4 = É muito verdade)", yaxis_title="Adolescentes", bargap=0.05)
fig.show()
''')
md(r"""
* **Resultado e interpretação:** o escore tem leve assimetria negativa. A metade central fica aproximadamente entre 2,2 e 2,9, ou seja, entre "É um pouco verdade" e "É verdade", e há uma cauda de adolescentes com escores baixos. O escore tem muitos valores distintos, o que reduz os empates no teste de postos. O resultado de T1A aparece na saída acima: uma minoria relata ter sido tratada de forma ofensiva.
* **Limitações:** a média de itens ordinais é uma **aproximação**, porque supõe distâncias iguais entre as categorias. Ela é usada aqui apenas como escore de ordenação, e o teste de H3 usa somente os postos.

### 4.2. Análise bivariada

#### 4.2.1. Tabela cruzada responsável × criança
* **Objetivo:** ver a concordância dentro de cada domicílio.
* **Metodologia:** tabela de contingência 5 × 5 com percentuais do total. A diagonal reúne os pares concordantes.
""")
code(r'''
ordem = [5, 4, 3, 2, 1]
tab_rc = pd.crosstab(pares.E9_G, pares.U6_G).reindex(index=ordem, columns=ordem, fill_value=0)
pct_rc = tab_rc / tab_rc.values.sum()
rot = [ROT_FREQ[k] for k in ordem]

fig = go.Figure(go.Heatmap(
    z=pct_rc.values, x=rot, y=rot, colorscale="Blues", colorbar=dict(title="% dos pares", tickformat=".0%"),
    text=[[f"{tab_rc.values[i, j]}<br>{fmt_pct(pct_rc.values[i, j])}" for j in range(5)] for i in range(5)],
    texttemplate="%{text}", hovertemplate="Responsável: %{y}<br>Criança: %{x}<br>%{z:.1%} dos pares<extra></extra>"))
for i in range(5):
    fig.add_shape(type="rect", x0=i - 0.5, x1=i + 0.5, y0=i - 0.5, y1=i + 0.5, fillcolor="rgba(0,0,0,0)",
                  line=dict(color=COR["destaque"], width=3))
fig.update_layout(title="Responsável × criança: concordância no mesmo domicílio (diagonal em vermelho)", height=520,
                  margin=dict(l=120),
                  xaxis_title="Relato da criança (U6_G)", yaxis=dict(title="Relato do responsável (E9_G)", autorange="reversed"))
fig.show()

concordancia = pd.Series({"Responsável relata MAIS (acima da diagonal)": (pares.d > 0).mean(),
                          "Concordam (diagonal)": (pares.d == 0).mean(),
                          "Responsável relata MENOS (abaixo da diagonal)": (pares.d < 0).mean()})
display(concordancia.map(fmt_pct).to_frame("% dos pares"))
pesos_kappa = np.subtract.outer(range(5), range(5)) ** 2                      # penaliza discordâncias distantes
observado = pct_rc.values
esperado = np.outer(observado.sum(axis=1), observado.sum(axis=0))              # sob independência
kappa_pond = 1 - (pesos_kappa * observado).sum() / (pesos_kappa * esperado).sum()
print(f"Kappa ponderado quadrático (concordância além do acaso): {fmt(kappa_pond, 2)}")
''')
code(r'''
mostrar(f"""
* **Resultado:** {fmt_pct(concordancia.iloc[0])} dos pares ficam acima da diagonal (responsável relata mais),
  {fmt_pct(concordancia.iloc[1])} concordam e {fmt_pct(concordancia.iloc[2])} ficam abaixo. A célula mais frequente
  é "Sempre × Sempre", mas há muitos casos em que o responsável diz "Sempre" e a criança diz "Às vezes" ou "Nunca".
  O kappa ponderado é {fmt(kappa_pond, 2)}.
* **Interpretação:** a discordância é **assimétrica**: quando os dois discordam, o responsável tende a relatar mais. A concordância além do acaso é apenas *razoável*: na escala de Landis & Koch (1977), um kappa entre 0,21 e 0,40 recebe essa classificação.
* **Limitações:** a assimetria ainda é descritiva; a Seção 7 testa se ela supera o que o acaso produziria.
""")
''')
md(r"""
#### 4.2.2. Habilidades por faixa etária e por classe social
* **Objetivo:** examinar a relação de H3 e um possível confundidor (classe).
* **Metodologia:** boxplots por grupo; tabela faixa × classe para ver se a composição social muda com a idade.
""")
code(r'''
fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=("Por faixa etária", "Por classe social"))
for i, (g, d) in enumerate(habil.groupby("faixa")):
    fig.add_box(y=d.escore_hab, name=g, marker_color=COR["seq"][i + 1], boxmean=False, row=1, col=1,
                hovertemplate=g + "<br>escore %{y}<extra></extra>")
for i, c in enumerate(["AB", "C", "DE"]):
    d = habil[habil.classe == c]
    fig.add_box(y=d.escore_hab, name=c, marker_color=[COR["resp"], COR["neutro"], COR["cri"]][i], row=1, col=2)
fig.update_layout(title="Escore de habilidades digitais por faixa etária e por classe social", showlegend=False, height=460,
                  yaxis_title="Escore (1 a 4)")
fig.show()

forma = habil.groupby("faixa").escore_hab.agg(n="size", mediana="median",
                                              IIQ=lambda x: x.quantile(.75) - x.quantile(.25),
                                              assimetria=lambda x: stats.skew(x)).round(2)
display(forma)
comp_classe = pd.crosstab(habil.faixa, habil.classe, normalize="index")
display((comp_classe * 100).round(1).rename_axis(index="faixa", columns="classe (% na faixa)"))
''')
code(r'''
mostrar(f"""
* **Resultado:** a mediana do escore sobe de {fmt(forma.mediana.iloc[0])} (11–12 anos) para {fmt(forma.mediana.iloc[1])}
  (13–14) e {fmt(forma.mediana.iloc[2])} (15–17), e a **dispersão diminui** (IIQ de {fmt(forma.IIQ.iloc[0])} para
  {fmt(forma.IIQ.iloc[2])}). Entre classes, as diferenças de mediana são menores. A composição por classe é parecida
  nas três faixas etárias.
* **Interpretação:** há um gradiente claro com a idade. Como a composição social não varia entre as faixas, a classe
  **não parece explicar** esse gradiente (isso é verificado formalmente na análise estratificada em 7.3).
* **Limitações:** como as formas das distribuições diferem entre as faixas (dispersões diferentes), o Kruskal-Wallis
  deve ser lido como teste de **dominância estocástica**, e não estritamente como comparação de medianas.
""")
''')
md(r"""
#### 4.2.3. T1A por sexo
* **Objetivo:** explorar a relação de H4.
* **Metodologia:** tabela 2 × 2 com proporções condicionais e IC de 95% de Wilson.
""")
code(r'''
from statsmodels.stats.proportion import proportion_confint

tab_of = pd.crosstab(ofensa.sexo, ofensa.T1A).rename(columns={0: "Não", 1: "Sim"})
prop_of = ofensa.groupby("sexo").T1A.agg(["sum", "size"])
prop_of["prop"] = prop_of["sum"] / prop_of["size"]
prop_of[["ic_inf", "ic_sup"]] = np.array([proportion_confint(s, n, method="wilson") for s, n in zip(prop_of["sum"], prop_of["size"])])
display(tab_of)

fig = go.Figure(go.Bar(
    x=prop_of.index, y=prop_of.prop, marker_color=[COR["cri"], COR["resp"]],
    error_y=dict(type="data", symmetric=False, array=prop_of.ic_sup - prop_of.prop, arrayminus=prop_of.prop - prop_of.ic_inf),
    text=prop_of.prop.map(fmt_pct), textposition="outside", customdata=np.c_[prop_of["sum"], prop_of["size"]],
    hovertemplate="%{x}<br>%{y:.1%} (%{customdata[0]} de %{customdata[1]})<extra></extra>"))
fig.update_layout(title="Proporção tratada de forma ofensiva na Internet (12 meses), por sexo · IC 95% de Wilson",
                  yaxis=dict(tickformat=".0%", title="Proporção", range=[0, 0.25]), xaxis_title="", height=420)
fig.show()
mostrar(f"""
* **Resultado:** {fmt_pct(prop_of.prop['Feminino'])} das meninas e {fmt_pct(prop_of.prop['Masculino'])} dos meninos responderam "Sim".
  Os intervalos de confiança se sobrepõem amplamente.
* **Interpretação:** descritivamente, as proporções são muito próximas.
* **Limitações:** a sobreposição de intervalos não é, sozinha, um teste formal; T1A também tem
  {fmt_pct(dados_brutos.T1A.isin([97, 98]).sum() / (dados_brutos.T1A != 99).sum())} de "Não sabe"/"Não respondeu" entre os usuários (item sensível).
""")
''')
md(r"""
### 4.3. Análise orientada às hipóteses: a discordância depende de quem é a criança?

* **Objetivo:** ver se a direção da discordância responsável × criança se mantém em subgrupos ou se é explicada por algum deles (possível confundimento).
* **Metodologia:** proporção de pares em que o responsável relata mais, igual ou menos, por faixa etária, sexo da criança e sexo do responsável. Os percentuais estão empilhados.
""")
code(r'''
def composicao_discordancia(col, ordem_grupos):
    t = pares.assign(cat=np.select([pares.d > 0, pares.d == 0], ["Responsável relata mais", "Concordam"], "Responsável relata menos"))
    return pd.crosstab(t[col], t.cat, normalize="index").reindex(ordem_grupos)[
        ["Responsável relata mais", "Concordam", "Responsável relata menos"]]


pares["sexo_resp"] = pares.RESP_SEXO.map({1: "Resp. masculino", 2: "Resp. feminino"})
blocos = [("faixa", list(ROT_FAIXA.values())), ("sexo", ["Masculino", "Feminino"]), ("sexo_resp", ["Resp. masculino", "Resp. feminino"])]
comp = pd.concat([composicao_discordancia(c, o) for c, o in blocos])
ns = pd.concat([pares[c].value_counts().reindex(o) for c, o in blocos])

fig = go.Figure()
for cat, cor in zip(comp.columns, [COR["resp"], COR["neutro"], COR["cri"]]):
    fig.add_bar(name=cat, y=[f"{g} (n={ns[g]})" for g in comp.index], x=comp[cat], orientation="h", marker_color=cor,
                text=comp[cat].map(lambda v: fmt_pct(v, 0)), textposition="inside",
                hovertemplate="%{y}<br>" + cat + ": %{x:.1%}<extra></extra>")
fig.update_layout(title="Direção da discordância responsável × criança, por subgrupo", barmode="stack", height=480,
                  legend=dict(traceorder="normal"),
                  xaxis=dict(tickformat=".0%", title="Proporção dos pares"), yaxis=dict(autorange="reversed", title=""))
fig.show()
display((comp * 100).round(1))
''')
code(r'''
mais_min, mais_max = comp["Responsável relata mais"].min(), comp["Responsável relata mais"].max()
menos_faixa = comp.loc[list(ROT_FAIXA.values()), "Responsável relata menos"]
mostrar(f"""
* **Resultado:** em **todos** os subgrupos, a proporção em que o responsável relata mais
  ({fmt_pct(mais_min)} a {fmt_pct(mais_max)}) supera a proporção em que relata menos. Com a idade, cresce a
  proporção em que o responsável relata **menos** ({fmt_pct(menos_faixa.iloc[0])} aos 9–10 anos e
  {fmt_pct(menos_faixa.iloc[-1])} aos 15–17), o que reduz a assimetria entre os adolescentes mais velhos.
* **Interpretação:** a direção da discordância não parece ser um artefato de uma faixa etária ou de um sexo específicos.
  Ela aparece em todos os subgrupos examinados. A idade parece modular a intensidade, e esse padrão é exploratório.
* **Limitações:** são comparações descritivas, sem teste. Outros fatores não observados (por exemplo, quem do domicílio
  respondeu e a qualidade da relação familiar) podem influenciar os relatos.
""")
''')

# =============================================================================
# 5. HIPÓTESES
# =============================================================================
md(r"""
## 5. Formulação das hipóteses estatísticas

Nível de significância em todos os testes: $\alpha = 0{,}05$, bilateral. As quatro hipóteses principais formam uma família, e seus p-valores são ajustados por Holm (Seção 7.5).

> **Transparência:** as hipóteses foram definidas depois de uma exploração preliminar desta mesma base, sem pré-registro. Os testes devem ser lidos como uma confirmação interna e consistente, não como um teste confirmatório independente.

### H1: concordância ordinal responsável × criança
* **Pergunta:** P1.
* **Justificativa:** se o relato do responsável descreve bem a mediação percebida pela criança, as diferenças devem se distribuir simetricamente em torno de zero.
* **Variáveis:** `E9_G` e `U6_G`, recodificadas como $X_i$ (responsável) e $Y_i$ (criança), com 1 = Nunca … 5 = Sempre. Escala ordinal.
* **Estrutura:** **pareada**, com os dois informantes do mesmo domicílio $i$. Diferença $D_i = X_i - Y_i$.
* **Teste candidato:** postos sinalizados de Wilcoxon.

$$H_0:\ D \text{ tem distribuição simétrica em torno de } 0 \qquad\qquad H_1:\ D \text{ não é simétrica em torno de } 0$$

### H2: concordância binária ("Sempre")
* **Variáveis:** $A_i = \mathbb{1}[X_i = \text{Sempre}]$ e $B_i = \mathbb{1}[Y_i = \text{Sempre}]$, ambas binárias e **pareadas**.
* **Teste candidato:** McNemar.

$$H_0:\ \pi_{\text{resp}} = \pi_{\text{cri}} \;\;(\Leftrightarrow\ \pi_{10} = \pi_{01}) \qquad\qquad H_1:\ \pi_{\text{resp}} \neq \pi_{\text{cri}}$$

em que $\pi_{10}$ é a probabilidade de só o responsável dizer "Sempre" e $\pi_{01}$, a de só a criança.

### H3: habilidades digitais × faixa etária
* **Pergunta:** P2. **Justificativa:** as habilidades digitais tendem a se desenvolver com a experiência e a idade.
* **Variáveis:** escore de habilidades $S$ (ordinal, de 1 a 4) e faixa etária $G \in \{11\text{–}12,\ 13\text{–}14,\ 15\text{–}17\}$.
* **Estrutura:** **3 grupos independentes**, um adolescente por domicílio.
* **Teste candidato:** Kruskal-Wallis, seguido de Dunn com correção de Holm.

$$H_0:\ F_{11\text{–}12} = F_{13\text{–}14} = F_{15\text{–}17} \qquad\qquad H_1:\ \exists\, j \neq k:\ P(S_j > S_k) \neq \tfrac12$$

### H4: experiência ofensiva × sexo
* **Pergunta:** P3. **Justificativa:** a literatura sobre riscos online discute diferenças por gênero na exposição a conteúdos e contatos ofensivos.
* **Variáveis:** T1A (binária: Sim/Não) e sexo da criança (nominal binária).
* **Estrutura:** **2 grupos independentes**.
* **Teste candidato:** qui-quadrado de independência, com o teste exato de Fisher como verificação.

$$H_0:\ \pi_{\text{fem}} = \pi_{\text{masc}} \qquad\qquad H_1:\ \pi_{\text{fem}} \neq \pi_{\text{masc}}$$
""")

# =============================================================================
# 6. SELEÇÃO DOS TESTES
# =============================================================================
md(r"""
## 6. Seleção e justificativa dos testes estatísticos

A escolha decorre do **tipo de variável** e da **estrutura de dependência** de cada hipótese, como resume a tabela abaixo. Ela não depende dos p-valores.

| | Pareado | Independente |
|---|---|---|
| **Ordinal** | Wilcoxon pareado (H1) | Kruskal-Wallis (H3) |
| **Binário** | McNemar (H2) | Qui-quadrado / Fisher (H4) |

### 6.1. Wilcoxon pareado (H1)
1. **Por que é adequado?** Os dados são pareados e ordinais; o teste *t* pareado exigiria uma escala intervalar com diferenças aproximadamente normais.
2. **Pressupostos:** (i) pares independentes entre si; (ii) as diferenças $D_i$ podem ser ordenadas em magnitude; (iii) sob $H_0$, $D$ é simétrica em torno de 0.
3. **Verificação:** (i) há uma criança por domicílio, mas os domicílios estão agrupados em UPA (ver 6.5). (ii) Esse é o ponto mais delicado: ordenar $|D|$ supõe que "Sempre − Às vezes" (2) é maior que "Sempre − Quase sempre" (1). É razoável numa escala de frequência, mas é uma suposição. Por isso usamos como **sensibilidade** o **teste do sinal**, que só usa o sinal de $D$, e o teste de **homogeneidade marginal de Stuart-Maxwell**, que só usa a tabela 5 × 5.
4. **Estatística:** $W^+ = \sum_{i:\,D_i>0} R_i$, em que $R_i$ é o posto de $|D_i|$ entre as diferenças não nulas.
5. **p-valor:** aproximação normal, adequada para $n$ grande, com correção de empates:
$$z = \frac{W^+ - \frac{n(n+1)}{4}}{\sqrt{\frac{n(n+1)(2n+1)}{24} - \frac{\sum_g (t_g^3 - t_g)}{48}}}, \qquad p = 2\,[1-\Phi(|z|)]$$
6. **Decisão:** rejeitar $H_0$ se $p < \alpha$.
7. **Interpretação:** $W^+ > W^-$ indica que o responsável tende a relatar mais.
8. **Efeito e IC:** correlação bisserial de postos $r_{rb} = (W^+ - W^-)/(W^+ + W^-)$ (Kerby, 2014), com IC bootstrap, e $r = z/\sqrt{N}$. O estimador de Hodges-Lehmann não é informativo aqui: com diferenças inteiras de −4 a 4, ele cai num valor inteiro com IC degenerado.

### 6.2. McNemar (H2)
1. **Adequação:** a resposta é binária e pareada.
2. **Pressupostos:** pares independentes; $b + c$ suficientemente grande para a aproximação qui-quadrado (caso contrário, usa-se a versão exata binomial).
3. **Verificação:** $b + c$ é calculado abaixo; a versão exata também é reportada.
4. **Estatística:** $\chi^2 = \dfrac{(|b-c|-1)^2}{b+c}$, em que $b$ = só o responsável diz "Sempre" e $c$ = só a criança diz.
5. **p-valor:** $\chi^2_1$; versão exata: $B \sim \text{Bin}(b+c,\ 1/2)$.
6. **Decisão:** rejeitar $H_0$ se $p < \alpha$.
7. **Interpretação:** $b > c$ indica que o "Sempre" é mais frequente no relato do responsável.
8. **Efeito e IC:** diferença de proporções pareadas $\hat\pi_{\text{resp}} - \hat\pi_{\text{cri}} = (b-c)/n$, com IC de Wald pareado, e razão de chances pareada $b/c$, com IC $\exp\!\left(\ln\frac{b}{c} \pm 1{,}96\sqrt{\frac1b + \frac1c}\right)$.

### 6.3. Kruskal-Wallis + Dunn (H3)
1. **Adequação:** escore ordinal com 3 grupos independentes; é a alternativa à ANOVA de um fator, sem supor normalidade nem variâncias iguais.
2. **Pressupostos:** observações independentes; variável pelo menos ordinal. Para interpretar o resultado como diferença de **medianas**, as distribuições precisariam ter a mesma forma.
3. **Verificação:** a EDA (4.2.2) mostrou dispersões diferentes, então interpretamos o teste como **dominância estocástica**: a probabilidade de um adolescente de um grupo ter escore maior que um de outro grupo.
4. **Estatística:**
$$H = \frac{12}{N(N+1)}\sum_{j=1}^{k} n_j\,\bar R_j^{\,2} - 3(N+1), \quad \text{dividida pelo fator de correção de empates } 1 - \frac{\sum (t^3-t)}{N^3-N}$$
5. **p-valor:** $H \sim \chi^2_{k-1}$ sob $H_0$.
6. **Decisão:** rejeitar $H_0$ se $p < \alpha$, e então aplicar Dunn com Holm para localizar as diferenças.
7. **Interpretação:** pelo menos uma faixa tende a ter escores maiores.
8. **Efeito e IC:** $\varepsilon^2 = H/(N-1)$ (Tomczak & Tomczak, 2014), com IC bootstrap.

### 6.4. Qui-quadrado de independência e Fisher (H4)
1. **Adequação:** duas variáveis binárias, com grupos independentes.
2. **Pressupostos:** observações independentes; contagens esperadas ≥ 5 em todas as células.
3. **Verificação:** as contagens esperadas são calculadas abaixo. O teste exato de Fisher, que não depende dessa condição, é reportado como verificação.
4. **Estatística:** $\chi^2 = \sum \frac{(O-E)^2}{E}$, com 1 grau de liberdade.
5. **p-valor:** $\chi^2_1$; Fisher: distribuição hipergeométrica.
6. **Decisão:** rejeitar $H_0$ se $p < \alpha$.
7. **Interpretação:** um resultado não significativo leva a **não rejeitar** $H_0$, o que não prova que as proporções sejam iguais.
8. **Efeito e IC:** diferença de proporções (feminino − masculino) com IC de Newcombe, e $\varphi$ (V de Cramér para tabelas 2 × 2).

### 6.5. E o plano amostral complexo?
Os testes acima supõem **observações independentes e identicamente distribuídas**. A TIC Kids Online usa estratos, conglomerados (UPA) e pesos desiguais. Consequências:
* **Pesos:** sem eles, as estimativas descrevem a amostra, não a população.
* **Conglomerados:** domicílios da mesma UPA tendem a ser parecidos, e ignorar isso costuma **subestimar** os erros-padrão, tornando os testes liberais.

Não existe uma versão "por desenho" consagrada para todos os testes de postos. A estratégia adotada tem **duas camadas**:
1. **Camada clássica (não paramétrica):** os testes da Seção 6, sem pesos. Eles respondem se o padrão é sistemático na amostra e servem ao objetivo didático.
2. **Camada populacional (por desenho):** para cada hipótese, estimamos uma quantidade análoga, com `PESO`, e o erro-padrão pelos **200 pesos replicados** (bootstrap de Rao-Wu, que incorpora estratos e conglomerados). Daí saem um IC de 95% e um teste de Wald.

| Hipótese | Quantidade estimada por desenho |
|---|---|
| H1 | $\Delta = P(X>Y) - P(X<Y)$, a versão populacional do teste do sinal |
| H2 | $\pi_{\text{resp}} - \pi_{\text{cri}}$ (proporção de "Sempre") |
| H3 | Médias ponderadas do escore por faixa e teste de Wald conjunto com 2 g.l. |
| H4 | $\pi_{\text{fem}} - \pi_{\text{masc}}$ |

Se as duas camadas concordarem, a conclusão é robusta ao plano amostral.
""")

# =============================================================================
# 7. EXECUÇÃO
# =============================================================================
md(r"""
## 7. Execução dos testes de hipóteses

### 7.1. H1: Wilcoxon pareado

**A · Contextualização.** Testamos se, no mesmo domicílio, o responsável relata conversar sobre a Internet com a mesma frequência que a criança percebe.

**B · Fundamentação.** Ver 6.1. Em resumo: (1) calcular $D_i$ e descartar $D_i = 0$; (2) atribuir postos a $|D_i|$, usando postos médios nos empates; (3) somar os postos positivos ($W^+$) e negativos ($W^-$); (4) padronizar $W^+$ com a variância corrigida por empates.

**C · Preparação.** São usados os pares da Seção 3 (`pares`). Os empates são descartados pelo próprio procedimento.
""")
code(r'''
# D · Implementação (manual, conferida com scipy.stats.wilcoxon dentro da função)
h1 = wilcoxon_pareado(pares.E9_G, pares.U6_G)
d_array = pares.d.to_numpy()
h1["rbc_ic"] = ic_bootstrap(correlacao_bisserial_pareada, d_array)
h1["r_ic"] = ic_bootstrap(lambda d: wilcoxon_pareado(d, np.zeros_like(d))["r"], d_array, B=500)

# Sensibilidade: teste do sinal (só usa o sinal) e Stuart-Maxwell (homogeneidade marginal da tabela 5x5)
n_pos, n_neg = int((d_array > 0).sum()), int((d_array < 0).sum())
sinal = stats.binomtest(n_pos, n_pos + n_neg, 0.5)
stuart = SquareTable(tab_rc.values, shift_zeros=False).homogeneity()

# E · Resultados
res_h1 = pd.Series({
    "Pares analisados (N)": h1["n_pares"], "Empates descartados": h1["n_empates"], "Diferenças não nulas (n)": h1["n_util"],
    "W+": fmt(h1["W_pos"], 1), "W−": fmt(h1["W_neg"], 1), "E[W+] sob H0": fmt(h1["media_W"], 1),
    "z": fmt(h1["z"]), "p-valor": fmt_p(h1["p"]), "α": fmt(ALPHA),
    "Correlação bisserial de postos [IC 95% bootstrap]": f"{fmt(h1['rbc'])} [{fmt(h1['rbc_ic'][0])}; {fmt(h1['rbc_ic'][1])}]",
    "r = z/√N [IC 95% bootstrap]": f"{fmt(h1['r'])} [{fmt(h1['r_ic'][0])}; {fmt(h1['r_ic'][1])}]",
    "Decisão": "Rejeitar H0" if h1["p"] < ALPHA else "Não rejeitar H0",
    "Sensibilidade · teste do sinal": f"{n_pos} positivos × {n_neg} negativos, {fmt_p(sinal.pvalue)}",
    "Sensibilidade · Stuart-Maxwell": f"χ² = {fmt(stuart.statistic, 1)}, g.l. = {int(stuart.df)}, {fmt_p(stuart.pvalue)}",
})
display(res_h1.to_frame("H1 · Wilcoxon pareado"))
''')
code(r'''
fig = go.Figure()
cont_d = pares.d.value_counts().sort_index()
fig.add_bar(x=cont_d.index, y=cont_d.values,
            marker_color=[COR["cri"] if v < 0 else COR["neutro"] if v == 0 else COR["resp"] for v in cont_d.index],
            text=cont_d.values, textposition="outside",
            hovertemplate="d = %{x}<br>%{y} pares<extra></extra>")
fig.add_annotation(x=2.5, y=cont_d.max() * 0.95, showarrow=False, align="left",
                   text=f"W+ = {fmt(h1['W_pos'], 0)}<br>W− = {fmt(h1['W_neg'], 0)}<br>z = {fmt(h1['z'])}, {fmt_p(h1['p'])}<br>"
                        f"r<sub>rb</sub> = {fmt(h1['rbc'])} [{fmt(h1['rbc_ic'][0])}; {fmt(h1['rbc_ic'][1])}]",
                   bordercolor="#999", borderwidth=1, bgcolor="white")
fig.update_layout(title="H1 · Distribuição de D = responsável − criança (positivo = responsável relata mais)", height=440,
                  xaxis=dict(title="D (diferença em níveis da escala)", dtick=1), yaxis_title="Número de pares")
fig.show()
''')
code(r'''
# F · Interpretação (gerada a partir dos resultados)
mostrar(f"""
**F · Interpretação.** Entre os {h1['n_util']} pares com relatos diferentes, a soma dos postos em que o responsável relata
**mais** ($W^+ = {fmt(h1['W_pos'], 0)}$) é muito maior que a soma dos postos em que relata menos ($W^- = {fmt(h1['W_neg'], 0)}$),
com z = {fmt(h1['z'])} e {fmt_p(h1['p'])}. **Rejeitamos $H_0$** ao nível de 5%: as diferenças não são simétricas em torno de zero.
O responsável tende a relatar uma frequência de conversa **maior** do que a percebida pela criança.

O tamanho de efeito ($r_{{rb}} = {fmt(h1['rbc'])}$, IC 95% [{fmt(h1['rbc_ic'][0])}; {fmt(h1['rbc_ic'][1])}];
$r = {fmt(h1['r'])}$) indica um efeito **moderado a grande**, então o resultado não se deve apenas ao tamanho da amostra.
O teste do sinal e o de Stuart-Maxwell, que não supõem que as distâncias entre as categorias sejam comparáveis, chegam à mesma
conclusão. Portanto, o resultado não depende dessa suposição do Wilcoxon.

*Cuidado:* o p-valor **não** é a probabilidade de $H_0$ ser verdadeira. Ele é a probabilidade de observar uma assimetria
pelo menos tão extrema quanto esta se $H_0$ fosse verdadeira. O teste também não diz **quem está certo**, apenas que os relatos divergem
sistematicamente.
""")
''')

md(r"""
### 7.2. H2: McNemar ("Sempre" × não "Sempre")

**A · Contextualização.** Simplificamos a pergunta para "o responsável conversa *sempre*?" e comparamos a proporção de "Sempre" nos dois relatos. É uma leitura mais intuitiva, adequada a uma apresentação.

**B · Fundamentação.** Ver 6.2. Os pares concordantes ($a$, $d$) não informam sobre a diferença; só os discordantes ($b$, $c$) contam.

**C · Preparação.** Usamos os mesmos pares de H1, dicotomizados.
""")
code(r'''
# D · Implementação
A = (pares.E9_G == 5).to_numpy()
B = (pares.U6_G == 5).to_numpy()
tab_mc = pd.crosstab(pd.Series(A, name="Responsável: Sempre"), pd.Series(B, name="Criança: Sempre")).reindex(
    index=[True, False], columns=[True, False])
a_, b_, c_, d_ = tab_mc.loc[True, True], tab_mc.loc[True, False], tab_mc.loc[False, True], tab_mc.loc[False, False]
n2 = len(pares)
mc_assint = mcnemar(tab_mc.values, exact=False, correction=True)
mc_exato = mcnemar(tab_mc.values, exact=True)
dif_prop = (b_ - c_) / n2
ep_dif = np.sqrt((b_ + c_) - (b_ - c_) ** 2 / n2) / n2
or_par = b_ / c_
ic_or = np.exp(np.log(or_par) + np.array([-1, 1]) * 1.96 * np.sqrt(1 / b_ + 1 / c_))

display(tab_mc)
res_h2 = pd.Series({
    "n (pares)": n2, "b (só o responsável diz Sempre)": b_, "c (só a criança diz Sempre)": c_,
    "b + c (pares discordantes)": b_ + c_,
    "χ² de McNemar (com correção de continuidade)": fmt(mc_assint.statistic, 1),
    "p-valor (assintótico)": fmt_p(mc_assint.pvalue), "p-valor (exato binomial)": fmt_p(mc_exato.pvalue),
    "% Sempre · responsável": fmt_pct(A.mean()), "% Sempre · criança": fmt_pct(B.mean()),
    "Diferença de proporções [IC 95%]": f"{fmt(100 * dif_prop, 1)} p.p. [{fmt(100 * (dif_prop - 1.96 * ep_dif), 1)}; {fmt(100 * (dif_prop + 1.96 * ep_dif), 1)}]",
    "Razão de chances pareada b/c [IC 95%]": f"{fmt(or_par)} [{fmt(ic_or[0])}; {fmt(ic_or[1])}]",
    "Decisão": "Rejeitar H0" if mc_assint.pvalue < ALPHA else "Não rejeitar H0",
})
display(res_h2.to_frame("H2 · McNemar"))
''')
code(r'''
mostrar(f"""
**E/F · Interpretação.** Com $b + c = {b_ + c_}$ pares discordantes (bem acima do necessário para a aproximação), há
**{b_}** domicílios em que só o responsável diz "Sempre" contra **{c_}** em que só a criança diz (χ² = {fmt(mc_assint.statistic, 1)},
{fmt_p(mc_assint.pvalue)}; o teste exato concorda). **Rejeitamos $H_0$.** A proporção de "Sempre" é
{fmt(100 * dif_prop, 1)} pontos percentuais maior no relato do responsável (IC 95%: {fmt(100 * (dif_prop - 1.96 * ep_dif), 1)} a
{fmt(100 * (dif_prop + 1.96 * ep_dif), 1)} p.p.). Entre os pares discordantes, a configuração "só o responsável diz Sempre" é
cerca de **{fmt(or_par, 1)} vezes** mais comum que a inversa. O resultado confirma H1 com uma métrica mais fácil de comunicar,
mas que perde a informação das categorias intermediárias.
""")
''')

md(r"""
### 7.3. H3: Kruskal-Wallis e teste de Dunn

**A · Contextualização.** Testamos se as habilidades digitais autodeclaradas diferem entre as faixas etárias de 11–12, 13–14 e 15–17 anos.

**B · Fundamentação.** Ver 6.3. Todos os escores são ordenados juntos; se os grupos viessem da mesma distribuição, os postos médios seriam parecidos.

**C · Preparação.** `habil`: usuários de 11 a 17 anos com pelo menos 80% dos itens válidos.
""")
code(r'''
# D · Implementação
grupos_h3 = [g.escore_hab.to_numpy() for _, g in habil.groupby("faixa")]
nomes_h3 = [n for n, _ in habil.groupby("faixa")]
kw = stats.kruskal(*grupos_h3)
N3 = len(habil)
eps2 = epsilon_quadrado(kw.statistic, N3)
eps2_ic = ic_bootstrap(lambda v, g: epsilon_quadrado(stats.kruskal(*[v[g == k] for k in np.unique(g)]).statistic, len(v)),
                       habil.escore_hab.to_numpy(), habil.FAIXA_ETARIA.to_numpy(), B=1000)
postos_medios = pd.Series(stats.rankdata(habil.escore_hab), index=habil.index).groupby(habil.faixa).mean()

res_h3 = pd.Series({
    "N": N3, "Tamanho dos grupos": ", ".join(f"{n}: {len(g)}" for n, g in zip(nomes_h3, grupos_h3)),
    "Postos médios": ", ".join(f"{k}: {fmt(v, 0)}" for k, v in postos_medios.items()),
    "H (corrigido para empates)": fmt(kw.statistic, 1), "g.l.": len(grupos_h3) - 1, "p-valor": fmt_p(kw.pvalue),
    "ε² [IC 95% bootstrap]": f"{fmt(eps2, 3)} [{fmt(eps2_ic[0], 3)}; {fmt(eps2_ic[1], 3)}]",
    "Decisão": "Rejeitar H0" if kw.pvalue < ALPHA else "Não rejeitar H0",
})
display(res_h3.to_frame("H3 · Kruskal-Wallis"))

dunn = teste_dunn(habil.escore_hab, habil.faixa)
# probabilidade de superioridade P(S_j > S_k) + 0,5 P(empate) para cada par, como medida de efeito par a par
def prob_superioridade(x, y):
    u = stats.mannwhitneyu(x, y, alternative="two-sided").statistic
    return u / (len(x) * len(y))
dunn["P(maior que o 2º grupo)"] = [prob_superioridade(habil.escore_hab[habil.faixa == c.split(" × ")[0]],
                                                      habil.escore_hab[habil.faixa == c.split(" × ")[1]]) for c in dunn.comparacao]
display(dunn.style.format({"posto_medio_1": "{:.0f}", "posto_medio_2": "{:.0f}", "z": "{:.2f}",
                           "p": "{:.1e}", "p_holm": "{:.1e}", "P(maior que o 2º grupo)": "{:.2f}"}).hide(axis="index"))
''')
code(r'''
# Verificação de confundimento: o gradiente etário persiste dentro de cada classe social?
estrat = []
for c in ["AB", "C", "DE"]:
    sub = habil[habil.classe == c]
    k = stats.kruskal(*[g.escore_hab for _, g in sub.groupby("faixa")])
    estrat.append(dict(classe=c, n=len(sub), H=k.statistic, p=k.pvalue, eps2=epsilon_quadrado(k.statistic, len(sub)),
                       medianas=" / ".join(fmt(m) for m in sub.groupby("faixa").escore_hab.median())))
estrat = pd.DataFrame(estrat)
display(estrat.style.format({"H": "{:.1f}", "p": "{:.1e}", "eps2": "{:.3f}"}).hide(axis="index"))
rho = stats.spearmanr(habil.IDADE_KIDS, habil.escore_hab)
print(f"Spearman idade (anos) × escore: rho = {fmt(rho.statistic)} ({fmt_p(rho.pvalue)})")
''')
code(r'''
sig = dunn[dunn.p_holm < ALPHA].comparacao.tolist()
nsig = dunn[dunn.p_holm >= ALPHA].comparacao.tolist()
mostrar(f"""
**E/F · Interpretação.** O Kruskal-Wallis rejeita $H_0$ (H = {fmt(kw.statistic, 1)}, g.l. = 2, {fmt_p(kw.pvalue)}): as faixas
etárias **não** vêm da mesma distribuição de habilidades. O efeito é relevante (ε² = {fmt(eps2, 3)}, IC 95%
[{fmt(eps2_ic[0], 3)}; {fmt(eps2_ic[1], 3)}]); por convenção, valores entre 0,04 e 0,16 são considerados moderados e
valores a partir de 0,16, relativamente fortes.
No teste de Dunn com correção de Holm, diferem significativamente: {', '.join(sig) if sig else 'nenhum par'}{('; não diferem: ' + ', '.join(nsig)) if nsig else ''}.
A medida par a par mostra, por exemplo, que, sorteando um adolescente de 15–17 anos e um de 11–12, a probabilidade de o
mais velho ter escore **maior** é de {fmt(1 - dunn.loc[dunn.comparacao.str.contains('11–12 anos × 15–17'), 'P(maior que o 2º grupo)'].iloc[0])}
(0,5 significaria nenhuma diferença).

O gradiente se mantém **dentro de cada classe social** (todos os Kruskal-Wallis estratificados com p < 0,001), então a classe
não explica a diferença entre as faixas. Como as dispersões diferem entre os grupos (4.2.2), lemos o resultado como
**dominância estocástica**: os mais velhos tendem a relatar mais habilidades. Não lemos como uma simples diferença de medianas.
Por ser um estudo transversal, também não podemos afirmar que é *a idade* que causa o aumento: coortes diferentes podem ter
trajetórias digitais diferentes.
""")
''')

md(r"""
### 7.4. H4: qui-quadrado de independência (T1A × sexo)

**A · Contextualização.** Testamos se ter sido tratado de forma ofensiva na Internet nos últimos 12 meses é igualmente frequente entre meninos e meninas usuários.

**B · Fundamentação.** Ver 6.4. Comparamos as contagens observadas com as esperadas sob independência, $E_{ij} = \frac{n_{i\cdot}\,n_{\cdot j}}{n}$.

**C · Preparação.** `ofensa`: usuários com T1A = 0/1. Antes, verificamos se a não resposta (97/98) difere entre os sexos, porque um item sensível pode ter recusa seletiva.
""")
code(r'''
usuarios = dados_brutos.T1A != 99
nr_sexo = pd.crosstab(dados_brutos.loc[usuarios, "SEXO_KIDS"].map(ROT_SEXO), dados_brutos.loc[usuarios, "T1A"].isin([97, 98]))
chi_nr = stats.chi2_contingency(nr_sexo)
print("Não resposta (97/98) em T1A por sexo:",
      {s: fmt_pct(nr_sexo.loc[s, True] / nr_sexo.loc[s].sum()) for s in nr_sexo.index}, "|", fmt_p(chi_nr.pvalue))

# D · Implementação
tab_h4 = pd.crosstab(ofensa.sexo, ofensa.T1A).reindex(index=["Feminino", "Masculino"], columns=[1, 0])
chi = stats.chi2_contingency(tab_h4, correction=False)
fisher = stats.fisher_exact(tab_h4.values)
esperadas = pd.DataFrame(chi.expected_freq, index=tab_h4.index, columns=["Sim (esperado)", "Não (esperado)"])
phi = np.sqrt(chi.statistic / tab_h4.values.sum())

p_f, n_f = tab_h4.loc["Feminino", 1] / tab_h4.loc["Feminino"].sum(), tab_h4.loc["Feminino"].sum()
p_m, n_m = tab_h4.loc["Masculino", 1] / tab_h4.loc["Masculino"].sum(), tab_h4.loc["Masculino"].sum()
from statsmodels.stats.proportion import confint_proportions_2indep
ic_rd = confint_proportions_2indep(tab_h4.loc["Feminino", 1], n_f, tab_h4.loc["Masculino", 1], n_m, method="newcomb")

display(pd.concat([tab_h4.rename(columns={1: "Sim", 0: "Não"}), esperadas.round(1)], axis=1))
res_h4 = pd.Series({
    "n": int(tab_h4.values.sum()), "Menor contagem esperada": fmt(chi.expected_freq.min(), 1),
    "χ² (g.l. = 1, sem correção)": fmt(chi.statistic, 2), "p-valor (χ²)": fmt_p(chi.pvalue),
    "p-valor (Fisher exato)": fmt_p(fisher.pvalue),
    "Proporção feminino × masculino": f"{fmt_pct(p_f)} × {fmt_pct(p_m)}",
    "Diferença (fem − masc) [IC 95% Newcombe]": f"{fmt(100 * (p_f - p_m), 1)} p.p. [{fmt(100 * ic_rd[0], 1)}; {fmt(100 * ic_rd[1], 1)}]",
    "φ (V de Cramér)": fmt(phi, 3),
    "Decisão": "Rejeitar H0" if chi.pvalue < ALPHA else "Não rejeitar H0",
})
display(res_h4.to_frame("H4 · Qui-quadrado"))
''')
code(r'''
mostrar(f"""
**E/F · Interpretação.** Todas as contagens esperadas são muito maiores que 5, então a aproximação do qui-quadrado é adequada.
O resultado é χ² = {fmt(chi.statistic, 2)}, {fmt_p(chi.pvalue)}, e o Fisher concorda ({fmt_p(fisher.pvalue)}).
**Não rejeitamos $H_0$**: os dados não mostram evidência de que a proporção de crianças tratadas de forma ofensiva difira entre
meninas ({fmt_pct(p_f)}) e meninos ({fmt_pct(p_m)}).

Isso **não prova** que as proporções sejam iguais. O IC 95% da diferença ({fmt(100 * ic_rd[0], 1)} a {fmt(100 * ic_rd[1], 1)} p.p.)
mostra que diferenças de até cerca de {fmt(max(abs(100 * ic_rd[0]), abs(100 * ic_rd[1])), 0)} p.p. em qualquer direção são
compatíveis com os dados. O que se pode dizer é que, se existe diferença, ela provavelmente é pequena. A não resposta ao item
não difere entre os sexos ({fmt_p(chi_nr.pvalue)}), o que reduz a preocupação com recusa seletiva. T1A também mistura
tipos muito diferentes de experiência ofensiva, e meninos e meninas podem viver tipos diferentes com a mesma frequência total.
""")
''')

md(r"""
### 7.5. Correção para múltiplas comparações

**Por que?** Com 4 testes a 5%, a chance de pelo menos um falso positivo, se todas as $H_0$ forem verdadeiras, chega a cerca de $1-0{,}95^4 \approx 18,5\%$. O método de **Holm** controla a taxa de erro familiar (FWER) em 5% e é uniformemente mais poderoso que o de Bonferroni.
""")
code(r'''
familia = pd.DataFrame({
    "hipótese": ["H1 · Wilcoxon pareado", "H2 · McNemar", "H3 · Kruskal-Wallis", "H4 · Qui-quadrado"],
    "p bruto": [h1["p"], mc_assint.pvalue, kw.pvalue, chi.pvalue],
})
familia["p ajustado (Holm)"] = multipletests(familia["p bruto"], method="holm")[1]
familia["decisão (α = 0,05)"] = np.where(familia["p ajustado (Holm)"] < ALPHA, "Rejeitar H0", "Não rejeitar H0")
display(familia.style.format({"p bruto": "{:.2e}", "p ajustado (Holm)": "{:.2e}"}).hide(axis="index"))
''')
md(r"""
**Interpretação.** O ajuste não muda nenhuma decisão: H1, H2 e H3 continuam significativas, com p-valores muito pequenos, e H4 continua não significativa. A análise de robustez a seguir é uma família à parte, com seu próprio ajuste de Holm.

### 7.6. Robustez de H1: os 14 itens de mediação
**O que vamos fazer?** Repetir o Wilcoxon pareado em todos os 14 pares de itens perguntados aos dois informantes, com ajuste de Holm e IC bootstrap para o tamanho de efeito.

**Por que?** Isso mostra se o resultado de H1 é específico do item "conversa" ou se é um padrão geral da mediação.
""")
code(r'''
ROT_ITENS = {
    "E10_A": "Senta junto e participa", "E10_B": "Fica por perto sem olhar", "E10_C": "Incentiva a aprender sozinho",
    "E10_D": "Senta junto sem participar", "E10_E": "Olha o celular da criança", "E10_F": "Põe regras para o celular",
    "E10_G": "Deixa sem celular por um tempo", "E9_A": "Ajuda quando não entende", "E9_B": "Explica sites bons e ruins",
    "E9_C": "Ensina uso seguro", "E9_D": "Ensina comportamento online", "E9_E": "Ajuda quando algo incomoda",
    "E9_F": "Explica o que fazer se algo incomodar", "E9_G": "Conversa sobre o que faz na Internet",
}
linhas = []
for e, u in PARES_MEDIACAO.items():
    sub = base.dropna(subset=[e, u])
    res = wilcoxon_pareado(sub[e], sub[u])
    dd = (sub[e] - sub[u]).to_numpy()
    ic = ic_bootstrap(correlacao_bisserial_pareada, dd, B=1000)
    linhas.append(dict(item=f"{e} × {u}", descricao=ROT_ITENS[e], tipo="Orientação (E9/U6)" if e.startswith("E9") else "Acompanhamento e regras (E10/U5)",
                       n=res["n_pares"], resp_mais=(dd > 0).mean(), resp_menos=(dd < 0).mean(),
                       rbc=res["rbc"], ic_inf=ic[0], ic_sup=ic[1], p=res["p"]))
robustez = pd.DataFrame(linhas)
robustez["p_holm"] = multipletests(robustez.p, method="holm")[1]
robustez = robustez.sort_values("rbc").reset_index(drop=True)
display(robustez.style.format({"resp_mais": "{:.0%}", "resp_menos": "{:.0%}", "rbc": "{:.2f}", "ic_inf": "{:.2f}",
                               "ic_sup": "{:.2f}", "p": "{:.1e}", "p_holm": "{:.1e}"}).hide(axis="index"))
''')
code(r'''
mostrar(f"""
**Interpretação.** Em **{(robustez.rbc > 0).sum()} de 14** itens, o responsável relata mais mediação que a criança
({(robustez.p_holm < ALPHA).sum()} significativos após Holm). O efeito vai de $r_{{rb}}$ = {fmt(robustez.rbc.min())}
("{robustez.descricao.iloc[0]}") a {fmt(robustez.rbc.max())} ("{robustez.descricao.iloc[-1]}"). Os maiores efeitos
estão nos itens de **orientação verbal e regras**, e o item "{robustez.descricao.iloc[0]}" está perto de zero. H1 não é
um acaso de um item isolado: a assimetria é um traço geral do relato sobre mediação, mais forte quando a prática é
uma conversa ou orientação, que a criança pode não registrar como tal.
""")
''')

md(r"""
### 7.7. Camada populacional: estimativas com os pesos replicados

**O que vamos fazer?** Reestimar cada hipótese com `PESO` e obter erros-padrão pelos 200 pesos replicados (ver 6.5).

**Por que?** Para saber se as conclusões valem para a **população** de crianças usuárias, levando em conta estratos, conglomerados e pesos desiguais.

**Como?** Para uma estimativa $\hat\theta$, calculamos $\hat\theta_r$ com cada réplica $r = 1, \dots, 200$:
$$\widehat{\mathrm{Var}}(\hat\theta) = \frac{1}{R-1}\sum_{r=1}^{R}\left(\hat\theta_r - \bar{\hat\theta}\right)^2, \qquad z = \hat\theta \big/ \sqrt{\widehat{\mathrm{Var}}(\hat\theta)}.$$
""")
code(r'''
sinal_d = np.sign(pares.d.to_numpy())
pop_h1 = estimativa_com_replicas(lambda w: media_ponderada(sinal_d, w[pares.index]), base)
dif_sempre = (pares.E9_G.to_numpy() == 5).astype(float) - (pares.U6_G.to_numpy() == 5)
pop_h2 = estimativa_com_replicas(lambda w: media_ponderada(dif_sempre, w[pares.index]), base)

y_of, fem = ofensa.T1A.to_numpy(), (ofensa.SEXO_KIDS == 2).to_numpy()
pop_h4 = estimativa_com_replicas(lambda w: media_ponderada(y_of, w[ofensa.index], fem) - media_ponderada(y_of, w[ofensa.index], ~fem), base)

# H3: médias ponderadas por faixa + Wald conjunto (2 contrastes) com a matriz de covariância das réplicas
esc, fx = habil.escore_hab.to_numpy(), habil.FAIXA_ETARIA.to_numpy()
medias_faixa = lambda w: np.array([media_ponderada(esc, w[habil.index], fx == k) for k in (2, 3, 4)])
theta_h3 = medias_faixa(base.PESO.to_numpy())
reps_h3 = np.array([medias_faixa(base[c].to_numpy()) for c in COLS_REPLICAS])
C = np.array([[1, -1, 0], [0, 1, -1]])
cov_contr = np.cov((reps_h3 @ C.T).T, ddof=1)
wald_h3 = float((C @ theta_h3) @ np.linalg.solve(cov_contr, C @ theta_h3))
p_wald_h3 = stats.chi2.sf(wald_h3, 2)
ep_h3 = reps_h3.std(axis=0, ddof=1)
dif_extremos = theta_h3[2] - theta_h3[0]
ep_extremos = (reps_h3[:, 2] - reps_h3[:, 0]).std(ddof=1)
dif_13_15 = theta_h3[2] - theta_h3[1]
ep_13_15 = (reps_h3[:, 2] - reps_h3[:, 1]).std(ddof=1)

camada_pop = pd.DataFrame([
    ("H1", "Δ = P(resp > cri) − P(resp < cri)", (sinal_d > 0).mean() - (sinal_d < 0).mean(), pop_h1["estimativa"], pop_h1["ic_inf"], pop_h1["ic_sup"], pop_h1["p"]),
    ("H2", "% Sempre: responsável − criança", dif_sempre.mean(), pop_h2["estimativa"], pop_h2["ic_inf"], pop_h2["ic_sup"], pop_h2["p"]),
    ("H3", "Escore médio: 15–17 − 11–12 anos", esc[fx == 4].mean() - esc[fx == 2].mean(), dif_extremos,
     dif_extremos - 1.96 * ep_extremos, dif_extremos + 1.96 * ep_extremos, 2 * stats.norm.sf(abs(dif_extremos / ep_extremos))),
    ("H3", "Escore médio: 15–17 − 13–14 anos", esc[fx == 4].mean() - esc[fx == 3].mean(), dif_13_15,
     dif_13_15 - 1.96 * ep_13_15, dif_13_15 + 1.96 * ep_13_15, 2 * stats.norm.sf(abs(dif_13_15 / ep_13_15))),
    ("H4", "% ofendido: feminino − masculino", p_f - p_m, pop_h4["estimativa"], pop_h4["ic_inf"], pop_h4["ic_sup"], pop_h4["p"]),
], columns=["hipótese", "quantidade", "sem peso", "com peso", "IC95% inf (desenho)", "IC95% sup (desenho)", "p (Wald, desenho)"])
display(camada_pop.style.format({"sem peso": "{:.3f}", "com peso": "{:.3f}", "IC95% inf (desenho)": "{:.3f}",
                                 "IC95% sup (desenho)": "{:.3f}", "p (Wald, desenho)": "{:.1e}"}).hide(axis="index"))
print(f"H3 · Wald conjunto (igualdade das 3 médias ponderadas): X² = {fmt(wald_h3, 1)}, g.l. = 2, {fmt_p(p_wald_h3)}")
print("Médias ponderadas do escore por faixa:", {ROT_FAIXA[k]: fmt(m) for k, m in zip((2, 3, 4), theta_h3)})
''')
code(r'''
mostrar(f"""
**Interpretação.**
* **H1 e H2 se mantêm na população.** Com pesos, $\\Delta$ = {fmt(pop_h1['estimativa'])} (IC 95% {fmt(pop_h1['ic_inf'])} a
  {fmt(pop_h1['ic_sup'])}) e a diferença de "Sempre" é de {fmt(100 * pop_h2['estimativa'], 1)} p.p. (IC 95%
  {fmt(100 * pop_h2['ic_inf'], 1)} a {fmt(100 * pop_h2['ic_sup'], 1)}). Os intervalos são **mais largos** que na camada
  clássica, por causa do efeito dos conglomerados e dos pesos, mas ficam longe de zero.
* **H3 se mantém, com uma nuance.** O teste conjunto por desenho rejeita a igualdade ({fmt_p(p_wald_h3)}), e a diferença entre
  os extremos (15–17 − 11–12) é clara. Já a diferença **entre 13–14 e 15–17 anos encolhe com os pesos**
  ({fmt(dif_13_15, 2)}, {fmt_p(camada_pop.iloc[3, -1])}). A separação entre as duas faixas mais velhas é, portanto, menos
  robusta que a separação em relação aos mais novos.
* **H4 se mantém não significativa:** diferença ponderada de {fmt(100 * pop_h4['estimativa'], 1)} p.p. (IC 95%
  {fmt(100 * pop_h4['ic_inf'], 1)} a {fmt(100 * pop_h4['ic_sup'], 1)}).
* **Limitação:** a camada populacional estima quantidades **análogas**, como o teste do sinal e as médias, e não os mesmos
  testes de postos. Ela confirma a direção e a relevância dos achados; não substitui os testes clássicos.
""")
''')

# =============================================================================
# 8. VISUALIZAÇÃO DOS RESULTADOS
# =============================================================================
md(r"""
## 8. Visualização e comunicação dos resultados

### 8.1. Tamanho de efeito por item de mediação (robustez de H1)
**Objetivo:** mostrar num único gráfico a direção, a magnitude e a incerteza da discordância em todos os itens. Cada ponto é a correlação bisserial de postos; as barras são o IC de 95% bootstrap. Valores à direita de zero indicam que o responsável relata mais.
""")
code(r'''
fig = go.Figure()
for tipo, cor in [("Acompanhamento e regras (E10/U5)", COR["cri"]), ("Orientação (E9/U6)", COR["resp"])]:
    t = robustez[robustez.tipo == tipo]
    fig.add_trace(go.Scatter(
        x=t.rbc, y=t.descricao, mode="markers", name=tipo, marker=dict(size=11, color=cor),
        error_x=dict(type="data", symmetric=False, array=t.ic_sup - t.rbc, arrayminus=t.rbc - t.ic_inf, thickness=2),
        customdata=np.c_[t.item, t.n, t.resp_mais * 100, t.resp_menos * 100, t.p_holm],
        hovertemplate="<b>%{y}</b> (%{customdata[0]})<br>r<sub>rb</sub> = %{x:.2f}<br>n = %{customdata[1]}"
                      "<br>Resp. relata mais: %{customdata[2]:.0f}% · menos: %{customdata[3]:.0f}%"
                      "<br>p (Holm) = %{customdata[4]:.1e}<extra></extra>"))
fig.add_vline(x=0, line_color="#555")
for x, txt in [(0.1, "pequeno"), (0.3, "moderado"), (0.5, "grande")]:
    fig.add_vline(x=x, line_dash="dot", line_color="#bbb", annotation_text=txt, annotation_position="top")
fig.update_layout(title="Responsável × criança: quem relata mais mediação? Correlação bisserial de postos (IC 95%)",
                  height=600, legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0),
                  margin=dict(b=130), xaxis=dict(title="r<sub>rb</sub>  (← criança relata mais | responsável relata mais →)", range=[-0.2, 0.75]),
                  yaxis=dict(title="", categoryorder="array", categoryarray=robustez.descricao.tolist()))
fig.show()
''')
md(r"""
**Interpretação.** Todos os pontos estão à direita de zero, e todos os intervalos de confiança também. Os itens de orientação verbal e de regras e controle ("Explica o que fazer…", "Conversa…", "Explica sites…", "Põe regras…") têm efeitos de moderados a grandes. Já a ajuda prática ("Ajuda quando não entende") difere pouco entre os dois relatos ($r_{rb} \approx 0{,}09$). Uma leitura possível é que práticas concretas e pontuais são lembradas da mesma forma pelos dois; práticas difusas, como "conversar", são mais sujeitas a percepções diferentes e à desejabilidade social do adulto.

### 8.2. Painel-síntese: amostra × população
**Objetivo:** comparar as estimativas sem peso e com peso, com o IC por desenho, para as quantidades de cada hipótese.
""")
code(r'''
fig = make_subplots(rows=1, cols=4, subplot_titles=["H1 · Δ (sinal)", "H2 · dif. % Sempre", "H3 · escore 15–17 − 11–12", "H4 · dif. % ofendido (F − M)"],
                    horizontal_spacing=0.07)
painel = camada_pop.drop(index=3).reset_index(drop=True)
for i, linha in painel.iterrows():
    fig.add_trace(go.Scatter(x=["sem peso"], y=[linha["sem peso"]], mode="markers", marker=dict(size=12, color=COR["neutro"]),
                             name="Amostra (sem peso)", showlegend=i == 0,
                             hovertemplate="sem peso: %{y:.3f}<extra></extra>"), row=1, col=i + 1)
    fig.add_trace(go.Scatter(x=["com peso"], y=[linha["com peso"]], mode="markers", marker=dict(size=12, color=COR["resp"]),
                             error_y=dict(type="data", symmetric=False, array=[linha["IC95% sup (desenho)"] - linha["com peso"]],
                                          arrayminus=[linha["com peso"] - linha["IC95% inf (desenho)"]]),
                             name="População (com peso, IC 95% por desenho)", showlegend=i == 0,
                             hovertemplate="com peso: %{y:.3f}<extra></extra>"), row=1, col=i + 1)
    fig.add_hline(y=0, line_dash="dot", line_color="#999", row=1, col=i + 1)
fig.update_yaxes(rangemode="tozero")
fig.update_layout(title="Estimativas por hipótese: amostra × população (linha pontilhada = ausência de efeito)", height=470,
                  margin=dict(t=110, b=110), legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0))
fig.show()
''')
md(r"""
**Interpretação.** Em H1, H2 e H3, as estimativas com e sem peso são próximas, e o IC por desenho fica bem longe de zero: as conclusões não dependem de ignorar o plano amostral. Em H4, os dois pontos ficam praticamente sobre o zero, e o intervalo cruza a linha, o que é coerente com não rejeitar $H_0$. O gráfico mostra o **tamanho** e a **incerteza** de cada efeito, e não apenas o p-valor.
""")

# =============================================================================
# 9. DISCUSSÃO
# =============================================================================
md(r"""
## 9. Discussão

### 9.1. Síntese dos achados
1. **Responsáveis relatam mais mediação do que as crianças percebem (H1 e H2).** No item "conversa sobre o que faz na Internet", a assimetria é forte (Wilcoxon e McNemar, p < 0,001), com efeito de moderado a grande. Ela se mantém no teste do sinal, no Stuart-Maxwell, em todos os subgrupos examinados, nos 14 itens de mediação (com intensidade variável) e com os pesos amostrais.
2. **As habilidades digitais aumentam entre as faixas etárias (H3).** O efeito é relevante (ε² ≈ 0,16, no limite entre moderado e relativamente forte) e se mantém dentro de cada classe social. A diferença robusta é entre os adolescentes de 11–12 anos e os mais velhos. Com pesos, a diferença entre 13–14 e 15–17 anos praticamente **desaparece**, embora o Dunn sem pesos a aponte como significativa.
3. **Não há evidência de diferença por sexo em ter sido tratado de forma ofensiva (H4)**, tanto na amostra quanto na população. Diferenças pequenas, de alguns pontos percentuais, não podem ser descartadas.

### 9.2. Significância estatística × relevância prática
Com milhares de observações, efeitos minúsculos já ficam "significativos". Por isso reportamos sempre o efeito com seu IC. Em H1, uma diferença de 26,5 p.p. na proporção de "Sempre" (cerca de 30 p.p. com pesos) tem relevância prática: indicadores de mediação baseados só no adulto podem superestimar substancialmente a orientação percebida pela criança. Já na EDA, diferenças de habilidade entre classes sociais dariam p < 0,001 com efeito pequeno, o que mostra que um p-valor baixo sozinho não basta.

### 9.3. Explicações possíveis (hipóteses, não conclusões)
* **Desejabilidade social:** o adulto pode tender a relatar o comportamento "esperado" de um bom responsável.
* **Diferença de percepção:** o que o adulto chama de "conversar sobre a Internet" pode não ser registrado pela criança como orientação.
* **Referenciais diferentes:** o adulto pode pensar em todas as crianças do domicílio ou em vários responsáveis; a criança pergunta sobre "pais ou responsáveis" em geral.
* **Modo de coleta:** seções de autopreenchimento e entrevista face a face podem produzir níveis diferentes de franqueza. Não é possível separar esse efeito com estes dados.

O padrão com a idade, em que a proporção de responsáveis que relatam *menos* cresce entre os adolescentes, sugere que a natureza da mediação muda ao longo da adolescência. Esse é um achado exploratório.

### 9.4. Limitações
* **Desenho observacional e transversal:** nenhuma das associações é causal. Em H3, a idade está confundida com a coorte.
* **Plano amostral:** os testes de postos clássicos ignoram estratos, conglomerados e pesos. A camada por desenho mitiga isso com quantidades análogas, mas não existe aqui um "Wilcoxon por desenho" exato (há propostas na literatura, como Lumley & Scott, 2013, que não foram implementadas).
* **Bootstrap dos ICs de efeito:** é i.i.d. e não reproduz os conglomerados, então os ICs dos tamanhos de efeito tendem a ser estreitos demais. Os ICs da Seção 7.7 corrigem isso para as quantidades por desenho.
* **Mensuração:** escalas autodeclaradas, redações não idênticas entre os questionários e um escore de habilidades que é a média de itens ordinais.
* **Seleção da população analisada:** H1 e H2 valem para usuários de Internet segundo os dois informantes; H3, para usuários de 11 a 17 anos.
* **Hipóteses definidas após exploração preliminar** da mesma base, sem pré-registro.

### 9.5. Generalização e pesquisas futuras
As conclusões sobre a direção da discordância valem para crianças usuárias de Internet no Brasil em 2024, apoiadas pela camada por desenho. Sugestões:
* replicar a análise em outras edições da pesquisa (2023 e 2025) para testar a estabilidade do padrão;
* modelar a discordância com regressão ordinal ou de concordância, incluindo idade, sexo, escolaridade do responsável e outros fatores;
* investigar qualitativamente o que pais e filhos entendem por "conversar sobre a Internet";
* implementar testes de postos para amostras complexas.
""")

# =============================================================================
# 10. CONCLUSÃO
# =============================================================================
md(r"""
## 10. Conclusão
""")
code(r'''
mostrar(f"""
**P1. Responsáveis e crianças relatam a mesma frequência de conversa sobre a Internet?** **Não.** Os relatos divergem de forma
sistemática: o responsável relata frequência maior (Wilcoxon pareado: z = {fmt(h1['z'])}, {fmt_p(h1['p'])},
$r_{{rb}}$ = {fmt(h1['rbc'])}). A proporção de "Sempre" é {fmt(100 * dif_prop, 1)} p.p. maior no relato do adulto
(McNemar, {fmt_p(mc_assint.pvalue)}). O padrão se repete, com intensidade variável, nos 14 itens de mediação e com os pesos amostrais.

**P2. As habilidades digitais diferem entre faixas etárias?** **Sim.** Elas tendem a ser maiores nas faixas mais velhas
(Kruskal-Wallis: H = {fmt(kw.statistic, 1)}, {fmt_p(kw.pvalue)}, ε² = {fmt(eps2, 3)}). A diferença mais robusta é entre os
adolescentes de 11–12 anos e os mais velhos.

**P3. A proporção de crianças tratadas de forma ofensiva difere entre meninos e meninas?** **Não encontramos evidência
de diferença** (χ² = {fmt(chi.statistic, 2)}, {fmt_p(chi.pvalue)}; diferença de {fmt(100 * (p_f - p_m), 1)} p.p., IC 95%
{fmt(100 * ic_rd[0], 1)} a {fmt(100 * ic_rd[1], 1)}). Isso não demonstra igualdade.

**Contribuição.** O estudo mostra, com dados oficiais e testes não paramétricos adequados a dados ordinais e pareados, que a
mediação parental **depende de quem informa**. Pesquisas e políticas que se baseiam apenas no relato dos responsáveis tendem
a superestimar a orientação percebida pelas crianças.

**Ressalvas que devem acompanhar os resultados:** delineamento transversal (sem causalidade), uso de testes clássicos numa
amostra complexa (mitigado pela camada por desenho), dados autodeclarados e hipóteses definidas após uma exploração
preliminar da mesma base.
""")
''')

# =============================================================================
# 11. REFERÊNCIAS E REPRODUTIBILIDADE
# =============================================================================
md(r"""
## 11. Referências e reprodutibilidade

### 11.1. Dados e documentação
* Cetic.br/NIC.br. *Pesquisa sobre o uso da Internet por crianças no Brasil: TIC Kids Online Brasil 2024*. Base de microdados v1.2, dicionário de variáveis v1.1, relatório metodológico e relatório de coleta de dados. São Paulo: Comitê Gestor da Internet no Brasil. Disponível em: https://cetic.br/pt/pesquisa/kids-online/microdados/

### 11.2. Métodos estatísticos
* Wilcoxon, F. (1945). Individual comparisons by ranking methods. *Biometrics Bulletin*, 1(6), 80–83.
* McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika*, 12(2), 153–157.
* Kruskal, W. H., & Wallis, W. A. (1952). Use of ranks in one-criterion variance analysis. *Journal of the American Statistical Association*, 47(260), 583–621.
* Dunn, O. J. (1964). Multiple comparisons using rank sums. *Technometrics*, 6(3), 241–252.
* Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70.
* Stuart, A. (1955). A test for homogeneity of the marginal distributions in a two-way classification. *Biometrika*, 42(3/4), 412–416.
* Maxwell, A. E. (1970). Comparing the classification of subjects by two independent judges. *British Journal of Psychiatry*, 116, 651–655.
* Kerby, D. S. (2014). The simple difference formula: an approach to teaching nonparametric correlation. *Comprehensive Psychology*, 3, 11.IT.3.1.
* Tomczak, M., & Tomczak, E. (2014). The need to report effect size estimates revisited. *Trends in Sport Sciences*, 21(1), 19–25.
* Newcombe, R. G. (1998). Interval estimation for the difference between independent proportions. *Statistics in Medicine*, 17(8), 873–890.
* Conover, W. J. (1999). *Practical Nonparametric Statistics* (3ª ed.). Wiley.
* Hollander, M., Wolfe, D. A., & Chicken, E. (2014). *Nonparametric Statistical Methods* (3ª ed.). Wiley.

### 11.3. Amostras complexas
* Rao, J. N. K., & Wu, C. F. J. (1988). Resampling inference with complex survey data. *Journal of the American Statistical Association*, 83(401), 231–241.
* Lumley, T. (2010). *Complex Surveys: A Guide to Analysis Using R*. Wiley.
* Lumley, T., & Scott, A. J. (2013). Two-sample rank tests under complex sampling. *Biometrika*, 100(4), 831–842.
* Opsomer, J. D., & Erciulescu, A. L. (2021). Replication variance estimation after sample-based calibration. *Survey Methodology*, 47(2), 265–277.
* Kish, L. (1965). *Survey Sampling*. Wiley.
* Landis, J. R., & Koch, G. G. (1977). The measurement of observer agreement for categorical data. *Biometrics*, 33(1), 159–174.

### 11.4. Tema
* Livingstone, S., & Helsper, E. J. (2008). Parental mediation of children's internet use. *Journal of Broadcasting & Electronic Media*, 52(4), 581–599.

### 11.5. Ambiente de execução
""")
code(r'''
versoes = pd.Series({"Python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                     "scipy": scipy.__version__, "statsmodels": statsmodels.__version__, "plotly": plotly.__version__,
                     "Semente (bootstrap)": SEED, "Reamostragens bootstrap (H1)": B_BOOT,
                     "Arquivo de dados": ARQ_DADOS.name, "Registros lidos": len(dados_brutos)})
versoes.to_frame("valor")
''')
md(r"""
### 11.6. Como executar
1. Instale as dependências, a partir da pasta `PROJ 2`:
   ```bash
   pip install -r requirements.txt
   ```
2. Abra o notebook e execute **todas as células em ordem** (*Run All*). Na linha de comando:
   ```bash
   jupyter nbconvert --to notebook --execute --inplace notebooks/02_tic_kids_wilcoxon_pareado.ipynb
   ```
3. Se `data/tic_kids_online_2024/` não existir, a Seção 0 baixa a base e o dicionário oficiais do site do Cetic.br.
4. Todos os procedimentos aleatórios usam a semente `SEED = 2024`, então os ICs bootstrap são reproduzíveis.
5. Este notebook é gerado por `scripts/build_notebook.py`. Edite o script e regenere o notebook para manter os dois sincronizados.
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"Notebook escrito em {OUT} ({len(cells)} células)")
