"""Gera o notebook do estudo TIC Domicílios 2025 (indivíduos).

Uso (a partir da pasta PROJ 2):
    python scripts/build_notebook_domicilios.py notebooks/03_tic_domicilios_2025_habilidades_ia.ipynb
    jupyter nbconvert --to notebook --execute --inplace notebooks/03_tic_domicilios_2025_habilidades_ia.ipynb
"""
import sys
from pathlib import Path

import nbformat as nbf

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "notebooks/03_tic_domicilios_2025_habilidades_ia.ipynb")
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip()))


# =============================================================================
# CAPA
# =============================================================================
md(r"""
# Habilidades digitais e uso de inteligência artificial no Brasil
### Um estudo não paramétrico com os microdados da TIC Domicílios 2025 (indivíduos)

**Base de dados:** TIC Domicílios 2025, módulo de indivíduos, microdados v1.0, Cetic.br/NIC.br.
**Foco metodológico:** testes de hipóteses não paramétricos para dados binários pareados (Cochran Q e McNemar), para grupos ordenados (Kruskal-Wallis, Dunn e Jonckheere-Terpstra), para dois grupos com estratificação (Mann-Whitney e van Elteren) e para tabelas de contingência (qui-quadrado e tendência de Cochran-Armitage). Todos são complementados por uma análise que respeita o plano amostral complexo.

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
from statsmodels.stats.contingency_tables import cochrans_q, mcnemar
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option("display.max_colwidth", 110)
pd.set_option("display.width", 160)

SEED = 2025          # semente única para todos os procedimentos aleatórios (bootstrap)
ALPHA = 0.05         # nível de significância adotado em todos os testes
B_BOOT = 1000        # reamostragens bootstrap para intervalos de confiança de efeitos
''')
code(r'''
# 0.2 Localização dos arquivos (baixa a base oficial se ela não estiver presente)
def localizar_raiz():
    """Retorna a pasta do projeto: a que contém data/tic_domicilios_2025 (ou a pai de notebooks/)."""
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / "data" / "tic_domicilios_2025").exists():
            return p
    return Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()


RAIZ = localizar_raiz()
PASTA_DADOS = RAIZ / "data" / "tic_domicilios_2025"
ARQ_DADOS = PASTA_DADOS / "tic_domicilios_2025_individuos_microdados_v1.0.csv"
ARQ_DICIO = PASTA_DADOS / "tic_domicilios_2025_individuos_dicionario_de_variaveis_v1.0.xlsx"
URLS = {
    ARQ_DADOS: "https://cetic.br/media/microdados/988/tic_domicilios_2025_individuos_base_de_microdados_v1.0.csv",
    ARQ_DICIO: "https://cetic.br/media/microdados/984/tic_domicilios_2025_individuos_dicionario_de_variaveis_v1.0.xlsx",
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


COR = {"principal": "#1f6f8b", "secundaria": "#e08e45", "neutro": "#9aa5b1", "destaque": "#c0392b",
       "seq": ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c", "#08306b"]}
pio.templates["tic"] = go.layout.Template(layout=dict(
    font=dict(family="Arial, Helvetica, sans-serif", size=14, color="#222"),
    title=dict(x=0.01, xanchor="left", font=dict(size=17)),
    paper_bgcolor="white", plot_bgcolor="white",
    xaxis=dict(showgrid=False, linecolor="#555", ticks="outside", title_standoff=8),
    yaxis=dict(gridcolor="#e6e6e6", linecolor="#555", zeroline=False),
    colorway=[COR["principal"], COR["secundaria"], "#4daf4a", "#984ea3", "#ff7f00", "#a65628"],
    legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
    margin=dict(t=90, l=70, r=30, b=60), hoverlabel=dict(font_size=13),
))
pio.templates.default = "simple_white+tic"
''')
code(r'''
# 0.4 Funções estatísticas reutilizáveis
ESPECIAIS = {97: "Não sabe", 98: "Não respondeu", 99: "Não se aplica"}


def sem_especiais(s):
    """Converte os códigos especiais (97, 98, 99) em ausente (NaN)."""
    return s.where(~s.isin(list(ESPECIAIS)))


def ic_bootstrap(estatistica, *arrays, B=B_BOOT, seed=SEED, nivel=0.95):
    """IC percentil por bootstrap não paramétrico (reamostra linhas com reposição)."""
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    est = np.array([estatistica(*(a[idx] for a in arrays))
                    for idx in (rng.integers(0, n, n) for _ in range(B))])
    a = (1 - nivel) / 2
    return np.quantile(est, [a, 1 - a])


def resumo_mcnemar(x, y):
    """McNemar para dois itens binários pareados: tabela, testes, diferença de proporções e razão de chances pareada."""
    x, y = np.asarray(x, int), np.asarray(y, int)
    n = len(x)
    b = int(((x == 1) & (y == 0)).sum())   # só o primeiro item = 1
    c = int(((x == 0) & (y == 1)).sum())   # só o segundo item = 1
    tab = np.array([[((x == 1) & (y == 1)).sum(), b], [c, ((x == 0) & (y == 0)).sum()]])
    assint = mcnemar(tab, exact=False, correction=True)
    dif = (b - c) / n
    ep = np.sqrt((b + c) - (b - c) ** 2 / n) / n
    razao = b / c
    ic_razao = np.exp(np.log(razao) + np.array([-1, 1]) * 1.96 * np.sqrt(1 / b + 1 / c))
    return dict(n=n, b=b, c=c, qui2=assint.statistic, p=assint.pvalue, dif=dif,
                dif_ic=(dif - 1.96 * ep, dif + 1.96 * ep), razao=razao, razao_ic=tuple(ic_razao))


def epsilon_quadrado(H, n_total):
    """Tamanho de efeito do Kruskal-Wallis: epsilon² = H / (n - 1) (Tomczak & Tomczak, 2014)."""
    return H / (n_total - 1)


def teste_dunn(valores, grupos, ordem=None):
    """Teste de Dunn (1964): postos conjuntos, correção de empates, p-valores ajustados por Holm."""
    dados = pd.DataFrame({"v": np.asarray(valores), "g": np.asarray(grupos)})
    dados["posto"] = stats.rankdata(dados.v)
    n = len(dados)
    _, t = np.unique(dados.v, return_counts=True)
    correcao = (t**3 - t).sum() / (12 * (n - 1))
    resumo = dados.groupby("g").posto.agg(["mean", "size"])
    nomes = ordem or list(resumo.index)
    linhas = []
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


def jonckheere_terpstra(grupos_ordenados):
    """Teste de Jonckheere-Terpstra para alternativa ordenada (Hollander, Wolfe & Chicken, 2014).

    J = soma, para i < j, de U_ij = #(x_i < x_j) + 0,5 #(x_i = x_j). Variância exata com correção de empates;
    aproximação normal. Retorna J, E[J], z e o p-valor unilateral (tendência crescente) e bilateral.
    """
    gs = [np.asarray(g, float) for g in grupos_ordenados]
    n_i = np.array([len(g) for g in gs], dtype=float)    # float: os produtos abaixo estouram int64 com n grande
    N = n_i.sum()
    J = sum(stats.mannwhitneyu(gs[j], gs[i]).statistic for i in range(len(gs)) for j in range(i + 1, len(gs)))
    _, t = np.unique(np.concatenate(gs), return_counts=True)
    t = t.astype(float)
    media = (N**2 - (n_i**2).sum()) / 4
    var = ((N * (N - 1) * (2 * N + 5) - (n_i * (n_i - 1) * (2 * n_i + 5)).sum() - (t * (t - 1) * (2 * t + 5)).sum()) / 72
           + (n_i * (n_i - 1) * (n_i - 2)).sum() * (t * (t - 1) * (t - 2)).sum() / (36 * N * (N - 1) * (N - 2))
           + (n_i * (n_i - 1)).sum() * (t * (t - 1)).sum() / (8 * N * (N - 1)))
    z = (J - media) / np.sqrt(var)
    return dict(J=J, media=media, z=z, p_unilateral=stats.norm.sf(z), p_bilateral=2 * stats.norm.sf(abs(z)))


def van_elteren(valores, grupo1, estratos):
    """Teste de van Elteren (1960): Wilcoxon-Mann-Whitney estratificado, com pesos 1/(N_s + 1) e correção de empates."""
    dados = pd.DataFrame({"v": np.asarray(valores), "g1": np.asarray(grupo1, bool), "s": np.asarray(estratos)})
    num, var = 0.0, 0.0
    for _, d in dados.groupby("s"):
        n1, N = d.g1.sum(), len(d)
        if n1 == 0 or n1 == N:
            continue
        r = stats.rankdata(d.v)
        _, t = np.unique(d.v, return_counts=True)
        W = r[d.g1.to_numpy()].sum()
        E = n1 * (N + 1) / 2
        V = n1 * (N - n1) / 12 * ((N + 1) - (t**3 - t).sum() / (N * (N - 1)))
        num += (W - E) / (N + 1)
        var += V / (N + 1) ** 2
    z = num / np.sqrt(var)
    return dict(z=z, p=2 * stats.norm.sf(abs(z)))


def cochran_armitage(sucessos, totais, escores=None):
    """Teste de tendência de Cochran-Armitage para proporções em grupos ordenados."""
    x, n = np.asarray(sucessos, float), np.asarray(totais, float)
    s = np.arange(1, len(x) + 1, dtype=float) if escores is None else np.asarray(escores, float)
    p = x.sum() / n.sum()
    s_bar = (n * s).sum() / n.sum()
    T = (x * (s - s_bar)).sum()
    var = p * (1 - p) * (n * (s - s_bar) ** 2).sum()
    z = T / np.sqrt(var)
    return dict(z=z, p=2 * stats.norm.sf(abs(z)))


def residuos_ajustados(tabela):
    """Resíduos padronizados ajustados de Haberman: (O - E) / sqrt(E (1 - linha/n) (1 - coluna/n))."""
    O = np.asarray(tabela, float)
    n = O.sum()
    lin, col = O.sum(axis=1, keepdims=True), O.sum(axis=0, keepdims=True)
    E = lin @ col / n
    return (O - E) / np.sqrt(E * (1 - lin / n) * (1 - col / n))
''')
code(r'''
# 0.5 Estimação com o plano amostral: pesos replicados bootstrap (Rao-Wu), como no pacote survey do R
def estimativa_com_replicas(estimador, base, nivel=0.95):
    """Aplica `estimador(pesos)` ao PESO e às 200 réplicas. Aceita estimativas escalares ou vetoriais.

    Variância bootstrap: V = 1/(R-1) * soma_r (theta_r - média(theta_r))².
    """
    theta = np.atleast_1d(estimador(base["PESO"].to_numpy()))
    reps = np.array([np.atleast_1d(estimador(base[c].to_numpy())) for c in COLS_REPLICAS])
    ep = reps.std(axis=0, ddof=1)
    zc = stats.norm.isf((1 - nivel) / 2)
    return dict(estimativa=theta, ep=ep, ic_inf=theta - zc * ep, ic_sup=theta + zc * ep,
                p=2 * stats.norm.sf(np.abs(theta / ep)), replicas=reps)


def wald_igualdade(theta, replicas):
    """Teste de Wald por desenho para H0: todos os componentes de theta iguais (contrastes sucessivos)."""
    k = len(theta)
    C = np.eye(k)[:-1] - np.eye(k)[1:]
    cov = np.atleast_2d(np.cov((replicas @ C.T).T, ddof=1))
    X2 = float((C @ theta) @ np.linalg.solve(cov, C @ theta))
    return X2, k - 1, stats.chi2.sf(X2, k - 1)


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
O acesso à Internet no Brasil está próximo da universalização entre os mais jovens, mas **usar** a Internet não é o mesmo que **saber usá-la**. A literatura sobre exclusão digital distingue três níveis: o acesso (primeiro nível), as **habilidades e usos** (segundo nível) e os **resultados** obtidos (terceiro nível) (van Deursen & van Dijk, 2014; Helsper, 2021). Em 2025, essa discussão ganhou um elemento novo: as ferramentas de **inteligência artificial generativa** (ChatGPT, Gemini, Copilot, Meta IA), cuja adoção pode reproduzir ou ampliar as desigualdades digitais existentes.

A TIC Domicílios 2025 mede, para pessoas de 10 anos ou mais, 12 habilidades digitais realizadas nos últimos 3 meses e, pela primeira vez com esse detalhamento, o uso de ferramentas de IA.

### 1.2. Motivação e relevância
* **Social:** identificar quem fica para trás nas habilidades digitais e na adoção de IA orienta políticas de letramento digital.
* **Desinformação e segurança:** checar informações e proteger contas e dados são habilidades de cidadania digital com impacto direto na vida das pessoas.
* **Didática:** a base tem itens binários repetidos na mesma pessoa (pareados), escores discretos com muitos empates, grupos ordenados e tabelas de contingência. É um bom laboratório para diferentes testes não paramétricos.

### 1.3. Objetivos
**Objetivo geral:** investigar, com testes não paramétricos, desigualdades em habilidades digitais e no uso de IA entre usuários de Internet no Brasil em 2025.

**Objetivos específicos:**
1. Comparar, **na mesma pessoa**, três práticas de cidadania digital: adotar medidas de segurança, ajustar a privacidade e verificar informações (H1).
2. Verificar se o número de habilidades digitais cresce com a escolaridade (H2).
3. Comparar usuários urbanos e rurais, inclusive entre pessoas com a mesma escolaridade (H3).
4. Verificar se o uso de IA está associado à faixa etária (H4).
5. Avaliar se as conclusões mudam quando se incorpora o plano amostral complexo (pesos replicados).

### 1.4. Perguntas de pesquisa
* **P1.** As três práticas de cidadania digital são realizadas com a mesma frequência pelos usuários de Internet?
* **P2.** O número de habilidades digitais difere entre os níveis de escolaridade? Ele cresce com a escolaridade?
* **P3.** Usuários urbanos e rurais diferem no número de habilidades digitais, mesmo quando comparamos pessoas com a mesma escolaridade?
* **P4.** O uso de ferramentas de IA nos últimos 3 meses está associado à faixa etária?

### 1.5. Caracterização inicial das hipóteses
* **H1** (3 itens binários, pareados): as proporções de "Sim" nas três práticas não são todas iguais.
* **H2** (escore discreto, 4 grupos ordenados independentes): a distribuição do escore difere entre os níveis de escolaridade, com tendência crescente.
* **H3** (escore discreto, 2 grupos independentes, estratificado): usuários urbanos tendem a ter mais habilidades que os rurais.
* **H4** (binária × 6 grupos ordenados): o uso de IA não é independente da faixa etária.

A formulação formal está na Seção 5.

### 1.6. Metodologia em resumo
Análise exploratória, seguida de testes não paramétricos (Cochran Q + McNemar; Kruskal-Wallis + Dunn + Jonckheere-Terpstra; Mann-Whitney + van Elteren; qui-quadrado + resíduos ajustados + Cochran-Armitage), com tamanhos de efeito e intervalos de confiança, correção de Holm e uma **camada de inferência populacional** com os 200 pesos replicados da pesquisa.

### 1.7. Contribuição e limitações antecipadas
A contribuição é quantificar, com dados oficiais de 2025, desigualdades de segundo nível (habilidades) e a adoção inicial de IA, com métodos adequados a dados binários e ordinais. As limitações principais, discutidas na Seção 9, são três. O estudo é observacional e transversal, então não permite conclusões causais. As habilidades são autodeclaradas e se referem aos últimos 3 meses (medem *prática recente*, não *competência*). E as hipóteses foram definidas depois de uma exploração preliminar desta mesma base, sem pré-registro.
""")

# =============================================================================
# 2. DATASET
# =============================================================================
md(r"""
## 2. Compreensão e documentação do dataset

**O que vamos fazer?** Documentar a origem, o desenho amostral e as variáveis usadas, consultando o **dicionário oficial** e os relatórios metodológico e de coleta.

**Por que isso é necessário?** O significado de uma variável não pode ser presumido pelo nome. Por exemplo, a seção 3 mostra que, numa das variáveis derivadas, o código 99 não significa "Não se aplica". O plano amostral também define o tipo de inferência que é válido.

### 2.1. Ficha técnica

| Item | Descrição |
|---|---|
| Nome | TIC Domicílios 2025, módulo de indivíduos, base de microdados v1.0 |
| Instituição | Centro Regional de Estudos para o Desenvolvimento da Sociedade da Informação (Cetic.br), do Núcleo de Informação e Coordenação do Ponto BR (NIC.br) |
| Período de referência | Coleta de março a setembro de 2025. Os itens analisados se referem aos últimos 3 meses |
| População-alvo | População com 10 anos ou mais residente em domicílios particulares permanentes no Brasil |
| Unidade de observação | Um morador por domicílio, selecionado aleatoriamente (10–17 anos ou 18 anos ou mais, conforme a composição do domicílio; a seleção é coordenada com a TIC Kids Online) |
| Coleta | Entrevista face a face com tablet (CAPI) |
| Plano amostral | Amostragem estratificada de conglomerados em 3 ou 4 estágios. A estratificação é geográfica (UF, capitais e regiões metropolitanas), por situação (urbana/rural) e socioeconômica, com cadastro do Censo 2022 |
| Pesos | `PESO` (corrigido para não resposta e calibrado) e 200 pesos replicados `rep1`–`rep200`, gerados por bootstrap com o mesmo desenho (`as.svrepdesign` do pacote *survey* do R) |
| Fontes | Dicionário de variáveis v1.0, relatório metodológico, relatório de coleta de dados e questionário (cetic.br) |
""")
code(r'''
# 2.2 Carga dos dados. `dados_brutos` é preservado intacto; todas as transformações usam cópias.
dados_brutos = pd.read_csv(ARQ_DADOS, sep=";", decimal=",", low_memory=False).copy()
COLS_REPLICAS = [f"rep{i}" for i in range(1, 201)]

dicionario = pd.read_excel(ARQ_DICIO, sheet_name="Indivíduos", header=1)
dicionario.columns = ["tamanho", "variavel", "descricao", "tipo_arquivo", "codigos"]
dicionario = dicionario.dropna(subset=["variavel"]).set_index("variavel")

print(f"Registros: {fmt(dados_brutos.shape[0], 0)} | Variáveis: {dados_brutos.shape[1]}")
print(f"Linhas do dicionário: {len(dicionario)} (as réplicas rep1–rep200 aparecem numa linha só)")
print(f"Soma do PESO (população estimada de 10 anos ou mais): {fmt(dados_brutos.PESO.sum(), 0)}")
''')
md(r"""
### 2.3. Variáveis utilizadas

A tabela é montada **a partir do dicionário oficial**. A descrição e os códigos vêm do arquivo do Cetic.br; o tipo estatístico e a utilização são a classificação adotada neste estudo.
""")
code(r'''
def codigos_do_dicionario(var):
    """Extrai os códigos numéricos válidos ('1="..."') da coluna de rótulos do dicionário."""
    texto = dicionario.loc[var, "codigos"]
    if pd.isna(texto):
        return None
    return sorted({int(c) for c in pd.Series(str(texto)).str.findall(r"(-?\d+)\s*=").iloc[0]})


HAB = [f"I1A_{l}" for l in "ABCDEFGHIJKL"]
ROT_HAB = {
    "I1A_A": "Copiou ou moveu arquivo/pasta", "I1A_B": "Usou copiar e colar", "I1A_C": "Anexou arquivo a mensagem",
    "I1A_D": "Usou fórmula em planilha", "I1A_E": "Conectou/instalou equipamento", "I1A_F": "Instalou programa ou app",
    "I1A_G": "Criou apresentação de slides", "I1A_H": "Transferiu arquivos entre dispositivos",
    "I1A_I": "Programou (linguagem de programação)", "I1A_J": "Adotou medidas de segurança",
    "I1A_K": "Mudou configurações de privacidade", "I1A_L": "Verificou se uma informação era verdadeira",
}
VARIAVEIS = [
    ("I1A_J", "Binária", "H1 (segurança), escore"),
    ("I1A_K", "Binária", "H1 (privacidade), escore"),
    ("I1A_L", "Binária", "H1 (checagem de informação), escore"),
    ("I1A_A", "Binária", f"Um dos {len(HAB)} itens do escore de habilidades (H2, H3)"),
    ("C13A", "Binária", "H4 (desfecho: uso de IA)"),
    ("FAIXA_ETARIA", "Ordinal (6 níveis)", "H4 (grupo), EDA"),
    ("GRAU_INST_1", "Ordinal (4 níveis)", "H2 (grupo), estrato em H3"),
    ("AREA", "Nominal binária", "H3 (grupo)"),
    ("SEXO", "Nominal binária", "EDA"),
    ("CLASSE_2015", "Ordinal (4 níveis)", "EDA (possível confundidor)"),
    ("COD_REGIAO_2", "Nominal (5 níveis)", "EDA (representatividade)"),
    ("IDADE", "Quantitativa discreta", "EDA, validação da faixa etária"),
    ("C3J3", "Binária", "Filtro: usuário de Internet (indicador ampliado)"),
    ("C4", "Ordinal (4 níveis)", "EDA (frequência de uso)"),
    ("I1A_NENHUM", "Binária (derivada)", "Auditoria de codificação"),
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
**Os 12 itens de habilidade digital** (`I1A_A`–`I1A_L`), todos com a pergunta "Nos últimos 3 meses, o respondente…":
copiou ou moveu arquivo; usou copiar e colar; anexou arquivo a mensagem; usou fórmula em planilha; conectou ou instalou equipamento; instalou programa ou aplicativo; criou apresentação de slides; transferiu arquivos entre dispositivos; programou em linguagem de programação; **adotou medidas de segurança** (senhas fortes, verificação em duas etapas); **mudou configurações de privacidade**; **verificou se uma informação encontrada na Internet era verdadeira**.

**Classificação das variáveis por tipo:**
* **Binárias:** os itens de habilidade, o uso de IA e o filtro de usuário.
* **Ordinais:** faixa etária, grau de instrução, classe e frequência de uso.
* **Nominais:** área, sexo e região.
* **Quantitativas:** idade (discreta) e pesos (contínua).

**Códigos especiais** (97 = Não sabe, 98 = Não respondeu, 99 = Não se aplica) não pertencem às escalas e são tratados na Seção 3.

### 2.4. Critérios de inclusão e exclusão

| Análise | População analisada | Inclusão | Exclusão |
|---|---|---|---|
| H1 | Usuários de Internet (`C3J3 = 1`) | `I1A_J`, `I1A_K` e `I1A_L` iguais a 0 ou 1 | Não usuários (99) e 97/98 em qualquer um dos 3 itens |
| H2, H3 | Usuários de Internet | Pelo menos 10 dos 12 itens de habilidade válidos (escore reescalado para 0–12) | Não usuários e quem tem mais de 2 itens com 97/98 |
| H4 | Usuários de Internet | `C13A` igual a 0 ou 1 | Não usuários (99) e 97/98, investigados na Seção 3 |

### 2.5. Limitações conhecidas da base
* A amostra **não é autoponderada**: há sobreamostragem de UF menores, de áreas rurais e de estratos socioeconômicos. Sem pesos, as proporções descrevem a amostra, não o Brasil.
* O microdado **não traz identificadores de estrato nem de UPA**. O plano amostral só pode ser incorporado pelos pesos replicados.
* As habilidades são **autodeclaradas** e se referem a atividades realizadas nos últimos 3 meses. Quem sabe fazer algo, mas não fez no período, responde "Não".
* O uso de IA depende de a pessoa **reconhecer** que usou uma ferramenta de IA. Assistentes embutidos em outros aplicativos podem não ser identificados como tal.
""")

# =============================================================================
# 3. PREPARAÇÃO
# =============================================================================
md(r"""
## 3. Preparação e qualidade dos dados

**O que vamos fazer?** Auditar a integridade da base e entender as ausências antes de excluir qualquer observação.

**Por que isso é necessário?** Excluir casos sem investigar o motivo pode introduzir viés de seleção. Códigos especiais tratados como números distorceriam os postos e as proporções.

### 3.1. Integridade: duplicados, ausentes "técnicos" e codificação
""")
code(r'''
VARS_CATEGORICAS = ["C13A", "C3J3", "C4", "C1", "SEXO", "FAIXA_ETARIA", "AREA", "COD_REGIAO_2", "CLASSE_2015",
                    "GRAU_INST_1", "I1A_NENHUM", "F_C3_IDADE"] + HAB
fora_do_dicionario = {v: sorted(set(dados_brutos[v].unique()) - set(codigos_do_dicionario(v)))
                      for v in VARS_CATEGORICAS if not set(dados_brutos[v].unique()) <= set(codigos_do_dicionario(v))}
faixa_pela_idade = pd.cut(dados_brutos.IDADE, [9, 15, 24, 34, 44, 59, 200], labels=[1, 2, 3, 4, 5, 6]).astype(int)
auditoria = pd.DataFrame({
    "verificação": ["Linhas duplicadas", "Domicílios repetidos (ID_DOMICILIO)", "Células vazias (NaN) no arquivo",
                    "Colunas não numéricas", "FAIXA_ETARIA incoerente com IDADE",
                    "Variáveis categóricas com código fora do dicionário"],
    "resultado": [
        int(dados_brutos.duplicated().sum()),
        int(dados_brutos.ID_DOMICILIO.duplicated().sum()),
        int(dados_brutos.isna().sum().sum()),
        len(dados_brutos.select_dtypes(exclude="number").columns),
        int((faixa_pela_idade != dados_brutos.FAIXA_ETARIA).sum()),
        len(fora_do_dicionario),
    ]})
display(auditoria)
print("Códigos observados que o dicionário não documenta:", fora_do_dicionario)
assert auditoria.resultado.iloc[:5].eq(0).all(), "Problema de integridade: investigar antes de prosseguir"
# única exceção conhecida: o filtro F_C3_IDADE tem o valor 0, não documentado (não é usado nas análises)
assert fora_do_dicionario == {"F_C3_IDADE": [0]}, "Código não documentado inesperado: investigar"
''')
md(r"""
**Interpretação.** O arquivo não tem duplicatas nem células vazias; há um morador por domicílio, e a faixa etária é coerente com a idade em todos os registros. Há **uma inconsistência de codificação**: o dicionário documenta apenas o código 1 da variável-filtro `F_C3_IDADE`, mas a base também tem o valor 0 (não pertence ao filtro). Essa variável não é usada nas análises. Como na TIC Kids Online, a ausência **não aparece como célula vazia**: ela vem codificada (97, 98, 99).

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


tabela_especiais(["C3J3", "C4", "C13A", "I1A_J", "I1A_K", "I1A_L", "I1A_I"])
''')
md(r"""
### 3.3. Regras de elegibilidade e saltos do questionário

**O que vamos fazer?** Testar se o código 99 corresponde exatamente às regras de salto do questionário.

**Por que?** Se a correspondência for exata, o 99 é uma **ausência estrutural**: a pergunta não se aplica a quem não usa a Internet. Excluir esses casos então não é um viés, e sim a definição da população de referência, que são os **usuários de Internet**.
""")
code(r'''
b = dados_brutos
nao_usuario = b.C3J3 != 1
regras = pd.DataFrame({
    "regra": ["C4 = 99  ⇔  não é usuário de Internet (C3J3 ≠ 1)",
              "C13A = 99  ⇔  não é usuário de Internet",
              f"Itens de habilidade I1A_A–L = 99  ⇔  não é usuário de Internet (todos os {len(HAB)})",
              "Entre usuários: I1A_NENHUM = 1  ⇔  nenhum item de habilidade igual a 1"],
    "vale para todos os registros?": [
        ((b.C4 == 99) == nao_usuario).all(),
        ((b.C13A == 99) == nao_usuario).all(),
        all(((b[c] == 99) == nao_usuario).all() for c in HAB),
        ((b.loc[~nao_usuario, "I1A_NENHUM"] == 1) == ~(b.loc[~nao_usuario, HAB] == 1).any(axis=1)).all()],
})
display(regras)
assert regras.iloc[:, 1].all()
display(pd.crosstab(b.I1A_NENHUM.rename("I1A_NENHUM"), b.C3J3.rename("C3J3 (usuário)")))
''')
md(r"""
**Interpretação.** As regras de salto valem para **todos** os registros: o 99 nos itens analisados é ausência estrutural.

A tabela cruzada revela uma **segunda inconsistência de codificação**, na variável derivada `I1A_NENHUM`. Pelo dicionário, 1 = "Nenhuma dessas atividades" e 99 = "Não se aplica". Na base:
* os **não usuários** de Internet recebem **1**, e não 99;
* os usuários com **pelo menos uma** habilidade recebem **99**.

Usar essa variável ao pé da letra contaria todos os não usuários como "usuários sem nenhuma habilidade". Por isso, **o escore deste estudo é calculado diretamente dos 12 itens**, e `I1A_NENHUM` serve apenas para conferência, já que a regra entre usuários bate.

### 3.4. A ausência por "Não sabe"/"Não respondeu" é aleatória?

**O que vamos fazer?** Ver como o "Não sabe" do uso de IA (C13A = 97) varia com a idade.

**Por que?** Se a proporção de "Não sabe" cresce com a idade, excluir esses casos pode distorcer justamente a comparação etária de H4.
""")
code(r'''
usu = b[~nao_usuario]
nao_sabe_ia = usu.groupby("FAIXA_ETARIA").C13A.apply(lambda z: (z == 97).mean())
ROT_FAIXA = {1: "10–15", 2: "16–24", 3: "25–34", 4: "35–44", 5: "45–59", 6: "60+"}
display(nao_sabe_ia.rename(index=ROT_FAIXA).map(fmt_pct).to_frame("C13A = 97 (Não sabe) entre usuários"))
excl_hab = (usu[HAB].isin([97, 98]).sum(axis=1) > 2).sum()
print(f"Usuários excluídos do escore de habilidades por mais de 2 itens com 97/98: {excl_hab} "
      f"({fmt_pct(excl_hab / len(usu))} dos usuários)")
''')
code(r'''
mostrar(f"""
**Interpretação.** O "Não sabe" sobre o uso de IA é raro, mas **cresce com a idade**: vai de {fmt_pct(nao_sabe_ia.iloc[0])}
na faixa de 10–15 anos a {fmt_pct(nao_sabe_ia.iloc[-1])} entre as pessoas de 60 anos ou mais. A ausência, portanto,
**não é completamente aleatória**. Provavelmente, parte dos usuários mais velhos não sabe se o que usou era "IA". Como a
proporção é pequena, a exclusão não deve mudar a conclusão, mas isso será **testado** numa análise de sensibilidade
(7.6), recodificando "Não sabe" como "Não". Já a exclusão do escore de habilidades atinge muito poucos usuários.
""")
''')
md(r"""
### 3.5. Valores extremos
As variáveis analisadas são binárias, ordinais ou um escore limitado de 0 a 12, então não existem "outliers" no sentido usual. O único componente contínuo é o **peso amostral**. Pesos muito desiguais aumentam a variância das estimativas ponderadas, efeito que medimos pelo **efeito de desenho de Kish** devido à ponderação: $\text{deff}_w = 1 + CV_w^2$.
""")
code(r'''
cv_peso = b.PESO.std(ddof=0) / b.PESO.mean()
deff_kish = 1 + cv_peso**2
print(f"PESO: mínimo {fmt(b.PESO.min(), 1)} | mediana {fmt(b.PESO.median(), 1)} | máximo {fmt(b.PESO.max(), 1)}")
print(f"Efeito de desenho de Kish (ponderação): {fmt(deff_kish, 2)}  ->  tamanho efetivo ≈ {fmt(len(b) / deff_kish, 0)} de {fmt(len(b), 0)}")
''')
md(r"""
**Interpretação.** Os pesos são bastante desiguais, o que reduz o tamanho efetivo da amostra para estimativas populacionais. Isso justifica a camada com pesos replicados (7.7). Os pesos **não são removidos nem truncados**, porque fazem parte do desenho oficial.

### 3.6. Recodificações e bases derivadas

| Transformação | Justificativa |
|---|---|
| 97/98/99 → ausente | Não são categorias de resposta substantivas |
| Escore de habilidades = (proporção de "Sim" entre os itens válidos) × 12, exigindo ≥ 10 itens válidos | Resume 12 itens binários num escore de 0 a 12. Quem tem os 12 itens válidos fica com o número exato de habilidades; o reescalonamento evita penalizar quem tem 1 ou 2 itens com "Não sabe" |
| `C13A`: mantém apenas 0/1 | Desfecho binário de H4. Os 97 são analisados como sensibilidade |
| Rótulos de faixa, escolaridade, área e sexo | Legibilidade de tabelas e gráficos |
""")
code(r'''
ROT_GRAU = {1: "Analfabeto/Infantil", 2: "Fundamental", 3: "Médio", 4: "Superior"}
ROT_AREA = {1: "Urbana", 2: "Rural"}
ROT_SEXO = {1: "Masculino", 2: "Feminino"}
ROT_CLASSE = {1: "A", 2: "B", 3: "C", 4: "DE"}
ROT_REGIAO = {1: "Sudeste", 2: "Nordeste", 3: "Sul", 4: "Norte", 5: "Centro-Oeste"}
ORDEM_FAIXA, ORDEM_GRAU = list(ROT_FAIXA.values()), list(ROT_GRAU.values())

base = dados_brutos[["ID_DOMICILIO", "IDADE", "FAIXA_ETARIA", "SEXO", "AREA", "COD_REGIAO_2", "CLASSE_2015",
                     "GRAU_INST_1", "C3J3", "C4", "PESO", *COLS_REPLICAS]].copy()
base["faixa"] = base.FAIXA_ETARIA.map(ROT_FAIXA)
base["grau"] = base.GRAU_INST_1.map(ROT_GRAU)
base["area"] = base.AREA.map(ROT_AREA)
base["sexo"] = base.SEXO.map(ROT_SEXO)
base["classe"] = base.CLASSE_2015.map(ROT_CLASSE)
base["regiao"] = base.COD_REGIAO_2.map(ROT_REGIAO)
base["usuario"] = base.C3J3 == 1

itens = dados_brutos[HAB].apply(sem_especiais)
base[HAB] = itens
base["n_itens_validos"] = itens.notna().sum(axis=1)
base["escore_hab"] = (itens.mean(axis=1) * len(HAB)).where(base.n_itens_validos >= 10)
base["IA"] = dados_brutos.C13A.where(dados_brutos.C13A.isin([0, 1]))

# Bases analíticas (subconjuntos rastreáveis da base tratada)
usuarios = base[base.usuario].copy()
cidadania = usuarios.dropna(subset=["I1A_J", "I1A_K", "I1A_L"]).copy()
habil = usuarios[usuarios.escore_hab.notna()].copy()
ia = usuarios[usuarios.IA.notna()].copy()

# validações
assert set(np.unique(cidadania[["I1A_J", "I1A_K", "I1A_L"]].to_numpy())) == {0, 1}
assert habil.escore_hab.between(0, 12).all()
assert set(ia.IA.unique()) == {0, 1}
''')
code(r'''
fluxo = pd.DataFrame([
    ("Base original (10 anos ou mais)", len(dados_brutos), "—"),
    ("Usuários de Internet (C3J3 = 1)", len(usuarios), "indicador ampliado de usuário"),
    ("H1 · três práticas de cidadania digital", len(cidadania), "I1A_J, I1A_K e I1A_L válidos"),
    ("H2/H3 · escore de habilidades", len(habil), "≥ 10 dos 12 itens válidos"),
    ("H4 · uso de IA válido", len(ia), "C13A = 0/1 (exclui 97/98)"),
], columns=["Etapa", "n", "Critério"])
fluxo["% da base"] = (fluxo.n / len(dados_brutos)).map(fmt_pct)
display(fluxo)

fig = go.Figure(go.Bar(x=fluxo.n, y=fluxo.Etapa, orientation="h",
                       marker_color=[COR["neutro"], COR["secundaria"]] + [COR["principal"]] * 3,
                       text=[fmt(n, 0) + f" ({p})" for n, p in zip(fluxo.n, fluxo["% da base"])],
                       textposition="auto", customdata=fluxo.Critério,
                       hovertemplate="<b>%{y}</b><br>n = %{x}<br>%{customdata}<extra></extra>"))
fig.update_layout(title="Fluxo amostral: da base original às amostras analíticas", height=360,
                  xaxis_title="Número de pessoas", yaxis=dict(autorange="reversed", title=""), margin=dict(l=260))
fig.show()
''')
md(r"""
**Resumo da preparação.** A base original foi preservada (`dados_brutos`). Os não usuários de Internet ficam fora de todas as hipóteses por uma regra estrutural do questionário. Entre os usuários, as exclusões por "Não sabe"/"Não respondeu" são pequenas e foram investigadas. As conclusões, portanto, valem para **usuários de Internet de 10 anos ou mais**.
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
* **Metodologia:** proporções sem peso (amostra) e com `PESO` (população), por região e por faixa etária.
""")
code(r'''
def proporcoes(col, ordem, dados=base):
    sem = dados[col].value_counts(normalize=True).reindex(ordem)
    com = dados.groupby(col).PESO.sum().reindex(ordem) / dados.PESO.sum()
    return pd.DataFrame({"Amostra (sem peso)": sem, "População estimada (com peso)": com})


comp_regiao = proporcoes("regiao", list(ROT_REGIAO.values()))
comp_faixa = proporcoes("faixa", ORDEM_FAIXA)
display((pd.concat({"Região": comp_regiao, "Faixa etária": comp_faixa}) * 100).round(1))

fig = make_subplots(rows=1, cols=2, subplot_titles=("Por região", "Por faixa etária"), horizontal_spacing=0.08)
for j, comp in enumerate([comp_regiao, comp_faixa], start=1):
    for nome, cor in zip(comp.columns, [COR["neutro"], COR["principal"]]):
        fig.add_bar(name=nome, x=comp.index, y=comp[nome], marker_color=cor, showlegend=j == 1,
                    hovertemplate="%{x}<br>" + nome + ": %{y:.1%}<extra></extra>", row=1, col=j)
fig.update_yaxes(tickformat=".0%")
fig.update_layout(title="Composição: amostra × população estimada", barmode="group", height=430)
fig.show()
''')
code(r'''
mostrar(f"""
* **Resultado:** o Sudeste tem {fmt_pct(comp_regiao.loc['Sudeste', 'Amostra (sem peso)'])} da amostra, mas
  {fmt_pct(comp_regiao.loc['Sudeste', 'População estimada (com peso)'])} da população estimada. O Norte tem
  {fmt_pct(comp_regiao.loc['Norte', 'Amostra (sem peso)'])} da amostra e
  {fmt_pct(comp_regiao.loc['Norte', 'População estimada (com peso)'])} da população. Por idade, a faixa de 60+ tem
  {fmt_pct(comp_faixa.loc['60+', 'Amostra (sem peso)'])} da amostra e
  {fmt_pct(comp_faixa.loc['60+', 'População estimada (com peso)'])} da população.
* **Interpretação:** a amostra foi **desenhada** para não ser proporcional. Os resultados sem peso descrevem a amostra, e
  por isso a Seção 7.7 repete as estimativas com pesos.
* **Limitações:** os pesos também corrigem não resposta e calibram para totais populacionais, efeitos que este gráfico não mostra.
""")
''')
md(r"""
#### 4.1.2. Quem usa a Internet e com que frequência
* **Objetivo:** contextualizar a população analisada.
* **Metodologia:** proporção de usuários por faixa etária e distribuição da frequência de uso (`C4`), sem peso.
""")
code(r'''
taxa_uso = base.groupby("faixa").usuario.mean().reindex(ORDEM_FAIXA)
freq_uso = usuarios.C4.map({1: "Todos os dias ou quase", 2: "Pelo menos 1×/semana", 3: "Pelo menos 1×/mês",
                            4: "Menos de 1×/mês"}).value_counts(normalize=True)
display(taxa_uso.map(fmt_pct).to_frame("% de usuários de Internet"))
display(freq_uso.map(fmt_pct).to_frame("Frequência de uso entre usuários"))
mostrar(f"""
* **Resultado:** a proporção de usuários passa de {fmt_pct(taxa_uso.max())} nas faixas jovens para
  {fmt_pct(taxa_uso['60+'])} entre as pessoas de 60 anos ou mais. Entre os usuários,
  {fmt_pct(freq_uso.get('Todos os dias ou quase', 0))} usam a Internet todos os dias ou quase.
* **Interpretação:** o primeiro nível da exclusão digital (o acesso) ainda pesa entre os idosos. Este estudo trata do
  **segundo nível** (habilidades e usos) **entre quem já usa**, então os idosos analisados são um grupo selecionado.
* **Limitações:** C4 tem pouca variação (quase todos estão na categoria mais frequente) e por isso não é usado como desfecho.
""")
''')
md(r"""
#### 4.1.3. As 12 habilidades digitais e o escore
* **Objetivo:** descrever a prevalência de cada habilidade e a distribuição do escore.
* **Metodologia:** proporção de "Sim" por item com IC 95% de Wilson; histograma do escore; medidas de posição e dispersão.
""")
code(r'''
prev = pd.DataFrame({
    "item": HAB, "habilidade": [ROT_HAB[h] for h in HAB],
    "n": [int(usuarios[h].notna().sum()) for h in HAB], "sim": [int(usuarios[h].sum()) for h in HAB]})
prev["prop"] = prev.sim / prev.n
prev[["ic_inf", "ic_sup"]] = np.array([proportion_confint(s, n, method="wilson") for s, n in zip(prev.sim, prev.n)])
prev = prev.sort_values("prop")
destaque_h1 = prev.item.isin(["I1A_J", "I1A_K", "I1A_L"])

fig = go.Figure(go.Bar(
    x=prev.prop, y=prev.habilidade, orientation="h",
    marker_color=np.where(destaque_h1, COR["secundaria"], COR["principal"]),
    error_x=dict(type="data", symmetric=False, array=prev.ic_sup - prev.prop, arrayminus=prev.prop - prev.ic_inf),
    text=prev.prop.map(fmt_pct), textposition="inside", insidetextanchor="start", textfont=dict(color="white"),
    customdata=np.c_[prev.item, prev.sim, prev.n],
    hovertemplate="<b>%{y}</b> (%{customdata[0]})<br>%{x:.1%} (%{customdata[1]} de %{customdata[2]})<extra></extra>"))
fig.update_layout(title="Prevalência das habilidades digitais entre usuários (sem peso) · laranja = itens de H1",
                  xaxis=dict(tickformat=".0%", title="Proporção que realizou nos últimos 3 meses", range=[0, 0.55]),
                  yaxis_title="", height=520, margin=dict(l=280))
fig.show()
''')
code(r'''
esc = habil.escore_hab
zeros = (esc == 0).mean()
print(f"Escore de habilidades (0 a 12): n = {fmt(len(esc), 0)} | média = {fmt(esc.mean())} | DP = {fmt(esc.std())} | "
      f"mediana = {fmt(esc.median(), 1)} | Q1–Q3 = {fmt(esc.quantile(.25), 1)}–{fmt(esc.quantile(.75), 1)} | "
      f"assimetria = {fmt(stats.skew(esc))} | % com escore 0 = {fmt_pct(zeros)}")
cont = esc.round().value_counts().sort_index()
fig = go.Figure(go.Bar(x=cont.index, y=cont.values, marker_color=[COR["destaque"]] + [COR["principal"]] * (len(cont) - 1),
                       text=(cont / cont.sum()).map(lambda v: fmt_pct(v, 0)), textposition="outside",
                       hovertemplate="%{x} habilidades<br>%{y} usuários<extra></extra>"))
fig.update_layout(title="Número de habilidades digitais realizadas nos últimos 3 meses (usuários)", height=420,
                  xaxis=dict(title="Escore (número de habilidades, 0 a 12)", dtick=1), yaxis_title="Usuários")
fig.show()
''')
code(r'''
mostrar(f"""
* **Resultado:** as habilidades mais comuns ("{prev.habilidade.iloc[-1]}", "{prev.habilidade.iloc[-2]}") foram realizadas por
  {fmt_pct(prev.prop.iloc[-2])} a {fmt_pct(prev.prop.iloc[-1])} dos usuários; a menos comum ("{prev.habilidade.iloc[0]}"),
  por {fmt_pct(prev.prop.iloc[0])}. O escore é **fortemente assimétrico**: {fmt_pct(zeros)} dos usuários não realizaram
  nenhuma das 12 atividades, e a mediana é {fmt(esc.median(), 0)}.
* **Interpretação:** muitos usuários se limitam a usos básicos, como mensagens e redes sociais, que não estão entre os 12 itens.
  O excesso de zeros e os muitos empates tornam inadequados testes baseados em média e normalidade, o que
  **reforça o uso de testes de postos com correção de empates**.
* **Limitações:** o escore soma itens heterogêneos (programar ≠ copiar e colar) com o mesmo peso; ele mede **amplitude** de
  práticas, não profundidade de competência.
""")
''')
md(r"""
#### 4.1.4. Uso de inteligência artificial
""")
code(r'''
n_ia, sim_ia = len(ia), int(ia.IA.sum())
ic_ia = proportion_confint(sim_ia, n_ia, method="wilson")
print(f"Usaram ferramenta de IA nos últimos 3 meses: {sim_ia} de {n_ia} usuários = {fmt_pct(sim_ia / n_ia)} "
      f"(IC 95% Wilson: {fmt_pct(ic_ia[0])} a {fmt_pct(ic_ia[1])}), sem peso")
''')
md(r"""
* **Interpretação:** uma minoria dos usuários relata ter usado IA. A pergunta é sobre **reconhecer** o uso de ferramentas como ChatGPT, Gemini, Copilot ou Meta IA, então os valores podem subestimar o contato real com IA embutida em outros serviços.

### 4.2. Análise bivariada

#### 4.2.1. Habilidades por escolaridade, faixa etária e classe
* **Objetivo:** examinar a relação de H2 e possíveis confundidores (idade, classe).
* **Metodologia:** como o escore tem muitos zeros, boxplots ficariam degenerados. Usamos a **distribuição do escore em categorias** (0, 1–2, 3–5, 6–12) empilhada por grupo, mais a mediana e a média.
""")
code(r'''
CATS = ["0", "1–2", "3–5", "6–12"]
habil["escore_cat"] = pd.cut(habil.escore_hab, [-0.1, 0.5, 2.5, 5.5, 12.1], labels=CATS)


def distribuicao_escore(col, ordem):
    return pd.crosstab(habil[col], habil.escore_cat, normalize="index").reindex(ordem)[CATS]


blocos = [("grau", ORDEM_GRAU, "Escolaridade"), ("faixa", ORDEM_FAIXA, "Faixa etária"), ("classe", list(ROT_CLASSE.values()), "Classe")]
fig = make_subplots(rows=1, cols=3, shared_yaxes=True, subplot_titles=[t for *_, t in blocos], horizontal_spacing=0.04)
for j, (col, ordem, _) in enumerate(blocos, start=1):
    dist = distribuicao_escore(col, ordem)
    for k, cat in enumerate(CATS):
        fig.add_bar(x=dist.index, y=dist[cat], name=f"{cat} habilidades", marker_color=COR["seq"][k + 1],
                    showlegend=j == 1, hovertemplate="%{x}<br>" + cat + ": %{y:.1%}<extra></extra>", row=1, col=j)
fig.update_yaxes(tickformat=".0%")
fig.update_layout(title="Distribuição do número de habilidades por escolaridade, faixa etária e classe", barmode="stack",
                  height=520, yaxis_title="Proporção dos usuários", margin=dict(b=150),
                  legend=dict(orientation="h", traceorder="normal", yanchor="top", y=-0.32, x=0))
fig.show()
resumo_grupos = pd.concat({t: habil.groupby(col).escore_hab.agg(n="size", mediana="median", media="mean").reindex(o)
                           for col, o, t in blocos}).round(2)
display(resumo_grupos)
''')
code(r'''
med_grau = resumo_grupos.loc["Escolaridade"]
med_faixa = resumo_grupos.loc["Faixa etária"]
mostrar(f"""
* **Resultado:** a média do escore vai de {fmt(med_grau.media.iloc[0])} (Analfabeto/Infantil) a {fmt(med_grau.media.iloc[-1])}
  (Superior), e a proporção de usuários com zero habilidades cai de forma acentuada com a escolaridade. Por idade, o padrão
  **não é monotônico**: o pico está em 16–24 anos (média {fmt(med_faixa.loc['16–24', 'media'])}), e os de 10–15 anos ficam
  abaixo dos jovens adultos. Por classe, há um gradiente de A para DE.
* **Interpretação:** escolaridade, idade e classe estão associadas às habilidades. Como essas variáveis também estão
  associadas entre si (pessoas mais jovens tendem a ter mais escolaridade; a classe acompanha a escolaridade), **parte do
  efeito de cada uma pode ser efeito das outras**. Isso é examinado na estratificação (7.6).
* **Limitações:** distribuições descritivas, sem controle simultâneo de todos os fatores.
""")
''')
md(r"""
#### 4.2.2. Urbano × rural e o papel da escolaridade
* **Objetivo:** examinar a relação de H3 e o confundidor mais provável (escolaridade).
* **Metodologia:** composição de escolaridade por área e médias do escore por área dentro de cada nível de escolaridade.
""")
code(r'''
comp_area = pd.crosstab(habil.area, habil.grau, normalize="index").reindex(columns=ORDEM_GRAU)
display((comp_area * 100).round(1).rename_axis(index="área", columns="escolaridade (% na área)"))
media_area_grau = habil.groupby(["grau", "area"]).escore_hab.mean().unstack().reindex(ORDEM_GRAU)
n_area_grau = habil.groupby(["grau", "area"]).size().unstack().reindex(ORDEM_GRAU)

fig = go.Figure()
for area, cor in [("Urbana", COR["principal"]), ("Rural", COR["secundaria"])]:
    fig.add_bar(name=area, x=ORDEM_GRAU, y=media_area_grau[area], marker_color=cor, customdata=n_area_grau[area],
                text=media_area_grau[area].map(lambda v: fmt(v, 1)), textposition="outside",
                hovertemplate="%{x} · " + area + "<br>média = %{y:.2f}<br>n = %{customdata}<extra></extra>")
fig.add_hline(y=habil[habil.area == "Urbana"].escore_hab.mean(), line_dash="dot", line_color=COR["principal"],
              annotation_text="média geral urbana", annotation_position="top left")
fig.add_hline(y=habil[habil.area == "Rural"].escore_hab.mean(), line_dash="dot", line_color=COR["secundaria"],
              annotation_text="média geral rural", annotation_position="bottom right")
fig.update_layout(title="Escore médio de habilidades por área, dentro de cada nível de escolaridade", barmode="group",
                  height=450, yaxis_title="Escore médio (0 a 12)", xaxis_title="Escolaridade")
fig.show()
''')
code(r'''
dif_geral = habil[habil.area == "Urbana"].escore_hab.mean() - habil[habil.area == "Rural"].escore_hab.mean()
dif_estratos = (media_area_grau.Urbana - media_area_grau.Rural)
mostrar(f"""
* **Resultado:** a diferença bruta de médias (urbano − rural) é de {fmt(dif_geral)} habilidades. A população rural tem
  escolaridade mais baixa ({fmt_pct(comp_area.loc['Rural', 'Fundamental'])} com Fundamental, contra
  {fmt_pct(comp_area.loc['Urbana', 'Fundamental'])} na urbana). **Dentro** de cada nível de escolaridade, a diferença
  urbano − rural fica entre {fmt(dif_estratos.min())} e {fmt(dif_estratos.max())}.
* **Interpretação:** a escolaridade é um **confundidor**. Parte da vantagem urbana vem do fato de a população urbana ser mais
  escolarizada, mas, à primeira vista, a diferença não desaparece quando comparamos pessoas com a mesma escolaridade.
  Por isso H3 é testada também de forma **estratificada** (van Elteren).
* **Limitações:** outros fatores (idade, renda, qualidade da conexão) podem também diferir entre as áreas.
""")
''')
md(r"""
#### 4.2.3. Uso de IA por faixa etária, escolaridade e sexo
* **Objetivo:** explorar a relação de H4 e possíveis confundidores.
* **Metodologia:** proporções de uso de IA com IC 95% de Wilson por grupo; mapa de calor faixa × escolaridade.
""")
code(r'''
def prop_por_grupo(col, ordem):
    g = ia.groupby(col).IA.agg(["sum", "size"]).reindex(ordem)
    g["prop"] = g["sum"] / g["size"]
    g[["ic_inf", "ic_sup"]] = np.array([proportion_confint(s, n, method="wilson") for s, n in zip(g["sum"], g["size"])])
    return g


ia_faixa = prop_por_grupo("faixa", ORDEM_FAIXA)
ia_grau = prop_por_grupo("grau", ORDEM_GRAU)
ia_sexo = prop_por_grupo("sexo", ["Masculino", "Feminino"])
fig = make_subplots(rows=1, cols=3, shared_yaxes=True, subplot_titles=("Faixa etária", "Escolaridade", "Sexo"),
                    column_widths=[0.45, 0.35, 0.2], horizontal_spacing=0.04)
for j, g in enumerate([ia_faixa, ia_grau, ia_sexo], start=1):
    fig.add_bar(x=g.index, y=g.prop, marker_color=COR["principal"], showlegend=False, row=1, col=j,
                error_y=dict(type="data", symmetric=False, array=g.ic_sup - g.prop, arrayminus=g.prop - g.ic_inf),
                text=g.prop.map(lambda v: fmt_pct(v, 0)), textposition="outside", customdata=np.c_[g["sum"], g["size"]],
                hovertemplate="%{x}<br>%{y:.1%} (%{customdata[0]} de %{customdata[1]})<extra></extra>")
fig.update_yaxes(tickformat=".0%", range=[0, 0.7])
fig.update_layout(title="Uso de ferramentas de IA nos últimos 3 meses, entre usuários (IC 95% de Wilson)", height=450,
                  yaxis_title="Proporção que usou IA")
fig.show()

mapa = ia.pivot_table(index="faixa", columns="grau", values="IA", aggfunc="mean").reindex(index=ORDEM_FAIXA, columns=ORDEM_GRAU)
n_mapa = ia.pivot_table(index="faixa", columns="grau", values="IA", aggfunc="size").reindex(index=ORDEM_FAIXA, columns=ORDEM_GRAU)
fig = go.Figure(go.Heatmap(z=mapa.values, x=ORDEM_GRAU, y=ORDEM_FAIXA, colorscale="Blues", zmin=0, zmax=0.8,
                           colorbar=dict(title="% usou IA", tickformat=".0%"),
                           text=[[("" if pd.isna(v) else fmt_pct(v, 0)) + f"<br>n={int(n) if pd.notna(n) else 0}"
                                  for v, n in zip(lv, ln)] for lv, ln in zip(mapa.values, n_mapa.values)],
                           texttemplate="%{text}", hovertemplate="%{y} · %{x}<br>%{z:.1%}<extra></extra>"))
fig.update_layout(title="Uso de IA por faixa etária × escolaridade (proporção e n de cada célula)", height=480,
                  xaxis_title="Escolaridade", yaxis=dict(title="Faixa etária", autorange="reversed"))
fig.show()
''')
code(r'''
mostrar(f"""
* **Resultado:** o uso de IA é maior entre 16–24 anos ({fmt_pct(ia_faixa.loc['16–24', 'prop'])}) e cai nas faixas mais
  velhas, chegando a {fmt_pct(ia_faixa.loc['60+', 'prop'])} entre as pessoas de 60 anos ou mais. A faixa de 10–15 anos
  ({fmt_pct(ia_faixa.loc['10–15', 'prop'])}) fica **abaixo** da de 16–24, então o padrão não é estritamente monotônico. Por
  escolaridade, o uso vai de {fmt_pct(ia_grau.prop.iloc[0])} a {fmt_pct(ia_grau.prop.iloc[-1])}. Entre os sexos, a diferença
  é pequena ({fmt_pct(ia_sexo.loc['Masculino', 'prop'])} × {fmt_pct(ia_sexo.loc['Feminino', 'prop'])}).
* **Interpretação:** o mapa de calor mostra um caso de **confundimento**. **Dentro** de cada nível de escolaridade (exceto
  Analfabeto/Infantil, onde o uso é quase nulo), o uso de IA **cai continuamente com a idade**, e a faixa de 10–15 anos é a que
  **mais** usa: no Fundamental, {fmt_pct(mapa.loc['10–15', 'Fundamental'])} contra {fmt_pct(mapa.loc['16–24', 'Fundamental'])}
  em 16–24; no Médio, {fmt_pct(mapa.loc['10–15', 'Médio'])} contra {fmt_pct(mapa.loc['16–24', 'Médio'])}. A "queda" dos 10–15 anos
  no total vem da **composição**: a maioria deles ainda está no Fundamental, nível em que o uso é menor. Idade e escolaridade
  atuam juntas, e a comparação marginal entre faixas mistura os dois efeitos.
* **Limitações:** algumas células do mapa têm n pequeno (ver os valores de n); as proporções nessas células são instáveis.
""")
''')
md(r"""
### 4.3. Análise orientada às hipóteses: as três práticas de cidadania digital na mesma pessoa

* **Objetivo:** entender a estrutura **pareada** de H1: quantos usuários fazem cada combinação das três práticas.
* **Metodologia:** frequência das 8 combinações possíveis de (segurança, privacidade, checagem).
""")
code(r'''
combo = cidadania[["I1A_J", "I1A_K", "I1A_L"]].astype(int).astype(str).agg("".join, axis=1)
nomes_combo = {f"{j}{k}{l}": " + ".join([n for n, v in zip(["Segurança", "Privacidade", "Checagem"], (j, k, l)) if v == "1"]) or "Nenhuma"
               for j in "01" for k in "01" for l in "01"}
freq_combo = combo.map(nomes_combo).value_counts(normalize=True)
fig = go.Figure(go.Bar(x=freq_combo.values, y=freq_combo.index, orientation="h", marker_color=COR["principal"],
                       text=freq_combo.map(fmt_pct), textposition="outside",
                       hovertemplate="%{y}<br>%{x:.1%} dos usuários<extra></extra>"))
fig.update_layout(title=f"Combinações de práticas de cidadania digital na mesma pessoa (n = {fmt(len(cidadania), 0)})",
                  height=440, xaxis=dict(tickformat=".0%", title="Proporção dos usuários", range=[0, 0.6]),
                  yaxis=dict(autorange="reversed", title=""), margin=dict(l=280))
fig.show()
prop_jkl = cidadania[["I1A_J", "I1A_K", "I1A_L"]].mean()
display(prop_jkl.rename(index={"I1A_J": "Segurança (J)", "I1A_K": "Privacidade (K)", "I1A_L": "Checagem (L)"}).map(fmt_pct).to_frame("% Sim"))
''')
code(r'''
mostrar(f"""
* **Resultado:** {fmt_pct(freq_combo.get('Nenhuma', 0))} dos usuários não realizaram nenhuma das três práticas, e
  {fmt_pct(freq_combo.get('Segurança + Privacidade + Checagem', 0))} realizaram as três. A checagem de informações
  ({fmt_pct(prop_jkl['I1A_L'])}) é a mais frequente, seguida da segurança ({fmt_pct(prop_jkl['I1A_J'])}) e da privacidade
  ({fmt_pct(prop_jkl['I1A_K'])}).
* **Interpretação:** as práticas tendem a **andar juntas** (muitas pessoas fazem nenhuma ou todas), o que confirma que as
  três respostas da mesma pessoa **não são independentes**. Comparar as proporções como se viessem de grupos diferentes
  (qui-quadrado) seria um erro; o desenho exige testes **pareados**.
* **Limitações:** as diferenças entre as três proporções ainda são descritivas.
""")
''')

# =============================================================================
# 5. HIPÓTESES
# =============================================================================
md(r"""
## 5. Formulação das hipóteses estatísticas

Nível de significância em todos os testes: $\alpha = 0{,}05$. As quatro hipóteses principais formam uma família, e seus p-valores são ajustados por Holm (Seção 7.5).

> **Transparência:** as hipóteses foram definidas depois de uma exploração preliminar desta mesma base, sem pré-registro. Os testes devem ser lidos como uma confirmação interna e consistente, não como um teste confirmatório independente.

### H1: três práticas de cidadania digital (pareado, binário, k = 3)
* **Pergunta:** P1. **Justificativa:** campanhas de segurança digital e de combate à desinformação partem de diagnósticos diferentes; saber qual prática é menos difundida orienta prioridades.
* **Variáveis:** $X_{i1}$ = segurança (`I1A_J`), $X_{i2}$ = privacidade (`I1A_K`), $X_{i3}$ = checagem (`I1A_L`), todas binárias (0/1).
* **Estrutura:** **pareada** (blocos = pessoas; os 3 itens são medidos na mesma pessoa $i$).
* **Teste candidato:** Q de Cochran, com comparações par a par por McNemar e correção de Holm.

$$H_0:\ \pi_{\text{seg}} = \pi_{\text{priv}} = \pi_{\text{checagem}} \qquad\qquad H_1:\ \text{pelo menos uma proporção difere}$$

### H2: habilidades × escolaridade (4 grupos ordenados independentes)
* **Pergunta:** P2. **Justificativa:** a escolaridade é um dos principais preditores de habilidades digitais na literatura sobre exclusão digital de segundo nível.
* **Variáveis:** escore $S \in [0, 12]$ (discreto, muitos empates) e escolaridade $G \in$ {Analfabeto/Infantil < Fundamental < Médio < Superior}.
* **Estrutura:** **independente** (uma pessoa por domicílio).
* **Testes candidatos:** Kruskal-Wallis, seguido de Dunn com Holm, para a hipótese geral; Jonckheere-Terpstra para a alternativa **ordenada**.

$$H_0:\ F_1 = F_2 = F_3 = F_4 \qquad H_1^{KW}:\ \exists\, j \neq k:\ F_j \neq F_k \qquad H_1^{JT}:\ F_1 \succeq F_2 \succeq F_3 \succeq F_4 \text{ (com pelo menos uma desigualdade estrita)}$$

em que $F_j \succeq F_k$ significa que $S$ tende a ser maior no grupo $k$ (dominância estocástica).

### H3: urbano × rural, controlando a escolaridade (2 grupos independentes, estratificado)
* **Pergunta:** P3. **Justificativa:** desigualdades territoriais de infraestrutura e de oportunidades podem afetar as habilidades além da escolaridade.
* **Variáveis:** escore $S$ e área $A \in$ {Urbana, Rural}; estratos: escolaridade.
* **Testes candidatos:** Mann-Whitney U (não estratificado) e **van Elteren** (Mann-Whitney estratificado por escolaridade).

$$H_0:\ P(S_U > S_R) = \tfrac12 \text{ (em cada estrato, no van Elteren)} \qquad\qquad H_1:\ P(S_U > S_R) \neq \tfrac12$$

### H4: uso de IA × faixa etária (binária × 6 grupos ordenados)
* **Pergunta:** P4. **Justificativa:** a adoção de tecnologias novas costuma começar entre os mais jovens; resta saber quão forte é esse gradiente na IA generativa.
* **Variáveis:** IA (binária) e faixa etária (6 categorias ordenadas).
* **Testes candidatos:** qui-quadrado de independência (hipótese geral), com resíduos ajustados, e **Cochran-Armitage** (tendência linear).

$$H_0:\ \pi_{10\text{–}15} = \pi_{16\text{–}24} = \dots = \pi_{60+} \qquad\qquad H_1:\ \text{pelo menos uma proporção difere}$$
""")

# =============================================================================
# 6. SELEÇÃO DOS TESTES
# =============================================================================
md(r"""
## 6. Seleção e justificativa dos testes estatísticos

A escolha decorre do **tipo de variável** e da **estrutura de dependência** de cada hipótese, como resume a tabela abaixo. Ela não depende dos p-valores.

| | Pareado (mesma pessoa) | Independente |
|---|---|---|
| **Binário** | Cochran Q + McNemar (H1) | Qui-quadrado + Cochran-Armitage (H4) |
| **Escore discreto/ordinal** | — | Kruskal-Wallis + Dunn + Jonckheere-Terpstra (H2); Mann-Whitney + van Elteren (H3) |

### 6.1. Q de Cochran e McNemar (H1)
1. **Por que é adequado?** São $k = 3$ respostas binárias **da mesma pessoa**. O Q de Cochran generaliza o McNemar para $k > 2$ condições pareadas.
2. **Pressupostos:** blocos (pessoas) independentes entre si; resposta binária; número suficiente de blocos **não constantes** (pessoas que não responderam igual nos três itens) para a aproximação qui-quadrado.
3. **Verificação:** uma pessoa por domicílio. O número de blocos não constantes é calculado abaixo.
4. **Estatística:** com $C_j$ = total de "Sim" no item $j$, $R_i$ = total de "Sim" da pessoa $i$ e $N = \sum_j C_j$:
$$Q = \frac{(k-1)\left[k\sum_{j=1}^{k} C_j^2 - N^2\right]}{kN - \sum_{i=1}^{n} R_i^2}$$
5. **p-valor:** $Q \sim \chi^2_{k-1}$ sob $H_0$.
6. **Decisão:** rejeitar $H_0$ se $p < \alpha$ e, então, localizar as diferenças com McNemar par a par (Holm).
7. **Interpretação:** pelo menos uma prática tem prevalência diferente das demais **na mesma população de pessoas**.
8. **Efeito e IC:** $\eta^2_Q = Q / [n(k-1)]$ para o teste global (Tomczak & Tomczak, 2014). Par a par: diferença de proporções pareadas com IC de Wald pareado e razão de chances pareada $b/c$.

### 6.2. Kruskal-Wallis, Dunn e Jonckheere-Terpstra (H2)
1. **Adequação:** o escore é discreto, limitado, assimétrico e cheio de zeros, então a ANOVA (normalidade, variâncias iguais) não se aplica.
2. **Pressupostos:** observações independentes; variável pelo menos ordinal. Para ler o resultado como diferença de medianas, as distribuições precisariam ter a mesma forma.
3. **Verificação:** a EDA mostrou formas muito diferentes entre os grupos (zeros concentrados na baixa escolaridade), então a leitura é de **dominância estocástica**.
4. **Estatísticas:**
$$H = \frac{\frac{12}{N(N+1)}\sum_j n_j \bar R_j^{\,2} - 3(N+1)}{1 - \frac{\sum (t^3-t)}{N^3-N}} \qquad\qquad J = \sum_{j<k} U_{jk},\quad U_{jk} = \#\{(x,y): x \in G_j,\ y \in G_k,\ x<y\} + \tfrac12 \#\{x=y\}$$
5. **p-valor:** $H \sim \chi^2_{k-1}$; $J$ padronizado com variância exata corrigida para empates $\to N(0,1)$.
6. **Decisão:** rejeitar $H_0$ se $p < \alpha$.
7. **Interpretação:** o Kruskal-Wallis detecta **qualquer** diferença; o Jonckheere-Terpstra é mais poderoso quando a alternativa é uma **tendência** na ordem prevista.
8. **Efeito e IC:** $\varepsilon^2 = H/(N-1)$ com IC bootstrap; probabilidade de superioridade par a par.

### 6.3. Mann-Whitney U e van Elteren (H3)
1. **Adequação:** dois grupos independentes, desfecho discreto com empates. O van Elteren combina testes de Mann-Whitney **dentro de estratos**, controlando um confundidor categórico sem supor nenhum modelo.
2. **Pressupostos:** observações independentes; no van Elteren, o efeito deve ter a **mesma direção** nos estratos (caso contrário, os efeitos se cancelam).
3. **Verificação:** a EDA (4.2.2) e os Mann-Whitney por estrato (abaixo) mostram a direção em cada nível de escolaridade.
4. **Estatística:** $U = W_1 - \frac{n_1(n_1+1)}{2}$. No van Elteren: $T = \sum_s \frac{W_s}{N_s + 1}$, em que $W_s$ é a soma de postos do grupo urbano no estrato $s$.
5. **p-valor:** aproximação normal com correção de empates, nos dois casos.
6. **Decisão:** rejeitar $H_0$ se $p < \alpha$.
7. **Interpretação:** se o Mann-Whitney rejeita $H_0$ e o van Elteren também, a diferença urbano × rural **não é explicada só** pela escolaridade.
8. **Efeito e IC:** probabilidade de superioridade $\hat{P}(S_U > S_R) + \tfrac12 \hat P(S_U = S_R) = U/(n_1 n_2)$ e correlação bisserial de postos $r_{rb} = 2\,PS - 1$, com IC bootstrap.

### 6.4. Qui-quadrado, resíduos ajustados e Cochran-Armitage (H4)
1. **Adequação:** desfecho binário em 6 grupos independentes **ordenados**.
2. **Pressupostos:** observações independentes; contagens esperadas ≥ 5.
3. **Verificação:** as contagens esperadas são calculadas abaixo.
4. **Estatísticas:** $\chi^2 = \sum \frac{(O-E)^2}{E}$ com $(r-1)(c-1) = 5$ g.l. Cochran-Armitage, com escores $s_i$ e $\bar p$ a proporção geral:
$$Z_{CA} = \frac{\sum_i x_i (s_i - \bar s)}{\sqrt{\bar p (1-\bar p) \sum_i n_i (s_i - \bar s)^2}}$$
5. **p-valor:** $\chi^2_5$; $Z_{CA} \sim N(0,1)$.
6. **Decisão:** rejeitar $H_0$ se $p < \alpha$.
7. **Interpretação:** o qui-quadrado detecta qualquer associação; os **resíduos ajustados** (> |2|) mostram quais faixas se afastam da independência; o Cochran-Armitage testa a tendência linear. Ele pode ser significativo mesmo com um padrão não estritamente monotônico.
8. **Efeito e IC:** V de Cramér; diferença de proporções entre faixas extremas com IC de Newcombe.

### 6.5. E o plano amostral complexo?
Os testes acima supõem **observações independentes e identicamente distribuídas**. A TIC Domicílios usa estratos, conglomerados (UPA) e pesos desiguais. Sem pesos, as estimativas descrevem a amostra; ignorar os conglomerados tende a **subestimar** os erros-padrão.

A estratégia tem **duas camadas**, como no estudo da TIC Kids Online:
1. **Camada clássica (não paramétrica):** os testes acima, sem pesos.
2. **Camada populacional (por desenho):** para cada hipótese, estimamos uma quantidade análoga com `PESO` e obtemos o erro-padrão pelos **200 pesos replicados**, que incorporam estratos e conglomerados. Daí saem um IC de 95% e um teste de Wald.

| Hipótese | Quantidade estimada por desenho |
|---|---|
| H1 | Proporções ponderadas das 3 práticas; Wald de igualdade (2 g.l.) e diferenças par a par |
| H2 | Escore médio ponderado por escolaridade; Wald de igualdade (3 g.l.) |
| H3 | Diferença urbano − rural no escore médio, bruta e **padronizada pela escolaridade** |
| H4 | Proporção ponderada de uso de IA por faixa; Wald de igualdade (5 g.l.) |
""")

# =============================================================================
# 7. EXECUÇÃO
# =============================================================================
md(r"""
## 7. Execução dos testes de hipóteses

### 7.1. H1: Q de Cochran e McNemar par a par

**A · Contextualização.** Testamos se segurança, privacidade e checagem de informações são praticadas com a mesma frequência **pelas mesmas pessoas**.

**B · Fundamentação.** Ver 6.1. Pessoas que responderam igual nos três itens (todos 0 ou todos 1) não contribuem para $Q$: só os blocos não constantes informam sobre diferenças entre os itens.

**C · Preparação.** Usamos `cidadania`: usuários com os três itens válidos.
""")
code(r'''
# D · Implementação
X = cidadania[["I1A_J", "I1A_K", "I1A_L"]].to_numpy(int)
n_blocos, k = X.shape
R, Cj, N_sim = X.sum(axis=1), X.sum(axis=0), X.sum()
Q_manual = (k - 1) * (k * (Cj**2).sum() - N_sim**2) / (k * N_sim - (R**2).sum())
q_ref = cochrans_q(X)
assert np.isclose(Q_manual, q_ref.statistic), "Implementação manual do Q diverge do statsmodels"
nao_constantes = int(((R > 0) & (R < k)).sum())
eta2_q = Q_manual / (n_blocos * (k - 1))

NOMES_H1 = {"I1A_J": "Segurança", "I1A_K": "Privacidade", "I1A_L": "Checagem"}
pares_h1 = []
for a, c in [("I1A_L", "I1A_J"), ("I1A_J", "I1A_K"), ("I1A_L", "I1A_K")]:
    r = resumo_mcnemar(cidadania[a], cidadania[c])
    pares_h1.append(dict(comparacao=f"{NOMES_H1[a]} − {NOMES_H1[c]}", b=r["b"], c=r["c"], qui2=r["qui2"], p=r["p"],
                         dif=r["dif"], dif_inf=r["dif_ic"][0], dif_sup=r["dif_ic"][1],
                         razao=r["razao"], razao_inf=r["razao_ic"][0], razao_sup=r["razao_ic"][1]))
pares_h1 = pd.DataFrame(pares_h1)
pares_h1["p_holm"] = multipletests(pares_h1.p, method="holm")[1]

# E · Resultados
res_h1 = pd.Series({
    "Pessoas (blocos)": n_blocos, "Blocos não constantes": nao_constantes, "k (itens)": k,
    "% Sim · segurança / privacidade / checagem": " / ".join(fmt_pct(v) for v in X.mean(axis=0)),
    "Q de Cochran": fmt(Q_manual, 1), "g.l.": k - 1, "p-valor": fmt_p(q_ref.pvalue), "α": fmt(ALPHA),
    "η²_Q": fmt(eta2_q, 3), "Decisão": "Rejeitar H0" if q_ref.pvalue < ALPHA else "Não rejeitar H0",
})
display(res_h1.to_frame("H1 · Q de Cochran"))
display(pares_h1.style.format({"qui2": "{:.1f}", "p": "{:.1e}", "p_holm": "{:.1e}", "dif": "{:+.3f}", "dif_inf": "{:+.3f}",
                               "dif_sup": "{:+.3f}", "razao": "{:.2f}", "razao_inf": "{:.2f}", "razao_sup": "{:.2f}"}).hide(axis="index"))
''')
code(r'''
mostrar(f"""
**F · Interpretação.** Com {fmt(n_blocos, 0)} pessoas, das quais {fmt(nao_constantes, 0)} responderam de forma diferente
entre os itens (bem acima do necessário para a aproximação), o Q de Cochran é {fmt(Q_manual, 1)} (g.l. = 2,
{fmt_p(q_ref.pvalue)}). **Rejeitamos $H_0$**: as três práticas não têm a mesma prevalência entre os usuários.

As comparações par a par (McNemar, Holm) mostram a **ordem** checagem > segurança > privacidade, com todas as diferenças
significativas: checagem − segurança = {fmt(100 * pares_h1.dif.iloc[0], 1)} p.p.; segurança − privacidade =
{fmt(100 * pares_h1.dif.iloc[1], 1)} p.p.; checagem − privacidade = {fmt(100 * pares_h1.dif.iloc[2], 1)} p.p.

O tamanho de efeito global é pequeno (η²_Q = {fmt(eta2_q, 3)}): as práticas estão **fortemente correlacionadas** na
mesma pessoa, e a maior parte da variação é *entre pessoas* (quem faz uma tende a fazer as outras), não *entre práticas*.
Em termos práticos, **ajustar a privacidade é a prática menos difundida**: entre os pares discordantes, a configuração
"checou mas não ajustou a privacidade" é {fmt(pares_h1.razao.iloc[2], 1)} vezes mais comum que a inversa.
""")
''')

md(r"""
### 7.2. H2: Kruskal-Wallis, Dunn e Jonckheere-Terpstra

**A · Contextualização.** Testamos se o número de habilidades digitais difere entre os níveis de escolaridade e se cresce com eles.

**B · Fundamentação.** Ver 6.2.

**C · Preparação.** Usamos `habil`: usuários com o escore de habilidades válido.
""")
code(r'''
# D · Implementação
grupos_h2 = [habil.loc[habil.grau == g, "escore_hab"].to_numpy() for g in ORDEM_GRAU]
kw = stats.kruskal(*grupos_h2)
N2 = len(habil)
eps2 = epsilon_quadrado(kw.statistic, N2)
cod_grau = habil.GRAU_INST_1.to_numpy()
eps2_ic = ic_bootstrap(lambda v, g: epsilon_quadrado(stats.kruskal(*[v[g == j] for j in np.unique(g)]).statistic, len(v)),
                       habil.escore_hab.to_numpy(), cod_grau, B=300)
jt = jonckheere_terpstra(grupos_h2)
tau = stats.kendalltau(cod_grau, habil.escore_hab)   # JT é equivalente a um tau de Kendall entre grupo e resposta
assert np.sign(jt["z"]) == np.sign(tau.statistic)
# validação da implementação: numa subamostra (p-valores não nulos), o p bilateral do JT deve coincidir com o do tau de Kendall
_sub = habil[habil.FAIXA_ETARIA == 1]
_jt = jonckheere_terpstra([_sub.loc[_sub.GRAU_INST_1 == j, "escore_hab"] for j in sorted(_sub.GRAU_INST_1.unique())])
_tau = stats.kendalltau(_sub.GRAU_INST_1, _sub.escore_hab)
assert np.isclose(_jt["p_bilateral"], _tau.pvalue, rtol=1e-3), (_jt["p_bilateral"], _tau.pvalue)
print(f"Validação JT × τ de Kendall (subamostra 10–15 anos): p_JT = {_jt['p_bilateral']:.3e} | p_τ = {_tau.pvalue:.3e}")

res_h2 = pd.Series({
    "N": N2, "Tamanho dos grupos": ", ".join(f"{g}: {len(x)}" for g, x in zip(ORDEM_GRAU, grupos_h2)),
    "Medianas": ", ".join(f"{g}: {fmt(np.median(x), 1)}" for g, x in zip(ORDEM_GRAU, grupos_h2)),
    "H (Kruskal-Wallis, corrigido para empates)": fmt(kw.statistic, 1), "g.l.": 3, "p-valor (KW)": fmt_p(kw.pvalue),
    "ε² [IC 95% bootstrap]": f"{fmt(eps2, 3)} [{fmt(eps2_ic[0], 3)}; {fmt(eps2_ic[1], 3)}]",
    "J (Jonckheere-Terpstra)": fmt(jt["J"], 0), "E[J] sob H0": fmt(jt["media"], 0), "z (JT)": fmt(jt["z"], 1),
    "p-valor (JT, unilateral: tendência crescente)": fmt_p(jt["p_unilateral"]),
    "τ-b de Kendall (escolaridade × escore)": fmt(tau.statistic, 3),
    "Decisão": "Rejeitar H0" if kw.pvalue < ALPHA else "Não rejeitar H0",
})
display(res_h2.to_frame("H2 · Kruskal-Wallis + Jonckheere-Terpstra"))

dunn_h2 = teste_dunn(habil.escore_hab, habil.grau, ordem=ORDEM_GRAU)
dunn_h2["P(2º grupo > 1º)"] = [stats.mannwhitneyu(habil.escore_hab[habil.grau == c.split(" × ")[1]],
                                                  habil.escore_hab[habil.grau == c.split(" × ")[0]]).statistic /
                               ((habil.grau == c.split(" × ")[1]).sum() * (habil.grau == c.split(" × ")[0]).sum())
                               for c in dunn_h2.comparacao]
display(dunn_h2.style.format({"posto_medio_1": "{:.0f}", "posto_medio_2": "{:.0f}", "z": "{:.1f}", "p": "{:.1e}",
                              "p_holm": "{:.1e}", "P(2º grupo > 1º)": "{:.2f}"}).hide(axis="index"))
''')
code(r'''
sig2 = (dunn_h2.p_holm < ALPHA).sum()
mostrar(f"""
**E/F · Interpretação.** O Kruskal-Wallis rejeita $H_0$ (H = {fmt(kw.statistic, 1)}, g.l. = 3, {fmt_p(kw.pvalue)}), e o
Jonckheere-Terpstra confirma uma **tendência crescente** na ordem prevista (z = {fmt(jt['z'], 1)}, {fmt_p(jt['p_unilateral'])}).
O efeito é **forte**: ε² = {fmt(eps2, 3)} (IC 95% {fmt(eps2_ic[0], 3)} a {fmt(eps2_ic[1], 3)}); por convenção, valores a partir
de 0,16 já são considerados relativamente fortes. No teste de Dunn, {sig2} das 6 comparações são significativas após Holm.
Na comparação extrema, a probabilidade de um usuário com ensino superior ter **mais** habilidades que um com Fundamental é
{fmt(dunn_h2.loc[dunn_h2.comparacao == 'Fundamental × Superior', 'P(2º grupo > 1º)'].iloc[0])} (0,5 = nenhuma diferença).

Como as distribuições têm formas muito diferentes (muitos zeros na baixa escolaridade), lemos o resultado como
**dominância estocástica**. Como a escolaridade está associada à idade e à classe, **não podemos atribuir o efeito
apenas à escolaridade**; a estratificação por idade (7.6) verifica se ele persiste dentro de cada faixa.
""")
''')

md(r"""
### 7.3. H3: Mann-Whitney U e van Elteren (urbano × rural)

**A · Contextualização.** Testamos se usuários urbanos tendem a ter mais habilidades que os rurais, inclusive entre pessoas com a mesma escolaridade.

**B · Fundamentação.** Ver 6.3.

**C · Preparação.** `habil`, dividido por área; estratos = 4 níveis de escolaridade.
""")
code(r'''
# D · Implementação
urb = habil.loc[habil.area == "Urbana", "escore_hab"].to_numpy()
rur = habil.loc[habil.area == "Rural", "escore_hab"].to_numpy()
mw = stats.mannwhitneyu(urb, rur, alternative="two-sided", method="asymptotic")
ps = mw.statistic / (len(urb) * len(rur))
rbc_area = 2 * ps - 1
es_area = habil.escore_hab.to_numpy()
urb_mask = (habil.area == "Urbana").to_numpy()


def rbc_mw(v, g):
    a, c = v[g], v[~g]
    return 2 * stats.mannwhitneyu(a, c).statistic / (len(a) * len(c)) - 1


rbc_ic = ic_bootstrap(rbc_mw, es_area, urb_mask, B=500)
ve = van_elteren(habil.escore_hab, urb_mask, habil.GRAU_INST_1)

por_estrato = []
for g in ORDEM_GRAU:
    sub = habil[habil.grau == g]
    a, c = sub.loc[sub.area == "Urbana", "escore_hab"], sub.loc[sub.area == "Rural", "escore_hab"]
    m = stats.mannwhitneyu(a, c, alternative="two-sided")
    por_estrato.append(dict(escolaridade=g, n_urbano=len(a), n_rural=len(c), PS=m.statistic / (len(a) * len(c)), p=m.pvalue))
por_estrato = pd.DataFrame(por_estrato)

res_h3 = pd.Series({
    "n urbano / rural": f"{fmt(len(urb), 0)} / {fmt(len(rur), 0)}",
    "Mediana urbano / rural": f"{fmt(np.median(urb), 1)} / {fmt(np.median(rur), 1)}",
    "U (Mann-Whitney)": fmt(mw.statistic, 0), "p-valor (MW)": fmt_p(mw.pvalue),
    "Probabilidade de superioridade P(S_U > S_R)": fmt(ps, 3),
    "r_rb [IC 95% bootstrap]": f"{fmt(rbc_area, 3)} [{fmt(rbc_ic[0], 3)}; {fmt(rbc_ic[1], 3)}]",
    "z (van Elteren, estratificado por escolaridade)": fmt(ve["z"], 2), "p-valor (van Elteren)": fmt_p(ve["p"]),
    "Decisão (MW e van Elteren)": "Rejeitar H0" if max(mw.pvalue, ve["p"]) < ALPHA else "Ver detalhes",
})
display(res_h3.to_frame("H3 · Mann-Whitney e van Elteren"))
display(por_estrato.style.format({"PS": "{:.3f}", "p": "{:.1e}"}).hide(axis="index"))
''')
code(r'''
mostrar(f"""
**E/F · Interpretação.** Sem estratificar, o Mann-Whitney rejeita $H_0$ ({fmt_p(mw.pvalue)}): a probabilidade de um usuário
urbano ter mais habilidades que um rural é {fmt(ps, 2)} ($r_{{rb}}$ = {fmt(rbc_area, 2)}, IC 95% {fmt(rbc_ic[0], 2)} a
{fmt(rbc_ic[1], 2)}), um efeito **pequeno a moderado**. O **van Elteren**, que compara urbanos e rurais **dentro** de cada
nível de escolaridade, também rejeita $H_0$ (z = {fmt(ve['z'], 1)}, {fmt_p(ve['p'])}).

Nos testes por estrato, a vantagem urbana aparece em todos os níveis, com PS entre {fmt(por_estrato.PS.min(), 2)} e
{fmt(por_estrato.PS.max(), 2)}. O estrato Analfabeto/Infantil é a exceção: lá quase ninguém tem habilidades, e o teste não
é significativo. **Conclusão:** a diferença urbano × rural é **menor** do que a comparação bruta sugere, porque parte dela
é escolaridade, mas **não desaparece** quando a escolaridade é controlada.
""")
''')

md(r"""
### 7.4. H4: qui-quadrado, resíduos ajustados e Cochran-Armitage (IA × faixa etária)

**A · Contextualização.** Testamos se o uso de IA é independente da faixa etária e se há tendência ao longo das idades.

**B · Fundamentação.** Ver 6.4.

**C · Preparação.** `ia`: usuários com `C13A` igual a 0 ou 1.
""")
code(r'''
# D · Implementação
tab_h4 = pd.crosstab(ia.faixa, ia.IA).reindex(ORDEM_FAIXA)[[1, 0]].rename(columns={1: "Usou IA", 0: "Não usou"})
chi = stats.chi2_contingency(tab_h4, correction=False)
v_cramer = np.sqrt(chi.statistic / (tab_h4.values.sum() * (min(tab_h4.shape) - 1)))
res_aj = pd.DataFrame(residuos_ajustados(tab_h4.values), index=tab_h4.index, columns=tab_h4.columns)
ca = cochran_armitage(tab_h4["Usou IA"], tab_h4.sum(axis=1))
ca_sem_10_15 = cochran_armitage(tab_h4["Usou IA"].iloc[1:], tab_h4.sum(axis=1).iloc[1:])

from statsmodels.stats.proportion import confint_proportions_2indep
x1, n1 = tab_h4.loc["16–24", "Usou IA"], tab_h4.loc["16–24"].sum()
x2, n2_ = tab_h4.loc["60+", "Usou IA"], tab_h4.loc["60+"].sum()
ic_dif_h4 = confint_proportions_2indep(x1, n1, x2, n2_, method="newcomb")

display(pd.concat([tab_h4, (tab_h4["Usou IA"] / tab_h4.sum(axis=1)).map(fmt_pct).rename("% usou"),
                   res_aj["Usou IA"].round(1).rename("resíduo ajustado")], axis=1))
res_h4 = pd.Series({
    "n": int(tab_h4.values.sum()), "Menor contagem esperada": fmt(chi.expected_freq.min(), 1),
    "χ² (g.l. = 5)": fmt(chi.statistic, 1), "p-valor (χ²)": fmt_p(chi.pvalue), "V de Cramér": fmt(v_cramer, 3),
    "Cochran-Armitage z (6 faixas)": fmt(ca["z"], 1), "p-valor (CA)": fmt_p(ca["p"]),
    "Cochran-Armitage z (sem 10–15)": fmt(ca_sem_10_15["z"], 1),
    "Diferença 16–24 − 60+ [IC 95% Newcombe]": f"{fmt(100 * (x1 / n1 - x2 / n2_), 1)} p.p. [{fmt(100 * ic_dif_h4[0], 1)}; {fmt(100 * ic_dif_h4[1], 1)}]",
    "Decisão": "Rejeitar H0" if chi.pvalue < ALPHA else "Não rejeitar H0",
})
display(res_h4.to_frame("H4 · Qui-quadrado + Cochran-Armitage"))
''')
code(r'''
pos = res_aj.index[res_aj["Usou IA"] > 2].tolist()
neg = res_aj.index[res_aj["Usou IA"] < -2].tolist()
mostrar(f"""
**E/F · Interpretação.** Todas as contagens esperadas são grandes, então a aproximação é adequada. O qui-quadrado rejeita
a independência (χ² = {fmt(chi.statistic, 1)}, g.l. = 5, {fmt_p(chi.pvalue)}), com **V de Cramér = {fmt(v_cramer, 2)}**, uma
associação de magnitude moderada. Os resíduos ajustados mostram **excesso** de uso nas faixas {', '.join(pos)} e **falta**
de uso nas faixas {', '.join(neg)}. O Cochran-Armitage indica uma tendência **decrescente** com a idade (z = {fmt(ca['z'], 1)},
{fmt_p(ca['p'])}).

No total, o padrão **não é estritamente monotônico**: a faixa de 10–15 anos usa menos IA que a de 16–24. Sem a faixa de
10–15, a tendência decrescente é ainda mais nítida (z = {fmt(ca_sem_10_15['z'], 1)}). Como a EDA (4.2.3) indicou, essa
quebra é um efeito de **composição por escolaridade**; a análise de sensibilidade (7.6c) testa a tendência dentro de cada
nível de escolaridade.
A diferença entre 16–24 e 60+ é de {fmt(100 * (x1 / n1 - x2 / n2_), 1)} p.p. (IC 95% {fmt(100 * ic_dif_h4[0], 1)} a
{fmt(100 * ic_dif_h4[1], 1)}).
""")
''')

md(r"""
### 7.5. Correção para múltiplas comparações

**Por que?** Com 4 testes principais a 5%, a chance de pelo menos um falso positivo, se todas as $H_0$ forem verdadeiras, chega a cerca de 18,5%. O método de **Holm** controla a taxa de erro familiar (FWER) em 5%. Os testes post hoc (McNemar par a par, Dunn) já têm seu próprio ajuste de Holm dentro de cada hipótese.
""")
code(r'''
familia = pd.DataFrame({
    "hipótese": ["H1 · Q de Cochran", "H2 · Kruskal-Wallis", "H3 · van Elteren (estratificado)", "H4 · Qui-quadrado"],
    "p bruto": [q_ref.pvalue, kw.pvalue, ve["p"], chi.pvalue],
})
familia["p ajustado (Holm)"] = multipletests(familia["p bruto"], method="holm")[1]
familia["decisão (α = 0,05)"] = np.where(familia["p ajustado (Holm)"] < ALPHA, "Rejeitar H0", "Não rejeitar H0")
display(familia.style.format({"p bruto": "{:.2e}", "p ajustado (Holm)": "{:.2e}"}).hide(axis="index"))
''')
md(r"""
**Interpretação.** Os quatro p-valores são extremamente pequenos, então o ajuste não muda nenhuma decisão. Com quase 20 mil usuários, isso era esperado; por isso a interpretação se apoia nos **tamanhos de efeito**: η²_Q pequeno em H1, ε² forte em H2, efeito pequeno a moderado em H3 e V moderado em H4.

### 7.6. Robustez e sensibilidade
**O que vamos fazer?**
* **(a) H2 estratificado por idade:** o efeito da escolaridade persiste dentro de cada faixa etária?
* **(b) H4 com o "Não sabe" recodificado como "Não":** a conclusão depende da exclusão dos 97?
* **(c) H4 estratificado por escolaridade:** a tendência etária no uso de IA persiste dentro de cada nível de escolaridade, inclusive na faixa de 10–15 anos?
""")
code(r'''
estr_h2 = []
for f in ORDEM_FAIXA:
    sub = habil[habil.faixa == f]
    grupos = [sub.loc[sub.grau == g, "escore_hab"] for g in ORDEM_GRAU if (sub.grau == g).sum() >= 10]
    k_ = stats.kruskal(*grupos)
    jt_ = jonckheere_terpstra(grupos)
    estr_h2.append(dict(faixa=f, n=len(sub), grupos=len(grupos), H=k_.statistic, p_KW=k_.pvalue,
                        eps2=epsilon_quadrado(k_.statistic, len(sub)), z_JT=jt_["z"]))
estr_h2 = pd.DataFrame(estr_h2)
display(estr_h2.style.format({"H": "{:.1f}", "p_KW": "{:.1e}", "eps2": "{:.3f}", "z_JT": "{:.1f}"}).hide(axis="index"))

sens = usuarios[usuarios.FAIXA_ETARIA.notna()].copy()
sens["IA_sens"] = dados_brutos.loc[sens.index, "C13A"].replace({97: 0}).where(lambda s: s.isin([0, 1]))
tab_sens = pd.crosstab(sens.faixa, sens.IA_sens).reindex(ORDEM_FAIXA)[[1.0, 0.0]]
chi_sens = stats.chi2_contingency(tab_sens, correction=False)
ca_sens = cochran_armitage(tab_sens.iloc[:, 0], tab_sens.sum(axis=1))
print(f"H4 com 97 → 'Não': χ² = {fmt(chi_sens.statistic, 1)} ({fmt_p(chi_sens.pvalue)}), "
      f"V = {fmt(np.sqrt(chi_sens.statistic / tab_sens.values.sum()), 3)}, Cochran-Armitage z = {fmt(ca_sens['z'], 1)}")
print("% usou IA por faixa (sensibilidade):", {f: fmt_pct(v) for f, v in (tab_sens.iloc[:, 0] / tab_sens.sum(axis=1)).items()})

estr_h4 = []
for g in ["Fundamental", "Médio", "Superior"]:          # Analfabeto/Infantil: uso praticamente nulo, sem variação
    sub = ia[ia.grau == g]
    t_ = pd.crosstab(sub.faixa, sub.IA).reindex([f for f in ORDEM_FAIXA if f in set(sub.faixa)])
    t_ = t_[t_.sum(axis=1) >= 30]                         # faixas com n mínimo para uma proporção estável
    ca_ = cochran_armitage(t_[1], t_.sum(axis=1))
    estr_h4.append(dict(escolaridade=g, n=int(t_.values.sum()), faixas=", ".join(t_.index),
                        **{"% usou IA (da mais jovem à mais velha)": " → ".join(fmt_pct(v, 0) for v in t_[1] / t_.sum(axis=1))},
                        z_CA=ca_["z"], p_CA=ca_["p"]))
estr_h4 = pd.DataFrame(estr_h4)
display(estr_h4.style.format({"z_CA": "{:.1f}", "p_CA": "{:.1e}"}).hide(axis="index"))
''')
code(r'''
mostrar(f"""
**Interpretação.**
* **(a)** O efeito da escolaridade sobre as habilidades aparece **dentro de todas as faixas etárias** (ε² entre
  {fmt(estr_h2.eps2.min(), 2)} e {fmt(estr_h2.eps2.max(), 2)}; JT com z positivo em todas). A idade não explica o resultado de H2.
* **(b)** Recodificar o "Não sabe" como "Não" praticamente não muda H4 (V = {fmt(np.sqrt(chi_sens.statistic / tab_sens.values.sum()), 2)}
  contra {fmt(v_cramer, 2)} na análise principal). A ausência não aleatória identificada na Seção 3.4 não altera a conclusão.
* **(c)** Dentro do Fundamental, do Médio e do Superior, o uso de IA **cai monotonicamente com a idade** (Cochran-Armitage
  z entre {fmt(estr_h4.z_CA.min(), 1)} e {fmt(estr_h4.z_CA.max(), 1)}), e a faixa mais jovem de cada estrato é a que mais usa.
  A tendência etária de H4 **não é explicada pela escolaridade**; ao contrário, controlar a escolaridade elimina a exceção
  dos 10–15 anos.
""")
''')

md(r"""
### 7.7. Camada populacional: estimativas com os pesos replicados

**O que vamos fazer?** Reestimar cada hipótese com `PESO` e obter erros-padrão pelos 200 pesos replicados (ver 6.5).

**Por que?** Para saber se as conclusões valem para a **população** de usuários de Internet de 10 anos ou mais.

**Como?** Para uma estimativa $\hat\theta$, calculamos $\hat\theta_r$ com cada réplica $r = 1, \dots, 200$:
$$\widehat{\mathrm{Var}}(\hat\theta) = \frac{1}{R-1}\sum_{r=1}^{R}\left(\hat\theta_r - \bar{\hat\theta}\right)^2.$$
Para vetores de estimativas (por exemplo, médias de 4 grupos), a matriz de covariância das réplicas alimenta um teste de Wald de igualdade.
""")
code(r'''
idx_c, idx_h, idx_i = cidadania.index.to_numpy(), habil.index.to_numpy(), ia.index.to_numpy()

# H1: proporções ponderadas das três práticas (mesmas pessoas)
pop_h1 = estimativa_com_replicas(lambda w: np.array([media_ponderada(cidadania[c], w[idx_c]) for c in ["I1A_J", "I1A_K", "I1A_L"]]), base)
wald_h1 = wald_igualdade(pop_h1["estimativa"], pop_h1["replicas"])

# H2: escore médio ponderado por escolaridade
gh = habil.GRAU_INST_1.to_numpy()
pop_h2 = estimativa_com_replicas(lambda w: np.array([media_ponderada(habil.escore_hab, w[idx_h], gh == j) for j in (1, 2, 3, 4)]), base)
wald_h2 = wald_igualdade(pop_h2["estimativa"], pop_h2["replicas"])


# H3: diferença urbano − rural, bruta e padronizada pela escolaridade (pesos = distribuição ponderada de escolaridade)
def dif_area(w, padronizar):
    ww = w[idx_h]
    if not padronizar:
        return media_ponderada(habil.escore_hab, ww, urb_mask) - media_ponderada(habil.escore_hab, ww, ~urb_mask)
    total = 0.0
    for j in (1, 2, 3, 4):
        m = gh == j
        peso_estrato = ww[m].sum() / ww.sum()
        total += peso_estrato * (media_ponderada(habil.escore_hab, ww, m & urb_mask) - media_ponderada(habil.escore_hab, ww, m & ~urb_mask))
    return total


pop_h3_bruta = estimativa_com_replicas(lambda w: dif_area(w, False), base)
pop_h3_padr = estimativa_com_replicas(lambda w: dif_area(w, True), base)

# H4: proporção ponderada de uso de IA por faixa
fi = ia.FAIXA_ETARIA.to_numpy()
pop_h4 = estimativa_com_replicas(lambda w: np.array([media_ponderada(ia.IA, w[idx_i], fi == j) for j in range(1, 7)]), base)
wald_h4 = wald_igualdade(pop_h4["estimativa"], pop_h4["replicas"])

linhas_pop = []
for nome, est, ic_i, ic_s, sem in zip(["Segurança", "Privacidade", "Checagem"], pop_h1["estimativa"], pop_h1["ic_inf"], pop_h1["ic_sup"], X.mean(axis=0)):
    linhas_pop.append(("H1", f"% {nome}", sem, est, ic_i, ic_s))
for g, est, ic_i, ic_s, x in zip(ORDEM_GRAU, pop_h2["estimativa"], pop_h2["ic_inf"], pop_h2["ic_sup"], grupos_h2):
    linhas_pop.append(("H2", f"Escore médio · {g}", x.mean(), est, ic_i, ic_s))
linhas_pop.append(("H3", "Urbano − rural (bruta)", urb.mean() - rur.mean(), pop_h3_bruta["estimativa"][0], pop_h3_bruta["ic_inf"][0], pop_h3_bruta["ic_sup"][0]))
linhas_pop.append(("H3", "Urbano − rural (padronizada p/ escolaridade)", np.nan, pop_h3_padr["estimativa"][0], pop_h3_padr["ic_inf"][0], pop_h3_padr["ic_sup"][0]))
for f, est, ic_i, ic_s in zip(ORDEM_FAIXA, pop_h4["estimativa"], pop_h4["ic_inf"], pop_h4["ic_sup"]):
    linhas_pop.append(("H4", f"% usou IA · {f}", tab_h4.loc[f, "Usou IA"] / tab_h4.loc[f].sum(), est, ic_i, ic_s))
camada_pop = pd.DataFrame(linhas_pop, columns=["hipótese", "quantidade", "sem peso", "com peso", "IC95% inf (desenho)", "IC95% sup (desenho)"])
display(camada_pop.style.format({c: "{:.3f}" for c in camada_pop.columns[2:]}, na_rep="—").hide(axis="index"))

testes_pop = pd.DataFrame([
    ("H1", "Igualdade das 3 proporções", *wald_h1),
    ("H2", "Igualdade das 4 médias", *wald_h2),
    ("H3", "Diferença padronizada = 0", float(pop_h3_padr["estimativa"][0] / pop_h3_padr["ep"][0]) ** 2, 1, float(pop_h3_padr["p"][0])),
    ("H4", "Igualdade das 6 proporções", *wald_h4),
], columns=["hipótese", "teste de Wald por desenho", "X²", "g.l.", "p"])
display(testes_pop.style.format({"X²": "{:.1f}", "p": "{:.1e}"}).hide(axis="index"))
''')
code(r'''
p_seg, p_priv, p_chec = pop_h1["estimativa"]
dif_lj = p_chec - p_seg
ep_lj = (pop_h1["replicas"][:, 2] - pop_h1["replicas"][:, 0]).std(ddof=1)
mostrar(f"""
**Interpretação.**
* **H1 se mantém na população:** checagem ({fmt_pct(p_chec)}) > segurança ({fmt_pct(p_seg)}) > privacidade
  ({fmt_pct(p_priv)}), com o Wald de igualdade rejeitando $H_0$ ({fmt_p(wald_h1[2])}). Com pesos, a distância entre checagem
  e segurança encolhe para {fmt(100 * dif_lj, 1)} p.p. (IC 95% {fmt(100 * (dif_lj - 1.96 * ep_lj), 1)} a
  {fmt(100 * (dif_lj + 1.96 * ep_lj), 1)}); a privacidade continua claramente atrás.
* **H2 se mantém:** o escore médio ponderado cresce de {fmt(pop_h2['estimativa'][0])} (Analfabeto/Infantil) a
  {fmt(pop_h2['estimativa'][-1])} (Superior) ({fmt_p(wald_h2[2])}).
* **H3 se mantém, com efeito menor depois do ajuste:** a diferença urbano − rural ponderada é de
  {fmt(pop_h3_bruta['estimativa'][0])} habilidades na comparação bruta e de {fmt(pop_h3_padr['estimativa'][0])} depois de
  padronizar pela escolaridade (IC 95% {fmt(pop_h3_padr['ic_inf'][0])} a {fmt(pop_h3_padr['ic_sup'][0])}).
* **H4 se mantém:** o uso ponderado de IA vai de {fmt_pct(pop_h4['estimativa'][1])} (16–24) a {fmt_pct(pop_h4['estimativa'][-1])}
  (60+) ({fmt_p(wald_h4[2])}).
* **Limitação:** a camada populacional estima quantidades **análogas** (proporções e médias), não os mesmos testes de postos.
  Ela confirma a direção e a magnitude dos achados; não substitui os testes clássicos.
""")
''')

# =============================================================================
# 8. VISUALIZAÇÃO DOS RESULTADOS
# =============================================================================
md(r"""
## 8. Visualização e comunicação dos resultados

### 8.1. Uso de IA por faixa etária: amostra × população
**Objetivo:** mostrar, num gráfico pronto para apresentação, o gradiente etário do uso de IA com a incerteza **por desenho**.
""")
code(r'''
sem_peso_h4 = tab_h4["Usou IA"] / tab_h4.sum(axis=1)
fig = go.Figure()
fig.add_trace(go.Scatter(x=ORDEM_FAIXA, y=sem_peso_h4, mode="markers", name="Amostra (sem peso)",
                         marker=dict(size=11, color=COR["neutro"], symbol="diamond"),
                         hovertemplate="%{x}<br>sem peso: %{y:.1%}<extra></extra>"))
fig.add_trace(go.Scatter(x=ORDEM_FAIXA, y=pop_h4["estimativa"], mode="lines+markers", name="População (com peso, IC 95% por desenho)",
                         line=dict(color=COR["principal"], width=3), marker=dict(size=11),
                         error_y=dict(type="data", symmetric=False, array=pop_h4["ic_sup"] - pop_h4["estimativa"],
                                      arrayminus=pop_h4["estimativa"] - pop_h4["ic_inf"]),
                         customdata=np.c_[pop_h4["ic_inf"], pop_h4["ic_sup"]],
                         hovertemplate="%{x}<br>com peso: %{y:.1%}<br>IC 95%: %{customdata[0]:.1%} a %{customdata[1]:.1%}<extra></extra>"))
fig.add_annotation(xref="paper", yref="paper", x=0.98, y=0.98, xanchor="right", yanchor="top", showarrow=False, text=f"V de Cramér = {fmt(v_cramer, 2)}<br>Cochran-Armitage z = {fmt(ca['z'], 1)}",
                   bordercolor="#999", borderwidth=1, bgcolor="white")
fig.update_layout(title="Uso de ferramentas de IA nos últimos 3 meses, por faixa etária (usuários de Internet)", height=460,
                  yaxis=dict(tickformat=".0%", title="Proporção que usou IA", rangemode="tozero"), xaxis_title="Faixa etária",
                  legend=dict(orientation="h", yanchor="top", y=-0.18, x=0), margin=dict(b=120))
fig.show()
''')
md(r"""
**Interpretação.** As estimativas com e sem peso seguem o mesmo desenho: pico em 16–24 anos e queda contínua a partir daí. Os intervalos por desenho não se sobrepõem entre os extremos. A faixa de 10–15 anos fica abaixo da de 16–24 **no total**, mas a análise estratificada (7.6c) mostra que isso se deve à composição por escolaridade: comparando pessoas com a mesma escolaridade, os mais jovens usam mais.

### 8.2. Painel-síntese: efeitos com IC por desenho
**Objetivo:** reunir as quantidades populacionais de H1–H3 num único painel, com a linha de "ausência de efeito" quando faz sentido.
""")
code(r'''
fig = make_subplots(rows=1, cols=3, subplot_titles=("H1 · % que realizou", "H2 · escore por escolaridade",
                                                    "H3 · urbano − rural"), horizontal_spacing=0.1)
nomes_h1 = ["Segurança", "Privacidade", "Checagem"]
fig.add_trace(go.Scatter(x=nomes_h1, y=pop_h1["estimativa"], mode="markers", marker=dict(size=13, color=COR["secundaria"]),
                         error_y=dict(type="data", symmetric=False, array=pop_h1["ic_sup"] - pop_h1["estimativa"],
                                      arrayminus=pop_h1["estimativa"] - pop_h1["ic_inf"]), showlegend=False,
                         hovertemplate="%{x}: %{y:.1%}<extra></extra>"), row=1, col=1)
fig.add_trace(go.Scatter(x=ORDEM_GRAU, y=pop_h2["estimativa"], mode="lines+markers", marker=dict(size=12, color=COR["principal"]),
                         error_y=dict(type="data", symmetric=False, array=pop_h2["ic_sup"] - pop_h2["estimativa"],
                                      arrayminus=pop_h2["estimativa"] - pop_h2["ic_inf"]), showlegend=False,
                         hovertemplate="%{x}: %{y:.2f}<extra></extra>"), row=1, col=2)
for nome, r, cor in [("Bruta", pop_h3_bruta, COR["neutro"]), ("Padronizada p/ escolaridade", pop_h3_padr, COR["principal"])]:
    fig.add_trace(go.Scatter(x=[nome], y=r["estimativa"], mode="markers", marker=dict(size=13, color=cor), showlegend=False,
                             error_y=dict(type="data", symmetric=False, array=r["ic_sup"] - r["estimativa"],
                                          arrayminus=r["estimativa"] - r["ic_inf"]),
                             hovertemplate="%{x}: %{y:.2f} habilidades<extra></extra>"), row=1, col=3)
fig.add_hline(y=0, line_dash="dot", line_color="#999", row=1, col=3)
fig.update_yaxes(tickformat=".0%", rangemode="tozero", row=1, col=1)
fig.update_yaxes(title_text="habilidades (0–12)", rangemode="tozero", row=1, col=2)
fig.update_yaxes(rangemode="tozero", row=1, col=3)
fig.update_layout(title="Estimativas populacionais (com peso) e IC 95% pelos pesos replicados", height=500, margin=dict(t=120, b=110))
fig.show()
''')
md(r"""
**Interpretação.** O painel resume o estudo: (H1) a privacidade é a prática de cidadania digital menos difundida; (H2) o número de habilidades cresce com a escolaridade, com saltos grandes entre o Médio e o Superior; (H3) a vantagem urbana diminui quando a escolaridade é levada em conta, mas o intervalo continua acima de zero. O gráfico mostra o **tamanho** e a **incerteza** de cada efeito, e não apenas o p-valor.
""")

# =============================================================================
# 9. DISCUSSÃO
# =============================================================================
md(r"""
## 9. Discussão

### 9.1. Síntese dos achados
1. **Cidadania digital (H1):** entre os usuários, verificar informações é mais comum que adotar medidas de segurança, que por sua vez é mais comum que ajustar a privacidade. As diferenças são estatisticamente claras, mas **pequenas perto da variação entre pessoas**: quem pratica uma tende a praticar as outras.
2. **Escolaridade (H2):** é o fator com o efeito mais forte sobre as habilidades (ε² na faixa forte), com tendência crescente confirmada pelo Jonckheere-Terpstra e mantida dentro de todas as faixas etárias.
3. **Território (H3):** usuários urbanos têm mais habilidades. Parte da diferença é escolaridade, mas ela persiste entre pessoas com a mesma escolaridade (van Elteren e estimativa padronizada).
4. **IA (H4):** o uso de IA tem forte gradiente etário e está concentrado entre os mais escolarizados. No total, o pico está em 16–24 anos; dentro de cada nível de escolaridade, os mais jovens são os que mais usam. O "pico" marginal é um efeito de composição.

### 9.2. Significância estatística × relevância prática
Com cerca de 19 mil usuários, todos os p-valores são minúsculos, então a **relevância** precisa ser julgada pelos efeitos. Em H1, η²_Q é pequeno, mas uma diferença de cerca de 13 p.p. entre checagem e privacidade é relevante para políticas de educação digital. Em H2, o efeito é grande em qualquer métrica. Em H3, o efeito é modesto. Em H4, V moderado e diferenças de mais de 40 p.p. entre faixas indicam uma desigualdade de adoção expressiva.

### 9.3. Explicações possíveis (hipóteses, não conclusões)
* **Checagem > privacidade:** checar uma informação pode ser feito sem conhecimento técnico, por exemplo buscando outra fonte. Ajustar a privacidade exige navegar por menus de configuração, o que é uma barreira de habilidade.
* **Escolaridade:** a escola e o trabalho qualificado expõem as pessoas a tarefas como planilhas, apresentações e arquivos. Além disso, vários itens do escore são típicos de computador, que os mais escolarizados usam mais.
* **Urbano × rural:** qualidade da conexão, disponibilidade de computador e oportunidades de trabalho podem explicar a diferença residual.
* **IA e idade:** dentro de cada nível de escolaridade, quanto mais jovem, maior o uso. O menor uso dos 10–15 anos no total reflete a composição (a maioria ainda está no Fundamental). Entre os mais velhos, pode haver também menor reconhecimento de que se está usando "IA" (ver o "Não sabe" na Seção 3.4).

### 9.4. Limitações
* **Desenho observacional e transversal:** nenhuma das associações é causal. Escolaridade, idade, classe e território estão interligados; a estratificação controla um fator de cada vez.
* **Plano amostral:** os testes de postos clássicos ignoram estratos, conglomerados e pesos. A camada por desenho mitiga isso com quantidades análogas; não foram implementados testes de postos para amostras complexas (Lumley & Scott, 2013).
* **Bootstrap dos ICs de efeito:** é i.i.d. e não reproduz os conglomerados, então os ICs dos tamanhos de efeito tendem a ser estreitos demais.
* **Mensuração:** habilidades autodeclaradas e referentes aos últimos 3 meses medem **prática recente**, não competência; o escore pesa igualmente itens de dificuldade muito diferente. O uso de IA depende de a pessoa reconhecer a ferramenta.
* **Codificação:** duas inconsistências no dicionário (`F_C3_IDADE` e `I1A_NENHUM`) foram documentadas e contornadas.
* **Hipóteses definidas após exploração preliminar** da mesma base, sem pré-registro.

### 9.5. Generalização e pesquisas futuras
As conclusões valem para **usuários de Internet de 10 anos ou mais** no Brasil em 2025, apoiadas pela camada por desenho. Elas não se estendem aos não usuários, que são principalmente idosos e pessoas de baixa escolaridade. Sugestões:
* modelar as habilidades com regressão (por exemplo, binomial negativa inflada de zeros, ou modelos de itens) incluindo escolaridade, idade, classe e área ao mesmo tempo;
* acompanhar a adoção de IA nas próximas edições, já que 2025 marca um início de série;
* investigar a relação entre uso de IA e checagem de informações (os que usam IA checam mais ou menos?).
""")

# =============================================================================
# 10. CONCLUSÃO
# =============================================================================
md(r"""
## 10. Conclusão
""")
code(r'''
mostrar(f"""
**P1. As três práticas de cidadania digital são realizadas com a mesma frequência?** **Não.** Checagem de informações
({fmt_pct(p_chec)} na população de usuários) > medidas de segurança ({fmt_pct(p_seg)}) > ajuste de privacidade
({fmt_pct(p_priv)}) (Q de Cochran = {fmt(Q_manual, 1)}, {fmt_p(q_ref.pvalue)}; todas as comparações par a par significativas).
As práticas andam juntas na mesma pessoa, e a privacidade é a menos difundida.

**P2. O número de habilidades digitais difere entre os níveis de escolaridade?** **Sim, e cresce com ela**
(Kruskal-Wallis H = {fmt(kw.statistic, 1)}, ε² = {fmt(eps2, 3)}; Jonckheere-Terpstra z = {fmt(jt['z'], 1)}; {fmt_p(kw.pvalue)}).
O efeito se mantém dentro de cada faixa etária.

**P3. Usuários urbanos e rurais diferem, mesmo com a mesma escolaridade?** **Sim, mas a diferença é modesta.**
P(urbano > rural) = {fmt(ps, 2)} sem estratificar; o van Elteren, estratificado por escolaridade, também rejeita $H_0$
({fmt_p(ve['p'])}). Na população, a diferença cai de {fmt(pop_h3_bruta['estimativa'][0])} para {fmt(pop_h3_padr['estimativa'][0])}
habilidades depois da padronização pela escolaridade.

**P4. O uso de IA está associado à faixa etária?** **Sim** (χ² = {fmt(chi.statistic, 1)}, V de Cramér = {fmt(v_cramer, 2)},
{fmt_p(chi.pvalue)}). O uso cai com a idade (Cochran-Armitage z = {fmt(ca['z'], 1)}). No total, a faixa de 10–15 anos fica
abaixo da de 16–24, mas, dentro de cada nível de escolaridade, os mais jovens são os que mais usam IA.

**Contribuição.** Com dados oficiais de 2025 e testes não paramétricos adequados a itens binários pareados, escores com
muitos empates e grupos ordenados, o estudo mostra que as desigualdades digitais de **segundo nível** continuam fortemente
associadas à escolaridade e, em menor grau, ao território, e que a **adoção inicial de IA** segue linhas de idade e
escolaridade. Esses resultados sugerem um risco de a IA ampliar desigualdades já existentes.

**Ressalvas que devem acompanhar os resultados:** delineamento transversal (sem causalidade), uso de testes clássicos numa
amostra complexa (mitigado pela camada por desenho), habilidades autodeclaradas que medem prática recente e hipóteses
definidas após uma exploração preliminar da mesma base.
""")
''')

# =============================================================================
# 11. REFERÊNCIAS E REPRODUTIBILIDADE
# =============================================================================
md(r"""
## 11. Referências e reprodutibilidade

### 11.1. Dados e documentação
* Cetic.br/NIC.br. *Pesquisa sobre o uso das tecnologias de informação e comunicação nos domicílios brasileiros: TIC Domicílios 2025*. Base de microdados de indivíduos v1.0, dicionário de variáveis v1.0, relatório metodológico, relatório de coleta de dados e questionário. São Paulo: Comitê Gestor da Internet no Brasil. Disponível em: https://cetic.br/pt/arquivos/domicilios/2025/individuos/

### 11.2. Métodos estatísticos
* Cochran, W. G. (1950). The comparison of percentages in matched samples. *Biometrika*, 37(3/4), 256–266.
* McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika*, 12(2), 153–157.
* Kruskal, W. H., & Wallis, W. A. (1952). Use of ranks in one-criterion variance analysis. *Journal of the American Statistical Association*, 47(260), 583–621.
* Dunn, O. J. (1964). Multiple comparisons using rank sums. *Technometrics*, 6(3), 241–252.
* Jonckheere, A. R. (1954). A distribution-free k-sample test against ordered alternatives. *Biometrika*, 41(1/2), 133–145.
* Terpstra, T. J. (1952). The asymptotic normality and consistency of Kendall's test against trend, when ties are present in one ranking. *Indagationes Mathematicae*, 14, 327–333.
* Mann, H. B., & Whitney, D. R. (1947). On a test of whether one of two random variables is stochastically larger than the other. *Annals of Mathematical Statistics*, 18(1), 50–60.
* van Elteren, P. H. (1960). On the combination of independent two-sample tests of Wilcoxon. *Bulletin of the International Statistical Institute*, 37, 351–361.
* Armitage, P. (1955). Tests for linear trends in proportions and frequencies. *Biometrics*, 11(3), 375–386.
* Haberman, S. J. (1973). The analysis of residuals in cross-classified tables. *Biometrics*, 29(1), 205–220.
* Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70.
* Newcombe, R. G. (1998). Interval estimation for the difference between independent proportions. *Statistics in Medicine*, 17(8), 873–890.
* Tomczak, M., & Tomczak, E. (2014). The need to report effect size estimates revisited. *Trends in Sport Sciences*, 21(1), 19–25.
* Hollander, M., Wolfe, D. A., & Chicken, E. (2014). *Nonparametric Statistical Methods* (3ª ed.). Wiley.
* Conover, W. J. (1999). *Practical Nonparametric Statistics* (3ª ed.). Wiley.

### 11.3. Amostras complexas
* Rao, J. N. K., & Wu, C. F. J. (1988). Resampling inference with complex survey data. *Journal of the American Statistical Association*, 83(401), 231–241.
* Lumley, T. (2010). *Complex Surveys: A Guide to Analysis Using R*. Wiley.
* Lumley, T., & Scott, A. J. (2013). Two-sample rank tests under complex sampling. *Biometrika*, 100(4), 831–842.
* Opsomer, J. D., & Erciulescu, A. L. (2021). Replication variance estimation after sample-based calibration. *Survey Methodology*, 47(2), 265–277.
* Kish, L. (1965). *Survey Sampling*. Wiley.

### 11.4. Tema
* van Deursen, A. J. A. M., & van Dijk, J. A. G. M. (2014). The digital divide shifts to differences in usage. *New Media & Society*, 16(3), 507–526.
* Helsper, E. J. (2021). *The Digital Disconnect: The Social Causes and Consequences of Digital Inequalities*. SAGE.

### 11.5. Ambiente de execução
""")
code(r'''
versoes = pd.Series({"Python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                     "scipy": scipy.__version__, "statsmodels": statsmodels.__version__, "plotly": plotly.__version__,
                     "Semente (bootstrap)": SEED, "Reamostragens bootstrap": B_BOOT,
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
   jupyter nbconvert --to notebook --execute --inplace notebooks/03_tic_domicilios_2025_habilidades_ia.ipynb
   ```
3. Se `data/tic_domicilios_2025/` não tiver a base, a Seção 0 baixa os microdados (cerca de 76 MB) e o dicionário oficiais do site do Cetic.br.
4. Todos os procedimentos aleatórios usam a semente `SEED = 2025`, então os ICs bootstrap são reproduzíveis.
5. Este notebook é gerado por `scripts/build_notebook_domicilios.py`. Edite o script e regenere o notebook para manter os dois sincronizados.
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"Notebook escrito em {OUT} ({len(cells)} células)")
