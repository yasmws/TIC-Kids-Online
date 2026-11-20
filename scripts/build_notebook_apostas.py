"""Gera o notebook do estudo de apostas digitais com a TIC Domicílios 2025 (indivíduos).

Reorganiza, na metodologia dos notebooks 02 e 03, a análise original da branch `lucas-dataset-bets`
(`projeto_apostas_digitais.ipynb`). Os testes originais são mantidos com a mesma lógica (camada clássica);
as melhorias entram como camadas adicionais: universo de internautas e estimação pelo plano amostral.

Uso (a partir da pasta PROJ 2):
    python scripts/build_notebook_apostas.py notebooks/04_tic_domicilios_2025_apostas.ipynb
    jupyter nbconvert --to notebook --execute --inplace notebooks/04_tic_domicilios_2025_apostas.ipynb
"""
import sys
from pathlib import Path

import nbformat as nbf

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "notebooks/04_tic_domicilios_2025_apostas.ipynb")
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip()))


# =============================================================================
# CAPA
# =============================================================================
md(r"""
# O perfil do apostador digital brasileiro
### Testes de associação, testes não paramétricos e regressão logística com os microdados da TIC Domicílios 2025 (indivíduos)

**Base de dados:** TIC Domicílios 2025, módulo de indivíduos, microdados v1.0, Cetic.br/NIC.br (bloco C14, apostas na Internet).
**Origem da análise:** este notebook reorganiza o estudo `projeto_apostas_digitais.ipynb` da branch `lucas-dataset-bets`. **Todos os testes e modelos originais foram mantidos com a mesma lógica de cálculo**: intervalos de confiança, qui-quadrado com V de Cramér, razão de chances, resíduos de Haberman, Mann-Whitney, Kruskal-Wallis com pós-teste de Bonferroni e regressão logística por IRLS.
**O que foi acrescentado:** cada teste original ganha duas camadas de verificação:
1. **Universo de internautas:** a mesma análise restrita a quem podia responder ao bloco de apostas.
2. **Camada por desenho:** estimativas com `PESO` e erros-padrão pelos 200 pesos replicados.

| Seção | Conteúdo |
|---|---|
| 0 | Configuração do ambiente e funções auxiliares |
| 1 | Introdução e contextualização |
| 2 | Compreensão e documentação do dataset |
| 3 | Preparação e qualidade dos dados |
| 4 | Análise exploratória de dados (EDA) |
| 5 | Formulação das hipóteses estatísticas |
| 6 | Seleção e justificativa dos testes |
| 7 | Execução dos testes de hipóteses e do modelo |
| 8 | Visualização e comunicação dos resultados |
| 9 | Discussão |
| 10 | Conclusão |
| 11 | Referências e reprodutibilidade |

> **Como ler este notebook.** Cada etapa segue a sequência *O que vamos fazer? → Por que isso é necessário? → Como vamos fazer? → O que encontramos? → Como interpretar?*. Os números citados nos textos são gerados pelas próprias células. Nas seções de execução, a célula marcada **"análise original"** traz o código do estudo de origem. A célula seguinte, **"verificação"**, traz as camadas acrescentadas.
""")

# =============================================================================
# 0. CONFIGURAÇÃO
# =============================================================================
md(r"""
## 0. Configuração do ambiente

**O que vamos fazer?** Importar as bibliotecas, localizar os arquivos e definir as funções auxiliares usadas nas camadas de verificação.

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
import scipy.stats as stats
from IPython.display import Markdown, display
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_colwidth", 110)
pd.set_option("display.width", 160)
pd.set_option("display.float_format", lambda x: f"{x:.3f}")

ALPHA = 0.05          # nível de significância adotado em todos os testes (o mesmo do estudo original)
EXPORTAR_BASE = False # True grava a base tratada em CSV, como no estudo original (Seção 3.7)
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
print("Pasta de dados:", PASTA_DADOS.relative_to(RAIZ))
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
       "terciaria": "#4daf4a", "quaternaria": "#984ea3",
       "seq": ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c", "#08306b"]}
COR_MOD = {"Cassino": COR["destaque"], "Rifas": COR["secundaria"], "Bets": COR["principal"], "Loteria": COR["terciaria"]}
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


def rotulos_pct(valores, casas=1):
    """Rótulos de barra em porcentagem, já multiplicados por 100 (como no estudo original)."""
    return [fmt(v, casas) + "%" for v in valores]
''')
code(r'''
# 0.4 Funções das camadas de verificação
def prop_ponderada(y, pesos, mascara=None):
    """Proporção (ou média) ponderada de y, ignorando ausentes. `mascara` restringe o universo."""
    y, pesos = np.asarray(y, float), np.asarray(pesos, float)
    ok = ~np.isnan(y) if mascara is None else (~np.isnan(y) & np.asarray(mascara, bool))
    return np.sum(y[ok] * pesos[ok]) / np.sum(pesos[ok])


def estimativa_com_replicas(estimador, nivel=0.95):
    """Aplica `estimador(pesos)` ao PESO e às 200 réplicas bootstrap (Rao-Wu) da TIC Domicílios.

    Variância: V = 1/(R-1) * soma_r (theta_r - média(theta_r))², como em `as.svrepdesign(type="bootstrap")`.
    Aceita estimativas escalares ou vetoriais.
    """
    theta = np.atleast_1d(estimador(W_PESO))
    reps = np.array([np.atleast_1d(estimador(W_REP[:, r])) for r in range(W_REP.shape[1])])
    ep = reps.std(axis=0, ddof=1)
    zc = stats.norm.isf((1 - nivel) / 2)
    return dict(estimativa=theta, ep=ep, ic_inf=theta - zc * ep, ic_sup=theta + zc * ep,
                p=2 * stats.norm.sf(np.abs(theta / ep)), replicas=reps)


def wald_igualdade(theta, replicas):
    """Wald por desenho para H0: todos os componentes de theta iguais (contrastes sucessivos)."""
    k = len(theta)
    C = np.eye(k)[:-1] - np.eye(k)[1:]
    cov = np.atleast_2d(np.cov((replicas @ C.T).T, ddof=1))
    X2 = float((C @ theta) @ np.linalg.solve(cov, C @ theta))
    return X2, k - 1, stats.chi2.sf(X2, k - 1)


def wald_zero(theta, replicas):
    """Wald por desenho para H0: theta = 0 (vetor)."""
    cov = np.atleast_2d(np.cov(replicas.T, ddof=1))
    X2 = float(theta @ np.linalg.solve(cov, theta))
    return X2, len(theta), stats.chi2.sf(X2, len(theta))


def montar_matriz(dados):
    """Matriz de desenho do modelo logístico, com as mesmas definições do estudo original (Seção 7.7)."""
    f = pd.DataFrame(index=dados.index)
    f["intercepto"] = 1.0
    f["idade_padronizada"] = (dados["IDADE"] - dados["IDADE"].mean()) / dados["IDADE"].std()
    f["sexo_masculino"] = (dados["sexo_desc"] == "Masculino").astype(float)
    f["ensino_medio"] = (dados["escolaridade_desc"] == "Médio").astype(float)
    f["ensino_superior"] = (dados["escolaridade_desc"] == "Superior").astype(float)
    f["classe_C"] = (dados["classe_desc"] == "Classe C").astype(float)
    f["classe_B"] = (dados["classe_desc"] == "Classe B").astype(float)
    f["classe_A"] = (dados["classe_desc"] == "Classe A").astype(float)
    f["trabalho_informal"] = (dados["tipo_ocupacao_desc"].fillna("Outro") == "Informal").astype(float)
    f["usa_pix"] = dados["C8_I"].fillna(0).astype(float)
    f["assiste_influenciadores"] = dados["TC4B_I"].fillna(0).astype(float)
    f["assiste_videos_esporte"] = dados["TC4B_B"].fillna(0).astype(float)
    f["joga_games_online"] = dados["C9_A"].fillna(0).astype(float)
    f["mobile_only"] = (dados["dispositivos_desc"] == "Apenas Celular").astype(float)
    return f


def irls(X, y, pesos=None, max_iter=50, tol=1e-8):
    """Newton-Raphson/IRLS do estudo original, com pesos opcionais (normalizados para média 1)."""
    w = np.ones(len(y)) if pesos is None else np.asarray(pesos, float) / np.mean(pesos)
    beta = np.zeros(X.shape[1])
    beta[0] = np.log(y.mean() / (1.0 - y.mean()))
    for _ in range(max_iter):
        mu = 1.0 / (1.0 + np.exp(-np.clip(X @ beta, -20, 20)))
        H = (X.T * (w * mu * (1.0 - mu))) @ X
        passo = np.linalg.solve(H, X.T @ (w * (y - mu)))
        beta += passo
        if np.linalg.norm(passo) < tol:
            break
    mu = 1.0 / (1.0 + np.exp(-np.clip(X @ beta, -20, 20)))
    return beta, np.linalg.inv((X.T * (w * mu * (1.0 - mu))) @ X), mu


def tabela_logit(beta, ep, nomes):
    """Tabela no mesmo formato da tabela original (coeficiente, EP, z de Wald, p, aOR e IC)."""
    z = beta / ep
    return pd.DataFrame({"Coeficiente (β)": beta, "Erro Padrão": ep, "z de Wald": z,
                         "p-valor": 2 * stats.norm.sf(np.abs(z)), "aOR (Odds Ratio)": np.exp(beta),
                         "IC 95% Inf": np.exp(beta - 1.96 * ep), "IC 95% Sup": np.exp(beta + 1.96 * ep)}, index=nomes)


def qui2_e_or(tabela):
    """Qui-quadrado com Yates e OR com IC de Woolf, exatamente como na Seção 7.3 original."""
    chi2, p, _, _ = stats.chi2_contingency(tabela, correction=True)
    a, b, c, d = np.asarray(tabela, float).ravel()
    odds = (a * d) / (b * c)
    ep = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return dict(chi2=chi2, p=p, OR=odds, ic=(np.exp(np.log(odds) - 1.96 * ep), np.exp(np.log(odds) + 1.96 * ep)))
''')

# =============================================================================
# 1. INTRODUÇÃO
# =============================================================================
md(r"""
## 1. Introdução e contextualização do estudo

### 1.1. Contexto
Nos últimos anos, o Brasil viu uma expansão rápida do mercado de apostas e jogos de azar online: *bets* esportivas, cassinos virtuais como o *jogo do tigrinho*, rifas digitais em redes sociais e loterias pela Internet. O tema passou a ocupar os debates sobre finanças das famílias, economia comportamental e saúde pública.

A **TIC Domicílios 2025**, do Cetic.br/NIC.br, incluiu pela primeira vez um bloco sobre apostas feitas na Internet nos últimos 3 meses (questão **C14**), com quatro modalidades: loteria federal, cassino online, aposta esportiva e rifa ou sorteio digital.

### 1.2. Motivação e relevância
A pesquisa é probabilística e de abrangência nacional. Por isso, permite descrever **quem aposta** com representatividade para a população de 10 anos ou mais, algo que dados de plataformas ou pesquisas de opinião não oferecem.

### 1.3. Objetivos
1. Estimar a prevalência de apostas digitais, no total e por modalidade.
2. Descrever o perfil demográfico, socioeconômico, geográfico e digital de quem aposta.
3. Testar associações entre esse perfil e a prática de apostas.
4. Estimar, num modelo multivariado, quais características seguem associadas à aposta quando as demais são mantidas constantes.

### 1.4. Perguntas de pesquisa
* **P1.** A modalidade preferida (só bets, só cassino ou ambas) depende do gênero?
* **P2.** Assistir a vídeos de influenciadores digitais está associado a apostar em cassino online?
* **P3.** A proporção de apostadores é a mesma em todas as classes econômicas?
* **P4.** Apostadores são mais jovens que não apostadores?
* **P5.** A idade difere entre as modalidades de aposta?
* **P6.** Quais fatores seguem associados à aposta no modelo multivariado?

### 1.5. Metodologia em resumo
O estudo original usa testes clássicos de tabelas de contingência (qui-quadrado, razão de chances e resíduos ajustados), testes não paramétricos de posição (Mann-Whitney e Kruskal-Wallis) e regressão logística ajustada por máxima verossimilhança. Este notebook mantém esses procedimentos e acrescenta, para cada um:
* a análise no **universo de internautas**, porque o bloco C14 só foi perguntado a quem usou a Internet;
* a **camada por desenho**, com pesos amostrais e erros-padrão pelos pesos replicados.

### 1.6. Contribuição e limitações antecipadas
O estudo é **transversal**: mede associações, não causas. As apostas são **autodeclaradas** e podem ser subnotificadas. O questionário não mede **valores apostados** nem perdas.
""")

# =============================================================================
# 2. DATASET
# =============================================================================
md(r"""
## 2. Compreensão e documentação do dataset

**O que vamos fazer?** Descrever a origem dos dados, carregar as variáveis do estudo e documentá-las a partir do dicionário oficial.

**Por que isso é necessário?** Os códigos especiais e os filtros do questionário definem quem pode responder a cada pergunta. Sem entendê-los, a análise pode comparar grupos que não são comparáveis.

### 2.1. Ficha técnica

| Item | Descrição |
|---|---|
| Pesquisa | TIC Domicílios 2025, módulo de indivíduos (Cetic.br/NIC.br) |
| População | Residentes em domicílios particulares permanentes do Brasil, com 10 anos ou mais |
| Unidade de análise | Indivíduo (um morador sorteado por domicílio) |
| Bloco analisado | C14: apostas feitas na Internet nos últimos 3 meses (loteria, cassino, esportes, rifa) |
| Quem responde ao C14 | Usuários de Internet. Os demais recebem o código 99 (*Não se aplica*) |
| Plano amostral | Amostragem estratificada de conglomerados em múltiplos estágios |
| Pesos | `PESO` (calibrado) e 200 pesos replicados `rep1`–`rep200`, gerados por bootstrap com o mesmo desenho |
""")
md(r"""
### 2.2. Carga dos dados (análise original)
O estudo original lê apenas as 32 variáveis de interesse com `usecols`. A única mudança é o caminho do arquivo, que agora vem da Seção 0. Os pesos replicados são lidos à parte, em `replicas`, para não alterar a base `df` do estudo original.
""")
code(r'''
csv_path = ARQ_DADOS

# Mapeamento e seleção das variáveis analíticas
colunas_selecionadas = [
    # Identificadores e Amostragem Probabilística
    'QUEST', 'ID_DOMICILIO', 'ID_MORADOR', 'PESO',

    # Módulo C14: Apostas e Jogos Digitais (Últimos 3 meses)
    'C14_A',       # Loteria federal online (Mega-Sena, etc.)
    'C14_B',       # Cassino online (Jogo do Tigrinho, etc.)
    'C14_C',       # Apostas esportivas (Bet365, Betano, Superbet, etc.)
    'C14_D',       # Rifa digital / sorteio em rede social ou app
    'C14_AGREG',   # Indicador agregado: Fez algum tipo de aposta

    # Perfil Sociodemográfico
    'SEXO', 'IDADE', 'FAIXA_ETARIA', 'GRAU_INST_1', 'RACA', 'RELIGIAO',

    # Condição Econômica e Inserção no Mercado de Trabalho
    'CLASSE_2015', 'RENDA_FAMILIAR_2', 'RENDA_PESSOAL', 'PEA_2', 'TIPO_OCUP',

    # Recorte Geográfico
    'COD_REGIAO_2', 'AREA', 'COD_UF',

    # Comportamento Digital e Meios de Acesso
    'C1',                # Já usou a internet
    'C3',                # Última utilização da internet
    'C4',                # Frequência de acesso à internet
    'C5_DISPOSITIVOS',   # Exclusividade móvel (apenas celular vs PC)
    'C8_H',              # Realização de transações financeiras online
    'C8_I',              # Uso de pagamentos via Pix
    'C9_A',              # Prática de jogos eletrônicos online
    'TC4B_B',            # Consumo de vídeos esportivos online
    'TC4B_I'             # Consumo de vídeos de influenciadores digitais
]

print(f"Carregando {len(colunas_selecionadas)} variáveis selecionadas...")
df_raw = pd.read_csv(csv_path, sep=';', usecols=colunas_selecionadas, decimal=',')

print("Microdados carregados.")
print(f"Dimensões: {df_raw.shape[0]:,} respondentes (indivíduos) e {df_raw.shape[1]} atributos.")
''')
code(r'''
# Pesos replicados (camada por desenho) e dicionário oficial
COLS_REPLICAS = [f"rep{i}" for i in range(1, 201)]
replicas = pd.read_csv(ARQ_DADOS, sep=";", decimal=",", usecols=COLS_REPLICAS)[COLS_REPLICAS]
W_PESO = df_raw["PESO"].to_numpy(float)
W_REP = replicas.to_numpy(float)

dicionario = pd.read_excel(ARQ_DICIO, sheet_name="Indivíduos", header=1)
dicionario.columns = ["tamanho", "variavel", "descricao", "tipo_arquivo", "codigos"]
dicionario = dicionario.dropna(subset=["variavel"]).set_index("variavel")

assert len(replicas) == len(df_raw), "réplicas e base com tamanhos diferentes"
print(f"Pesos replicados: {W_REP.shape[1]} | Soma do PESO (população de 10 anos ou mais): {fmt(W_PESO.sum(), 0)}")
''')
md(r"""
### 2.3. Variáveis utilizadas
A descrição e os códigos vêm do dicionário oficial. O papel de cada variável no estudo é a classificação adotada aqui.
""")
code(r'''
PAPEL = {
    "C14_A": "Desfecho (loteria online)", "C14_B": "Desfecho (cassino online)", "C14_C": "Desfecho (aposta esportiva)",
    "C14_D": "Desfecho (rifa/sorteio digital)", "C14_AGREG": "Desfecho principal (fez alguma aposta)",
    "SEXO": "H1, modelo", "IDADE": "H4, H5, modelo", "FAIXA_ETARIA": "EDA", "GRAU_INST_1": "EDA, modelo",
    "CLASSE_2015": "H3, modelo", "TIPO_OCUP": "EDA, modelo", "COD_REGIAO_2": "EDA", "AREA": "EDA",
    "C5_DISPOSITIVOS": "EDA, modelo", "C8_I": "EDA, modelo", "C9_A": "Modelo", "TC4B_B": "Modelo",
    "TC4B_I": "H2, modelo", "C3": "Filtro de uso da Internet", "PESO": "Camada por desenho",
}
pd.DataFrame([{"Variável": v, "Descrição (dicionário)": dicionario.loc[v, "descricao"],
               "Códigos": str(dicionario.loc[v, "codigos"]).replace("\n", "; ") if pd.notna(dicionario.loc[v, "codigos"]) else "contínua",
               "Papel no estudo": papel} for v, papel in PAPEL.items()])
''')
md(r"""
As demais variáveis lidas pelo estudo original (raça, religião, renda, condição de atividade, UF, `C1`, `C4`, `C8_H`) servem para a decodificação e a auditoria. Elas não entram nos testes.

### 2.4. Critérios de inclusão: dois universos

| Universo | Definição | Uso |
|---|---|---|
| **População de 10 anos ou mais** | Todos os 24.535 respondentes. Quem não usou a Internet recebe 99 no C14 e conta como "não apostou" | Análise original (prevalência na população) |
| **Internautas** | `C14_A ≠ 99`, ou seja, quem respondeu ao bloco de apostas | Verificação: comparações de perfil e modelo |

**Por que dois universos?** Para estimar quantos brasileiros apostam, faz sentido contar os não internautas como "não apostou", porque eles não apostam *online*. Para comparar perfis, isso distorce os resultados: quem não usa a Internet também não usa Pix *pela Internet*, não assiste a vídeos e é mais velho. Incluí-los cria uma associação mecânica entre essas variáveis e a aposta.

### 2.5. Limitações conhecidas da base
* A amostra **não é autoponderada**: sem pesos, as proporções descrevem a amostra, não o Brasil.
* O microdado **não traz estrato nem UPA**. O plano amostral só pode ser incorporado pelos pesos replicados.
* O C14 mede a **ocorrência** de aposta nos últimos 3 meses, não a frequência nem o valor apostado.
""")

# =============================================================================
# 3. PREPARAÇÃO
# =============================================================================
md(r"""
## 3. Preparação e qualidade dos dados

**O que vamos fazer?** Verificar a integridade da base, quantificar os códigos especiais e aplicar o tratamento, a decodificação e a engenharia de atributos do estudo original.

**Por que isso é necessário?** O tratamento dos códigos 97, 98 e 99 muda o denominador de cada proporção. É a decisão de preparação com mais efeito nos resultados.

### 3.1. Integridade
""")
code(r'''
duplicados = df_raw.duplicated(["ID_DOMICILIO", "ID_MORADOR"]).sum()
nulos = int(df_raw.isna().sum().sum())
print(f"Registros duplicados (domicílio + morador): {duplicados}")
print(f"Células vazias em todo o recorte: {nulos}")
print(f"Um morador por domicílio: {df_raw.ID_DOMICILIO.is_unique}")
''')
md(r"""
### 3.2. Frequência dos códigos especiais
""")
code(r'''
ESPECIAIS = {97: "Não sabe", 98: "Não respondeu", 99: "Não se aplica"}
VARS_CODIGOS = ["C14_A", "C14_B", "C14_C", "C14_D", "C8_I", "C9_A", "TC4B_B", "TC4B_I", "C5_DISPOSITIVOS"]
tabela_especiais = pd.DataFrame({v: {rot: int((df_raw[v] == cod).sum()) for cod, rot in ESPECIAIS.items()}
                                 for v in VARS_CODIGOS}).T
tabela_especiais["Respostas 0/1 (ou categoria)"] = len(df_raw) - tabela_especiais.sum(axis=1)
tabela_especiais
''')
code(r'''
n_nao_internautas = int((df_raw["C14_A"] == 99).sum())
mesmo_filtro = all((df_raw[v] == 99).equals(df_raw["C14_A"] == 99) for v in ["C14_B", "C14_C", "C14_D", "C8_I", "C9_A"])
mostrar(f"""
* **Resultado:** {fmt(n_nao_internautas, 0)} respondentes ({fmt_pct(n_nao_internautas / len(df_raw))}) têm código 99 no C14,
  ou seja, não usaram a Internet nos últimos 3 meses. O filtro é o mesmo em `C14_*`, `C8_I` e `C9_A`: {'sim' if mesmo_filtro else 'não'}.
  `TC4B_B` e `TC4B_I` têm mais códigos 99 ({fmt(int((df_raw['TC4B_I'] == 99).sum()), 0)}), porque também excluem quem não assistiu
  a nenhum vídeo pela Internet.
* **Interpretação:** 97 e 98 são raros (menos de 0,1%) e não afetam os resultados. O código 99 é o ponto decisivo, porque
  marca **quem não podia responder**. É por isso que este notebook analisa também o universo de internautas.
""")
''')
md(r"""
### 3.3. Tratamento dos códigos especiais (análise original)
Os critérios do estudo de origem são:
1. **Modalidades de aposta (`C14_A` a `C14_D`):** 1 = sim, 0 = não, 97/98 = ausente e 99 = 0, porque quem não usou a Internet não apostou online.
2. **Hábitos digitais (`C8_H`, `C8_I`, `C9_A`, `TC4B_B`, `TC4B_I`):** o mesmo princípio, com 99 = 0.
""")
code(r'''
df = df_raw.copy()

modalidades_aposta = ['C14_A', 'C14_B', 'C14_C', 'C14_D']

for col in modalidades_aposta:
    # 97 e 98 -> NaN (respostas não válidas)
    # 99 -> 0 (não usou internet nos últimos 3 meses, logo não apostou online)
    df[col] = df[col].replace({97: np.nan, 98: np.nan, 99: 0})

# O indicador C14_AGREG já é fornecido pelo Cetic.br com 1=Apostou e 0=Não apostou
df['C14_AGREG'] = df['C14_AGREG'].astype(int)

# Tratamento para hábitos digitais auxiliares
habitos_digitais = ['C8_H', 'C8_I', 'C9_A', 'TC4B_B', 'TC4B_I']
for col in habitos_digitais:
    df[col] = df[col].replace({97: np.nan, 98: np.nan, 99: 0})

print("Códigos especiais tratados.")
''')
code(r'''
# Verificação: marcador do universo de internautas (quem respondeu ao bloco C14)
df["internauta"] = (df_raw["C14_A"] != 99).astype(int)
INTERNAUTA = df["internauta"].to_numpy(bool)
print(f"Internautas: {fmt(INTERNAUTA.sum(), 0)} de {fmt(len(df), 0)} respondentes")
print(f"Apostadores fora do universo de internautas: {int(df.loc[~INTERNAUTA, 'C14_AGREG'].sum())} (esperado: 0)")
''')
md(r"""
### 3.4. Decodificação das categorias (análise original)
Os rótulos seguem o dicionário oficial da TIC Domicílios 2025. As variáveis ordinais (faixa etária, escolaridade e classe) ficam ordenadas.
""")
code(r'''
# Dicionários de decodificação
map_sexo = {1: 'Masculino', 2: 'Feminino'}

map_faixa_etaria = {
    1: '10 a 15 anos',
    2: '16 a 24 anos',
    3: '25 a 34 anos',
    4: '35 a 44 anos',
    5: '45 a 59 anos',
    6: '60 anos ou mais'
}

map_instrucao = {
    1: 'Analfabeto/Ed. Infantil',
    2: 'Fundamental',
    3: 'Médio',
    4: 'Superior'
}

map_classe = {
    1: 'Classe A',
    2: 'Classe B',
    3: 'Classe C',
    4: 'Classe DE'
}

map_renda_familiar = {
    1: 'Até 1 SM',
    2: 'Mais de 1 a 2 SM',
    3: 'Mais de 2 a 3 SM',
    4: 'Mais de 3 a 5 SM',
    5: 'Mais de 5 a 10 SM',
    6: 'Mais de 10 SM',
    7: 'Sem renda',
    97: np.nan,
    98: np.nan
}

map_regiao = {
    1: 'Sudeste',
    2: 'Nordeste',
    3: 'Sul',
    4: 'Norte',
    5: 'Centro-Oeste'
}

map_area = {
    1: 'Urbana',
    2: 'Rural'
}

map_raca = {
    1: 'Branca',
    2: 'Preta',
    3: 'Parda',
    4: 'Amarela',
    5: 'Indígena',
    98: np.nan
}

map_pea = {
    1: 'Na força de trabalho',
    2: 'Fora da força de trabalho'
}

map_ocup = {
    1: 'Formal',
    2: 'Informal',
    4: 'Não se aplica'
}

map_dispositivos = {
    1: 'Apenas Computador',
    2: 'Apenas Celular',
    3: 'Ambos (PC e Celular)',
    4: 'Nenhum desses',
    99: 'Não se aplica'
}

# Criação das colunas descritivas
df['sexo_desc'] = df['SEXO'].map(map_sexo)
df['faixa_etaria_desc'] = pd.Categorical(df['FAIXA_ETARIA'].map(map_faixa_etaria),
                                        categories=list(map_faixa_etaria.values()), ordered=True)
df['escolaridade_desc'] = pd.Categorical(df['GRAU_INST_1'].map(map_instrucao),
                                        categories=list(map_instrucao.values()), ordered=True)
df['classe_desc'] = pd.Categorical(df['CLASSE_2015'].map(map_classe),
                                   categories=['Classe DE', 'Classe C', 'Classe B', 'Classe A'], ordered=True)
df['renda_familiar_desc'] = df['RENDA_FAMILIAR_2'].map(map_renda_familiar)
df['regiao_desc'] = df['COD_REGIAO_2'].map(map_regiao)
df['area_desc'] = df['AREA'].map(map_area)
df['raca_desc'] = df['RACA'].map(map_raca)
df['situacao_trabalho_desc'] = df['PEA_2'].map(map_pea)
df['tipo_ocupacao_desc'] = df['TIPO_OCUP'].map(map_ocup)
df['dispositivos_desc'] = df['C5_DISPOSITIVOS'].map(map_dispositivos)

print("Variáveis categóricas e ordinais decodificadas.")
''')
md(r"""
### 3.5. Engenharia de atributos (análise original)
1. **`qtd_modalidades`:** número de modalidades em que a pessoa apostou (0 a 4).
2. **`poliapostador`:** indicador de quem aposta em 2 ou mais modalidades.
3. **`status_aposta`:** rótulo *Apostador* / *Não Apostador*.
4. **`aposta_loteria`, `aposta_cassino`, `aposta_esportiva`, `aposta_rifa`:** nomes explícitos para as modalidades.
""")
code(r'''
# Quantidade de modalidades praticadas pelo indivíduo
df['qtd_modalidades'] = df[modalidades_aposta].sum(axis=1)

# Poliapostador (engajamento em múltiplas modalidades)
df['poliapostador'] = (df['qtd_modalidades'] >= 2).astype(int)

# Rótulo de status de aposta
df['status_aposta'] = df['C14_AGREG'].map({1: 'Apostador', 0: 'Não Apostador'})

# Renomeação semântica das modalidades
df['aposta_loteria'] = df['C14_A']
df['aposta_cassino'] = df['C14_B']
df['aposta_esportiva'] = df['C14_C']
df['aposta_rifa'] = df['C14_D']

print("--- Distribuição da Quantidade de Modalidades de Aposta ---")
print(df['qtd_modalidades'].value_counts().sort_index())

print("\n--- Proporção de Poliapostadores na Amostra Geral ---")
pct_poli = df['poliapostador'].value_counts(normalize=True) * 100
for val, p in pct_poli.items():
    lbl = "Poliapostador (2+ tipos)" if val == 1 else "Aposta em 1 ou nenhuma modalidade"
    print(f"{lbl}: {p:.2f}%")
''')
md(r"""
### 3.6. Auditoria da base tratada (análise original)
""")
code(r'''
print("==================================================")
print("             AUDITORIA DO DATASET TRATADO         ")
print("==================================================")
print(f"Total de Respondentes: {len(df):,}")
print(f"Total de Variáveis Finais: {df.shape[1]}")
print(f"Apostadores Identificados na Amostra: {df['C14_AGREG'].sum():,} ({df['C14_AGREG'].mean():.2%})")

print("\n--- Resumo Estatístico das Variáveis Contínuas ---")
print(df[['IDADE', 'PESO', 'qtd_modalidades']].describe())

print("\n--- Prevalência por Modalidade de Aposta (Amostra Bruta) ---")
mod_stats = pd.DataFrame({
    'Modalidade': ['Cassino Online (Tigrinho)', 'Rifa Digital / Sorteio', 'Apostas Esportivas (Bets)', 'Loteria Federal Online'],
    'Respondentes (Sim)': [df['aposta_cassino'].sum(), df['aposta_rifa'].sum(), df['aposta_esportiva'].sum(), df['aposta_loteria'].sum()]
})
mod_stats['% Amostra Total'] = (mod_stats['Respondentes (Sim)'] / len(df) * 100).round(2).astype(str) + '%'
print(mod_stats.to_string(index=False))

print("\n--- Tabela de Frequência Inicial: Gênero vs Status de Aposta ---")
tabela_sexo = pd.crosstab(df['sexo_desc'], df['status_aposta'], margins=True, margins_name="Total")
tabela_sexo_pct = pd.crosstab(df['sexo_desc'], df['status_aposta'], normalize='index').mul(100).round(2)
print(tabela_sexo)
print("\nPercentual dentro de cada gênero:")
print(tabela_sexo_pct.astype(str) + '%')
''')
md(r"""
### 3.7. Exportação da base tratada (análise original, opcional)
O estudo original grava a base tratada em `tic_domicilios_processada_apostas.csv`. Aqui a gravação fica desligada por padrão (`EXPORTAR_BASE = False`), para que executar o notebook não crie arquivos.
""")
code(r'''
if EXPORTAR_BASE:
    output_csv = PASTA_DADOS / 'tic_domicilios_processada_apostas.csv'
    df.to_csv(output_csv, index=False, sep=';', decimal='.')
    print(f"Base processada gravada em '{output_csv}' ({output_csv.stat().st_size / 1024**2:.2f} MB).")
else:
    print("Exportação desligada (EXPORTAR_BASE = False).")
''')

# =============================================================================
# 4. EDA
# =============================================================================
md(r"""
## 4. Análise exploratória de dados (EDA)

**O que vamos fazer?** Descrever a prevalência de apostas e o perfil dos apostadores: demográfico, socioeconômico, geográfico e digital.

**Por que isso é necessário?** A EDA mostra os padrões que as hipóteses da Seção 5 vão testar e revela problemas de universo antes da inferência.

**Como vamos fazer?** Com os cálculos do estudo original (taxas na amostra, sem peso, salvo indicação). Os gráficos foram refeitos em Plotly.

### 4.1. Prevalência: amostra × estimativa ponderada
A estimativa populacional usa o peso amostral $w_i$ (`PESO`):
$$\hat{P} = \frac{\sum_{i=1}^n w_i \cdot Y_i}{\sum_{i=1}^n w_i}$$
""")
code(r'''
# Cálculo da prevalência amostral e ponderada (análise original)
mods = {
    'Cassino Online (Tigrinho)': 'aposta_cassino',
    'Rifa Digital / Sorteio': 'aposta_rifa',
    'Apostas Esportivas (Bets)': 'aposta_esportiva',
    'Loteria Federal Online': 'aposta_loteria',
    'Qualquer Tipo de Aposta (Total)': 'C14_AGREG'
}

peso_total = df['PESO'].sum()
pop_brasil_proj = peso_total

resumo_prev = []
for label, col in mods.items():
    taxa_amostral = df[col].mean() * 100
    taxa_ponderada = (df[col] * df['PESO']).sum() / peso_total * 100
    total_proj = (df[col] * df['PESO']).sum()

    resumo_prev.append({
        'Modalidade': label,
        'Taxa Amostral (%)': taxa_amostral,
        'Taxa Ponderada (%)': taxa_ponderada,
        'População Estimada (milhões)': total_proj / 1e6
    })

df_prev = pd.DataFrame(resumo_prev).sort_values(by='Taxa Ponderada (%)', ascending=True)

fig = go.Figure()
fig.add_bar(y=df_prev['Modalidade'], x=df_prev['Taxa Amostral (%)'], orientation='h', name='Amostra (sem peso)',
            marker_color=COR['neutro'], text=rotulos_pct(df_prev['Taxa Amostral (%)']), textposition='outside')
fig.add_bar(y=df_prev['Modalidade'], x=df_prev['Taxa Ponderada (%)'], orientation='h', name='População (com PESO)',
            marker_color=COR['principal'], text=rotulos_pct(df_prev['Taxa Ponderada (%)']), textposition='outside')
fig.update_layout(barmode='group', height=460, title='Prevalência de apostas digitais: amostra × estimativa ponderada',
                  xaxis_title='Prevalência (%)', xaxis_range=[0, df_prev['Taxa Ponderada (%)'].max() * 1.25])
fig.show()

df_prev.sort_values(by='Taxa Ponderada (%)', ascending=False).reset_index(drop=True)
''')
code(r'''
total = df_prev.set_index('Modalidade').loc['Qualquer Tipo de Aposta (Total)']
ranking = df_prev[df_prev.Modalidade != 'Qualquer Tipo de Aposta (Total)'].sort_values('Taxa Ponderada (%)', ascending=False)
mostrar(f"""
* **Resultado:** {fmt(total['Taxa Amostral (%)'])}% da amostra apostou em alguma modalidade nos últimos 3 meses. Com os pesos, a
  estimativa populacional é {fmt(total['Taxa Ponderada (%)'])}%, cerca de {fmt(total['População Estimada (milhões)'], 1)} milhões de
  pessoas de 10 anos ou mais. A modalidade mais comum é **{ranking.iloc[0]['Modalidade']}** ({fmt(ranking.iloc[0]['Taxa Ponderada (%)'])}%).
* **Interpretação:** a diferença entre as duas barras mostra que a amostra não é autoponderada. A Seção 7.1 acrescenta o
  intervalo de confiança calculado com os pesos replicados.
""")
''')
md(r"""
### 4.2. Perfil demográfico: gênero e idade
""")
code(r'''
# Taxa de adesão por modalidade separada por sexo (análise original)
genero_mods = []
for label, col in mods.items():
    if col != 'C14_AGREG':
        tx_m = df[df['sexo_desc'] == 'Masculino'][col].mean() * 100
        tx_f = df[df['sexo_desc'] == 'Feminino'][col].mean() * 100
        genero_mods.append({'Modalidade': label.split(' (')[0], 'Masculino': tx_m, 'Feminino': tx_f})
df_gen = pd.DataFrame(genero_mods)

idade_ap = df[df['status_aposta'] == 'Apostador']['IDADE']
idade_nao = df[df['status_aposta'] == 'Não Apostador']['IDADE']

fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                    subplot_titles=('Participação por modalidade, segundo o gênero', 'Idade: apostadores × não apostadores'))
for sexo, cor in [('Masculino', COR['principal']), ('Feminino', COR['secundaria'])]:
    fig.add_bar(x=df_gen['Modalidade'], y=df_gen[sexo], name=sexo, marker_color=cor,
                text=rotulos_pct(df_gen[sexo]), textposition='outside', row=1, col=1)
grade_idade = np.linspace(10, 100, 181)
for serie, nome, cor in [(idade_ap, f'Apostadores (mediana {idade_ap.median():.0f})', COR['destaque']),
                         (idade_nao, f'Não apostadores (mediana {idade_nao.median():.0f})', COR['neutro'])]:
    densidade = stats.gaussian_kde(serie.dropna())(grade_idade)   # KDE, como no estudo original
    fig.add_scatter(x=grade_idade, y=densidade, mode='lines', fill='tozeroy', name=nome, line=dict(color=cor, width=2),
                    row=1, col=2)
    fig.add_vline(x=serie.median(), line_dash='dash', line_color=cor, row=1, col=2)
fig.update_layout(barmode='group', height=480, title='Perfil demográfico dos apostadores')
fig.update_yaxes(title_text='Taxa de participação (%)', row=1, col=1)
fig.update_yaxes(title_text='Densidade', row=1, col=2)
fig.update_xaxes(title_text='Idade (anos)', row=1, col=2)
fig.show()

# Boxplot de idade por modalidade específica (análise original)
dados_idade_mods = []
for label, col in [('Cassino Online', 'aposta_cassino'), ('Rifas Digitais', 'aposta_rifa'),
                   ('Apostas Esportivas', 'aposta_esportiva'), ('Loteria Online', 'aposta_loteria')]:
    sub = df[df[col] == 1][['IDADE']].copy()
    sub['Modalidade'] = label
    dados_idade_mods.append(sub)
df_idade_mods = pd.concat(dados_idade_mods, ignore_index=True)

fig = go.Figure()
for (nome, sub), cor in zip(df_idade_mods.groupby('Modalidade', sort=False), COR_MOD.values()):
    fig.add_box(y=sub['IDADE'], name=nome, marker_color=cor, boxmean=True)
fig.update_layout(height=430, showlegend=False, title='Idade dos praticantes de cada modalidade', yaxis_title='Idade (anos)')
fig.show()

df_idade_mods.groupby('Modalidade')['IDADE'].describe()[['mean', 'std', '50%', 'min', 'max']]
''')
code(r'''
taxa_sexo = df.groupby('sexo_desc')['C14_AGREG'].mean()
g = df_gen.set_index('Modalidade')
mostrar(f"""
* **Resultado:** {fmt_pct(taxa_sexo['Masculino'])} dos homens e {fmt_pct(taxa_sexo['Feminino'])} das mulheres da amostra apostaram.
  Nas apostas esportivas, a diferença é grande ({fmt(g.loc['Apostas Esportivas', 'Masculino'])}% × {fmt(g.loc['Apostas Esportivas', 'Feminino'])}%).
  No cassino online, as taxas ficam próximas ({fmt(g.loc['Cassino Online', 'Masculino'])}% × {fmt(g.loc['Cassino Online', 'Feminino'])}%).
  A mediana de idade é {idade_ap.median():.0f} anos entre apostadores e {idade_nao.median():.0f} entre não apostadores.
* **Interpretação:** o gênero parece mudar **qual** jogo se pratica, mais do que **se** a pessoa aposta (hipótese H1). A
  diferença de idade (H4) é parcialmente inflada pelo universo: os não internautas, mais velhos, estão todos no grupo "não
  apostador". A Seção 7.5 refaz a comparação só entre internautas.
""")
''')
md(r"""
### 4.3. Perfil socioeconômico: classe, escolaridade e ocupação
""")
code(r'''
# Taxas por classe, escolaridade e tipo de ocupação (análise original)
classe_order = ['Classe DE', 'Classe C', 'Classe B', 'Classe A']
tx_classe = df.groupby('classe_desc')['C14_AGREG'].mean().reindex(classe_order) * 100
escolaridade_order = ['Analfabeto/Ed. Infantil', 'Fundamental', 'Médio', 'Superior']
tx_esc = df.groupby('escolaridade_desc')['C14_AGREG'].mean().reindex(escolaridade_order) * 100
ocup_validas = df[df['tipo_ocupacao_desc'].isin(['Formal', 'Informal'])]
tx_ocup = ocup_validas.groupby('tipo_ocupacao_desc')['C14_AGREG'].mean() * 100

fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.08,
                    subplot_titles=('Por classe econômica', 'Por escolaridade', 'Por tipo de ocupação'))
for col, serie, cor in [(1, tx_classe, COR['seq'][3]), (2, tx_esc, COR['principal']), (3, tx_ocup, COR['secundaria'])]:
    fig.add_bar(x=list(serie.index), y=serie.values, marker_color=cor, text=rotulos_pct(serie.values),
                textposition='outside', showlegend=False, row=1, col=col)
fig.update_yaxes(title_text='Taxa de apostadores (%)', row=1, col=1)
fig.update_yaxes(range=[0, max(tx_classe.max(), tx_esc.max(), tx_ocup.max()) * 1.2])
fig.update_layout(height=470, title='Prevalência de apostas por perfil socioeconômico (amostra)')
fig.show()

# Decomposição das modalidades nas classes extremas (análise original)
decomp = pd.DataFrame({cl: {lbl: df[df['classe_desc'] == cl][col].mean() * 100
                            for lbl, col in [('Cassino Online', 'aposta_cassino'), ('Rifas', 'aposta_rifa'),
                                             ('Bets Esportivas', 'aposta_esportiva'), ('Loteria', 'aposta_loteria')]}
                       for cl in ['Classe DE', 'Classe A']})
decomp.loc['n (indivíduos)'] = [int((df['classe_desc'] == cl).sum()) for cl in decomp.columns]
decomp
''')
code(r'''
mostrar(f"""
* **Resultado:** a taxa de apostadores cresce da classe DE ({fmt(tx_classe['Classe DE'])}%) para a classe A ({fmt(tx_classe['Classe A'])}%) e é
  maior no ensino médio ({fmt(tx_esc['Médio'])}%) do que no superior ({fmt(tx_esc['Superior'])}%). A classe A tem só
  {int(decomp.loc['n (indivíduos)', 'Classe A'])} respondentes, então as taxas dela são instáveis.
* **Interpretação:** parte do gradiente de classe pode refletir **acesso à Internet**, menor na classe DE. A Seção 7.4 separa os
  dois efeitos ao repetir a comparação entre internautas.
""")
''')
md(r"""
### 4.4. Panorama geográfico
""")
code(r'''
# Taxas por região e área (análise original)
reg_order = ['Sudeste', 'Sul', 'Centro-Oeste', 'Nordeste', 'Norte']
tx_reg = df.groupby('regiao_desc')['C14_AGREG'].mean().reindex(reg_order) * 100
tx_area = df.groupby('area_desc')['C14_AGREG'].mean() * 100

fig = make_subplots(rows=1, cols=2, column_widths=[0.65, 0.35], subplot_titles=('Por macrorregião', 'Urbana × rural'))
fig.add_bar(x=reg_order, y=tx_reg.values, marker_color=COR['principal'], text=rotulos_pct(tx_reg.values),
            textposition='outside', showlegend=False, row=1, col=1)
fig.add_bar(x=list(tx_area.index), y=tx_area.values, marker_color=[COR['principal'], COR['terciaria']],
            text=rotulos_pct(tx_area.values), textposition='outside', showlegend=False, row=1, col=2)
fig.update_yaxes(title_text='Taxa de apostadores (%)', row=1, col=1)
fig.update_yaxes(range=[0, max(tx_reg.max(), tx_area.max()) * 1.2])
fig.update_layout(height=430, title='Prevalência de apostas por região e área (amostra)')
fig.show()
mostrar(f"""
* **Resultado:** a taxa varia de {fmt(tx_reg.min())}% ({tx_reg.idxmin()}) a {fmt(tx_reg.max())}% ({tx_reg.idxmax()}); na área urbana é
  {fmt(tx_area['Urbana'])}% e na rural, {fmt(tx_area['Rural'])}%.
* **Interpretação:** é um retrato descritivo. Região e área não entram nos testes nem no modelo do estudo original.
""")
''')
md(r"""
### 4.5. Ecossistema digital: Pix, dispositivo e influenciadores
""")
code(r'''
# Taxas por uso de Pix, dispositivo e influenciadores (análise original)
tx_pix = df.groupby('C8_I')['C14_AGREG'].mean() * 100
disp_validos = df[df['dispositivos_desc'].isin(['Apenas Celular', 'Ambos (PC e Celular)', 'Apenas Computador'])]
tx_disp = disp_validos.groupby('dispositivos_desc')['C14_AGREG'].mean().sort_values(ascending=False) * 100
tx_infl = df.groupby('TC4B_I')['aposta_cassino'].mean() * 100

fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.08,
                    subplot_titles=('Aposta × uso de Pix', 'Aposta × dispositivo', 'Cassino × influenciadores'))
fig.add_bar(x=['Não usa Pix', 'Usa Pix'], y=tx_pix.values, marker_color=[COR['neutro'], COR['principal']],
            text=rotulos_pct(tx_pix.values), textposition='outside', showlegend=False, row=1, col=1)
fig.add_bar(x=list(tx_disp.index), y=tx_disp.values, marker_color=COR['seq'][3], text=rotulos_pct(tx_disp.values),
            textposition='outside', showlegend=False, row=1, col=2)
fig.add_bar(x=['Não assiste', 'Assiste'], y=tx_infl.values, marker_color=[COR['neutro'], COR['destaque']],
            text=rotulos_pct(tx_infl.values), textposition='outside', showlegend=False, row=1, col=3)
fig.update_yaxes(title_text='Taxa (%)', row=1, col=1)
fig.update_yaxes(range=[0, max(tx_pix.max(), tx_disp.max(), tx_infl.max()) * 1.2])
fig.update_layout(height=470, title='Apostas e hábitos digitais (amostra, população de 10 anos ou mais)')
fig.show()

# Razão de prevalência calculada (análise original)
rp_infl = tx_infl[1] / tx_infl[0]
print(f"Razão de prevalência (cassino): quem assiste a influenciadores / quem não assiste = {rp_infl:.2f}")
''')
code(r'''
# Verificação: as mesmas taxas só entre internautas
inter = df[df['internauta'] == 1]
comparacao_universo = pd.DataFrame({
    'População 10+ (análise original)': [tx_pix[0], tx_pix[1], tx_infl[0], tx_infl[1]],
    'Só internautas': [inter.groupby('C8_I')['C14_AGREG'].mean()[0] * 100, inter.groupby('C8_I')['C14_AGREG'].mean()[1] * 100,
                       inter.groupby('TC4B_I')['aposta_cassino'].mean()[0] * 100, inter.groupby('TC4B_I')['aposta_cassino'].mean()[1] * 100],
}, index=['Aposta | não usa Pix', 'Aposta | usa Pix', 'Cassino | não assiste influenciadores', 'Cassino | assiste influenciadores'])
rp_pix = comparacao_universo.iloc[1] / comparacao_universo.iloc[0]
rp_infl_u = comparacao_universo.iloc[3] / comparacao_universo.iloc[2]
mostrar(f"""
* **Resultado:** na população toda, quem usa Pix aposta {fmt(rp_pix.iloc[0], 1)} vezes mais que quem não usa. Entre internautas, a razão cai
  para {fmt(rp_pix.iloc[1], 1)}. Para influenciadores × cassino, a razão de prevalência vai de {fmt(rp_infl_u.iloc[0], 2)} para {fmt(rp_infl_u.iloc[1], 2)}.
* **Interpretação:** o grupo "não usa Pix" da análise original inclui os {fmt(n_nao_internautas, 0)} não internautas, que não podem apostar
  online. A taxa entre usuários de Pix não muda, porque todos são internautas. O que muda é o grupo de comparação. A associação
  continua forte, mas a análise original a superestima.
""")
comparacao_universo
''')
md(r"""
### 4.6. Sobreposição de modalidades
""")
code(r'''
# Correlação entre as modalidades e tabela cruzada bets × cassino (análise original)
cols_corr = ['aposta_cassino', 'aposta_esportiva', 'aposta_rifa', 'aposta_loteria']
labels_corr = ['Cassino (Tigrinho)', 'Bets Esportivas', 'Rifas Digitais', 'Loteria Online']

corr_matrix = df[cols_corr].corr()
corr_matrix.columns = labels_corr
corr_matrix.index = labels_corr

fig = go.Figure(go.Heatmap(z=corr_matrix.values, x=labels_corr, y=labels_corr, colorscale='Blues', zmin=0, zmax=0.4,
                           text=[[fmt(v, 3) for v in linha] for linha in corr_matrix.values], texttemplate='%{text}',
                           colorbar=dict(title='r')))
fig.update_layout(height=470, width=620, title='Correlação entre as modalidades (coeficiente phi)', yaxis_autorange='reversed')
fig.show()

cross_bet_cassino = pd.crosstab(df['aposta_esportiva'].map({1: 'Sim', 0: 'Não'}),
                                df['aposta_cassino'].map({1: 'Sim', 0: 'Não'}), margins=True)
cross_bet_cassino
''')
code(r'''
pct_bets_cassino = cross_bet_cassino.loc['Sim', 'Sim'] / cross_bet_cassino.loc['Sim', 'All']
mostrar(f"""
* **Resultado:** entre os {fmt(cross_bet_cassino.loc['Sim', 'All'], 0)} apostadores esportivos, {fmt_pct(pct_bets_cassino)} também apostam
  em cassino online. Todas as correlações entre modalidades são positivas. A maior é entre cassino e bets
  ({fmt(corr_matrix.loc['Cassino (Tigrinho)', 'Bets Esportivas'], 3)}).
* **Interpretação:** as modalidades não são públicos separados. Isso importa na Seção 7.6: quem pratica várias modalidades
  aparece em mais de um grupo do Kruskal-Wallis.
""")
''')
md(r"""
### 4.7. Síntese da EDA
""")
code(r'''
mostrar(f"""
1. **Prevalência:** {fmt(total['Taxa Ponderada (%)'])}% da população de 10 anos ou mais apostou online em 3 meses (estimativa ponderada).
2. **Gênero:** homens apostam mais ({fmt_pct(taxa_sexo['Masculino'])} × {fmt_pct(taxa_sexo['Feminino'])}), sobretudo em bets esportivas.
3. **Idade:** apostadores são mais jovens (mediana {idade_ap.median():.0f} × {idade_nao.median():.0f} anos).
4. **Classe:** a taxa cresce da classe DE para a A, mas o acesso à Internet pode explicar parte do gradiente.
5. **Digital:** Pix e vídeos de influenciadores estão fortemente associados à aposta. Na população toda, essas associações
   são infladas pela presença dos não internautas no grupo de comparação.
""")
''')

# =============================================================================
# 5. HIPÓTESES
# =============================================================================
md(r"""
## 5. Formulação das hipóteses estatísticas

**O que vamos fazer?** Formalizar as hipóteses testadas no estudo original, com $\alpha = 0{,}05$.

**Por que isso é necessário?** Escrever $H_0$ e $H_1$ antes do teste deixa claro o que cada p-valor responde.

### E1: prevalência (estimação, não teste)
Estimar, com IC de 95%, a proporção de apostadores no total e por modalidade.

### H1: gênero × modalidade preferida (entre apostadores de bets e/ou cassino)
* $H_0$: gênero e modalidade preferida (só bets, só cassino, ambas) são independentes.
* $H_1$: existe associação entre gênero e modalidade preferida.

### H2: influenciadores × cassino online
* $H_0$: assistir a vídeos de influenciadores e apostar em cassino são independentes ($OR = 1$).
* $H_1$: há associação ($OR \neq 1$). O estudo original espera $OR > 1$.

### H3: classe econômica × aposta
* $H_0$: $p_{DE} = p_C = p_B = p_A$ (mesma proporção de apostadores em todas as classes).
* $H_1$: ao menos uma classe difere.

### H4: idade de apostadores × não apostadores
* $H_0$: as distribuições de idade são iguais.
* $H_1$ (unilateral): a idade dos apostadores é estocasticamente menor.

### H5: idade entre as quatro modalidades
* $H_0$: as distribuições de idade dos praticantes das quatro modalidades são iguais.
* $H_1$: ao menos uma modalidade difere.

### H6: modelo multivariado
* $H_0$ global: todos os coeficientes, exceto o intercepto, são nulos (teste da razão de verossimilhança).
* $H_0$ para cada preditor: $\beta_j = 0$, ou seja, $aOR_j = 1$, mantidos os demais constantes (teste de Wald).
""")

# =============================================================================
# 6. SELEÇÃO DOS TESTES
# =============================================================================
md(r"""
## 6. Seleção e justificativa dos testes estatísticos

**O que vamos fazer?** Justificar cada procedimento do estudo original e explicar as camadas de verificação.

| Hipótese | Teste do estudo original | Tamanho de efeito | Por que este teste |
|---|---|---|---|
| E1 | IC de Wald para proporção | Margem de erro | $n$ grande e $p$ longe de 0 e 1 |
| H1 | Qui-quadrado de independência (2 × 3) | V de Cramér | Duas variáveis nominais. Todas as frequências esperadas passam de 5 |
| H2 | Qui-quadrado com correção de Yates (2 × 2) | OR com IC de Woolf | Tabela 2 × 2. O OR é a medida natural de associação |
| H3 | Qui-quadrado de homogeneidade (4 × 2) | Resíduos ajustados de Haberman | Compara proporções entre 4 grupos. Os resíduos mostram quais classes se afastam do esperado |
| H4 | Mann-Whitney U unilateral | r = \|Z\|/√N | A idade é assimétrica e limitada (10 a 100), então se usa um teste de posição sem supor normalidade |
| H5 | Kruskal-Wallis, com pós-teste Mann-Whitney + Bonferroni | Comparações par a par | Extensão não paramétrica do Mann-Whitney para k grupos. Bonferroni controla o erro nas 6 comparações |
| H6 | Regressão logística por IRLS, Wald e razão de verossimilhança | aOR, McFadden, AUC | O desfecho é binário. O modelo controla confundidores simultaneamente |

### 6.1. Pressupostos que o estudo original não verifica
1. **Universo.** Os testes H2, H4 e H6 incluem os não internautas, que não podem apostar online e são codificados como 0 em Pix, vídeos e jogos. Isso cria associação mecânica. **Verificação:** repetir a análise só entre internautas.
2. **Independência dos grupos em H5.** Uma pessoa que aposta em cassino e em bets entra nos dois grupos do Kruskal-Wallis, mas o teste supõe grupos independentes. **Verificação:** comparar com estimativas por desenho, cujas réplicas preservam a dependência.
3. **Plano amostral.** Todos os testes supõem observações independentes e de mesmo peso. A TIC Domicílios usa estratos, conglomerados e pesos desiguais, e ignorar os conglomerados tende a **subestimar** os erros-padrão.

### 6.2. Camada por desenho
Para cada hipótese, estimamos uma quantidade análoga com `PESO` e calculamos o erro-padrão pelos **200 pesos replicados**:
$$\widehat{V}(\hat\theta) = \frac{1}{R-1}\sum_{r=1}^{R}\left(\hat\theta_r - \bar\theta\right)^2, \quad R = 200.$$
Daí saem ICs de 95% e testes de Wald, sem supor independência entre observações nem entre grupos.

| Hipótese | Quantidade estimada por desenho |
|---|---|
| E1 | Proporção ponderada, IC por réplicas e efeito de desenho (deff) |
| H1 | Diferença homens − mulheres na participação de cada modalidade preferida; Wald conjunto (2 g.l.) |
| H2 | OR ponderado entre internautas, com IC por réplicas |
| H3 | Proporção ponderada por classe (população e internautas); Wald de igualdade (3 g.l.) |
| H4 | Diferença de idade média ponderada (apostadores − não apostadores), população e internautas |
| H5 | Idade média ponderada por modalidade; Wald de igualdade (3 g.l.) |
| H6 | Logística ponderada entre internautas, com EP por réplicas |
""")

# =============================================================================
# 7. EXECUÇÃO
# =============================================================================
md(r"""
## 7. Execução dos testes de hipóteses e do modelo

**Como ler esta seção.** Cada subseção tem a célula da **análise original**, com o mesmo código e a mesma lógica do estudo de origem, seguida da célula de **verificação**, com o universo de internautas e a camada por desenho. Os resultados são interpretados no fim de cada subseção.

### 7.1. E1: intervalos de confiança para as prevalências
$$IC_{95\%}(\hat{p}) = \hat{p} \pm z_{0,025} \cdot \sqrt{\frac{\hat{p}(1 - \hat{p})}{n}}$$
""")
code(r'''
# Análise original
n_total = len(df)
z_critico = stats.norm.ppf(0.975)  # 1.95996

modalidades_ic = {
    'Total de Apostadores (C14_AGREG)': 'C14_AGREG',
    'Cassino Online (Tigrinho)': 'aposta_cassino',
    'Rifas Digitais / Sorteios': 'aposta_rifa',
    'Apostas Esportivas (Bets)': 'aposta_esportiva',
    'Loteria Federal Online': 'aposta_loteria'
}

resultados_ic = []
peso_total = df['PESO'].sum()

for label, col in modalidades_ic.items():
    p_hat = df[col].mean()
    se = np.sqrt(p_hat * (1 - p_hat) / n_total)
    ic_inf = max(0, p_hat - z_critico * se)
    ic_sup = min(1, p_hat + z_critico * se)
    margem_erro = z_critico * se * 100

    # Estimativa Ponderada Expandida
    p_pond = (df[col] * df['PESO']).sum() / peso_total

    resultados_ic.append({
        'Modalidade': label,
        'Proporção Amostral (p̂)': f"{p_hat*100:.2f}%",
        'Margem de Erro (±%)': f"{margem_erro:.2f}%",
        'IC 95% Inferior': f"{ic_inf*100:.2f}%",
        'IC 95% Superior': f"{ic_sup*100:.2f}%",
        'Estimativa Ponderada': f"{p_pond*100:.2f}%"
    })

df_ic = pd.DataFrame(resultados_ic)
df_ic
''')
code(r'''
# Verificação: IC por desenho (PESO + 200 réplicas) e efeito de desenho
linhas = []
for label, col in modalidades_ic.items():
    y = df[col].to_numpy(float)
    est = estimativa_com_replicas(lambda w, y=y: prop_ponderada(y, w))
    p, ep = est['estimativa'][0], est['ep'][0]
    n_val = int(np.sum(~np.isnan(y)))
    linhas.append({'Modalidade': label, 'Ponderada': p, 'EP (réplicas)': ep, 'IC 95% inf': est['ic_inf'][0],
                   'IC 95% sup': est['ic_sup'][0], 'Margem (±)': 1.96 * ep,
                   'deff': ep**2 / (p * (1 - p) / n_val)})
ic_desenho = pd.DataFrame(linhas).set_index('Modalidade')
ic_desenho.style.format({c: (lambda v: fmt_pct(v, 2)) for c in ['Ponderada', 'EP (réplicas)', 'IC 95% inf', 'IC 95% sup', 'Margem (±)']} | {'deff': lambda v: fmt(v, 2)})
''')
code(r'''
t_d = ic_desenho.loc['Total de Apostadores (C14_AGREG)']
mostrar(f"""
* **Resultado:** a prevalência populacional de apostas é {fmt_pct(t_d['Ponderada'], 2)}, com IC de 95% por desenho de
  [{fmt_pct(t_d['IC 95% inf'], 2)}; {fmt_pct(t_d['IC 95% sup'], 2)}]. O efeito de desenho (deff) varia de {fmt(ic_desenho['deff'].min(), 1)} a
  {fmt(ic_desenho['deff'].max(), 1)}.
* **Interpretação:** o IC da análise original está centrado na proporção **amostral** e usa a variância de amostra aleatória simples.
  Por isso, ele não cobre a estimativa ponderada e é estreito demais. Um deff de {fmt(t_d['deff'], 1)} significa que a variância real é
  {fmt(t_d['deff'], 1)} vezes a de uma amostra aleatória simples do mesmo tamanho. **Para falar do Brasil, o IC correto é o da camada por desenho.**
""")
''')
md(r"""
### 7.2. H1: gênero × modalidade preferida (qui-quadrado e V de Cramér)
$$\chi^2 = \sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}, \qquad V = \sqrt{\frac{\chi^2}{n \cdot (\min(r, c) - 1)}}$$
""")
code(r'''
# Análise original
# Filtramos apostadores de esportes e/ou cassino
df_apostadores_jogos = df[(df['aposta_esportiva'] == 1) | (df['aposta_cassino'] == 1)].copy()

# Criamos categorias: Apenas Bets, Apenas Cassino, Ambos
def categorizar_jogo(row):
    if row['aposta_esportiva'] == 1 and row['aposta_cassino'] == 1:
        return 'Ambos (Bets e Cassino)'
    elif row['aposta_esportiva'] == 1:
        return 'Apenas Bets Esportivas'
    elif row['aposta_cassino'] == 1:
        return 'Apenas Cassino (Tigrinho)'
    return np.nan

df_apostadores_jogos['tipo_preferido'] = df_apostadores_jogos.apply(categorizar_jogo, axis=1)

# Tabela de Contingência
tabela_genero_jogo = pd.crosstab(df_apostadores_jogos['sexo_desc'],
                                 df_apostadores_jogos['tipo_preferido'])

# Teste Qui-Quadrado
chi2_val, p_val, dof, esperados = stats.chi2_contingency(tabela_genero_jogo)

# V de Cramér
n = tabela_genero_jogo.values.sum()
min_dim = min(tabela_genero_jogo.shape) - 1
cramer_v = np.sqrt(chi2_val / (n * min_dim))

print("Tabela observada:")
print(tabela_genero_jogo)
print("\nPercentual por gênero (% da linha):")
print(pd.crosstab(df_apostadores_jogos['sexo_desc'], df_apostadores_jogos['tipo_preferido'], normalize='index').mul(100).round(2))
print(f"\nχ² = {chi2_val:.4f} | g.l. = {dof} | p = {p_val:.4e} | V de Cramér = {cramer_v:.4f}")
print(f"Menor frequência esperada: {esperados.min():.1f}")
print("Decisão:", "rejeita H0" if p_val < ALPHA else "não rejeita H0")
''')
code(r'''
# Verificação: participação ponderada de cada modalidade preferida, por gênero (camada por desenho)
tipos = ['Apenas Bets Esportivas', 'Apenas Cassino (Tigrinho)', 'Ambos (Bets e Cassino)']
tipo_full = pd.Series(np.nan, index=df.index, dtype=object)
tipo_full.loc[df_apostadores_jogos.index] = df_apostadores_jogos['tipo_preferido']
masc = (df['sexo_desc'] == 'Masculino').to_numpy()
jogador = tipo_full.notna().to_numpy()


def participacoes(w):
    saida = []
    for sexo in (masc, ~masc):
        base = jogador & sexo
        saida += [w[base & (tipo_full == t).to_numpy()].sum() / w[base].sum() for t in tipos]
    return np.array(saida)


est_h1 = estimativa_com_replicas(participacoes)
dif_h1 = est_h1['estimativa'][:3] - est_h1['estimativa'][3:]
reps_dif = est_h1['replicas'][:, :3] - est_h1['replicas'][:, 3:]
ep_dif = reps_dif.std(axis=0, ddof=1)
X2_h1, gl_h1, p_h1_des = wald_zero(dif_h1[:2], reps_dif[:, :2])   # a 3ª diferença é determinada pelas outras duas
h1_desenho = pd.DataFrame({'Homens (pond.)': est_h1['estimativa'][:3], 'Mulheres (pond.)': est_h1['estimativa'][3:],
                           'Diferença H − M': dif_h1, 'IC 95% inf': dif_h1 - 1.96 * ep_dif, 'IC 95% sup': dif_h1 + 1.96 * ep_dif},
                          index=tipos)
print(f"Wald por desenho (H0: mesma distribuição em homens e mulheres): X² = {X2_h1:.1f}, g.l. = {gl_h1}, p = {p_h1_des:.2e}")
h1_desenho.style.format(lambda v: fmt_pct(v))
''')
code(r'''
pct_l = pd.crosstab(df_apostadores_jogos['sexo_desc'], df_apostadores_jogos['tipo_preferido'], normalize='index')
mostrar(f"""
* **Resultado:** entre as {fmt(n, 0)} pessoas que apostam em bets e/ou cassino, {fmt_pct(pct_l.loc['Masculino', 'Apenas Bets Esportivas'])} dos
  homens apostam só em bets, contra {fmt_pct(pct_l.loc['Feminino', 'Apenas Bets Esportivas'])} das mulheres. Já {fmt_pct(pct_l.loc['Feminino', 'Apenas Cassino (Tigrinho)'])}
  das mulheres apostam só em cassino, contra {fmt_pct(pct_l.loc['Masculino', 'Apenas Cassino (Tigrinho)'])} dos homens.
  χ² = {fmt(chi2_val, 1)} ({dof} g.l.), {fmt_p(p_val)}, V de Cramér = {fmt(cramer_v, 3)}. Na camada por desenho, o Wald conjunto também
  rejeita $H_0$ ({fmt_p(p_h1_des)}) e todas as diferenças têm IC que exclui o zero.
* **Interpretação:** **rejeitamos $H_0$**. A associação entre gênero e modalidade é de intensidade moderada e se mantém com os pesos
  e o plano amostral. Esse é o resultado mais robusto do estudo, porque o universo (só apostadores) não sofre do problema dos não internautas.
""")
''')
md(r"""
### 7.3. H2: influenciadores digitais × cassino online (qui-quadrado e OR)
$$OR = \frac{a \cdot d}{b \cdot c}, \qquad IC_{95\%}(OR) = \exp\left( \ln(OR) \pm 1{,}96 \sqrt{\tfrac{1}{a} + \tfrac{1}{b} + \tfrac{1}{c} + \tfrac{1}{d}} \right)$$
""")
code(r'''
# Análise original
# Tabela 2x2
tabela_infl = pd.crosstab(df['TC4B_I'].map({1: 'Assiste Influenciadores', 0: 'Não Assiste'}),
                          df['aposta_cassino'].map({1: 'Joga Cassino', 0: 'Não Joga'}))

# Ordenação para matriz 2x2 padrão
tabela_infl = tabela_infl.loc[['Assiste Influenciadores', 'Não Assiste'], ['Joga Cassino', 'Não Joga']]

chi2_infl, p_infl, dof_infl, exp_infl = stats.chi2_contingency(tabela_infl, correction=True)

# Cálculo de Odds Ratio e IC 95%
a = tabela_infl.loc['Assiste Influenciadores', 'Joga Cassino']
b = tabela_infl.loc['Assiste Influenciadores', 'Não Joga']
c = tabela_infl.loc['Não Assiste', 'Joga Cassino']
d = tabela_infl.loc['Não Assiste', 'Não Joga']

odds_ratio = (a * d) / (b * c)
se_ln_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
ic_or_inf = np.exp(np.log(odds_ratio) - 1.96 * se_ln_or)
ic_or_sup = np.exp(np.log(odds_ratio) + 1.96 * se_ln_or)

print(tabela_infl)
print(f"\nχ² (Yates) = {chi2_infl:.4f} | p = {p_infl:.4e}")
print(f"OR = {odds_ratio:.4f} | IC 95%: [{ic_or_inf:.4f} ; {ic_or_sup:.4f}]")
print("Decisão:", "rejeita H0" if p_infl < ALPHA else "não rejeita H0")
''')
code(r'''
# Verificação 1: o mesmo procedimento, só entre internautas
sub = df[df['internauta'] == 1]
tabela_infl_int = pd.crosstab(sub['TC4B_I'].map({1: 'Assiste Influenciadores', 0: 'Não Assiste'}),
                              sub['aposta_cassino'].map({1: 'Joga Cassino', 0: 'Não Joga'})
                              ).loc[['Assiste Influenciadores', 'Não Assiste'], ['Joga Cassino', 'Não Joga']]
assert np.isclose(qui2_e_or(tabela_infl)['OR'], odds_ratio)   # a função reproduz a análise original
h2_int = qui2_e_or(tabela_infl_int)

# Verificação 2: OR ponderado entre internautas, com IC por réplicas
infl = df['TC4B_I'].to_numpy(float)
cass = df['aposta_cassino'].to_numpy(float)
validos = INTERNAUTA & ~np.isnan(infl) & ~np.isnan(cass)


def log_or_ponderado(w):
    cel = lambda i, j: w[validos & (infl == i) & (cass == j)].sum()
    return np.log(cel(1, 1) * cel(0, 0) / (cel(1, 0) * cel(0, 1)))


est_h2 = estimativa_com_replicas(log_or_ponderado)
h2_tabela = pd.DataFrame({
    'OR': [odds_ratio, h2_int['OR'], np.exp(est_h2['estimativa'][0])],
    'IC 95% inf': [ic_or_inf, h2_int['ic'][0], np.exp(est_h2['ic_inf'][0])],
    'IC 95% sup': [ic_or_sup, h2_int['ic'][1], np.exp(est_h2['ic_sup'][0])],
}, index=['Análise original (população 10+, sem peso)', 'Internautas, sem peso', 'Internautas, ponderado (réplicas)'])
print(tabela_infl_int)
print(f"\nInternautas: χ² (Yates) = {h2_int['chi2']:.1f} | p = {h2_int['p']:.2e}")
h2_tabela
''')
code(r'''
mostrar(f"""
* **Resultado:** a análise original estima OR = {fmt(odds_ratio)} [{fmt(ic_or_inf)}; {fmt(ic_or_sup)}]. Entre internautas, o OR cai para
  {fmt(h2_int['OR'])} [{fmt(h2_int['ic'][0])}; {fmt(h2_int['ic'][1])}]. Com pesos e réplicas, fica em {fmt(h2_tabela.iloc[2, 0])}
  [{fmt(h2_tabela.iloc[2, 1])}; {fmt(h2_tabela.iloc[2, 2])}].
* **Interpretação:** **rejeitamos $H_0$** nas três versões. Quem assiste a influenciadores tem chance maior de apostar em cassino.
  Mas a magnitude original está inflada: os não internautas entram como "não assiste" e "não joga" ao mesmo tempo. O IC por desenho é mais
  largo porque incorpora o plano amostral. O OR é bivariado: a Seção 7.7 mostra quanto dele sobra depois de controlar idade, gênero e outros fatores.
""")
''')
md(r"""
### 7.4. H3: classe econômica × aposta (homogeneidade e resíduos de Haberman)
$$z_{ij} = \frac{O_{ij} - E_{ij}}{\sqrt{E_{ij} \cdot (1 - p_{i\cdot}) \cdot (1 - p_{\cdot j})}}, \qquad |z_{ij}| > 1{,}96 \Rightarrow \text{célula se afasta do esperado a 5\%}$$
""")
code(r'''
# Análise original
tabela_classe = pd.crosstab(df['classe_desc'], df['status_aposta'])
tabela_classe = tabela_classe.loc[['Classe DE', 'Classe C', 'Classe B', 'Classe A']]

chi2_cl, p_cl, dof_cl, exp_cl = stats.chi2_contingency(tabela_classe)

# Cálculo dos Resíduos Padronizados Ajustados
n_total_cl = tabela_classe.values.sum()
row_sums = tabela_classe.sum(axis=1).values
col_sums = tabela_classe.sum(axis=0).values

residuos_ajustados = np.zeros(tabela_classe.shape)
for i in range(len(row_sums)):
    for j in range(len(col_sums)):
        o = tabela_classe.iloc[i, j]
        e = exp_cl[i, j]
        denom = np.sqrt(e * (1 - row_sums[i] / n_total_cl) * (1 - col_sums[j] / n_total_cl))
        residuos_ajustados[i, j] = (o - e) / denom

df_residuos = pd.DataFrame(residuos_ajustados, index=tabela_classe.index, columns=tabela_classe.columns)

print(tabela_classe)
print(f"\nχ² = {chi2_cl:.4f} | g.l. = {dof_cl} | p = {p_cl:.4e}")
print("\nResíduos ajustados de Haberman:")
print(df_residuos.round(2))
print("Decisão:", "rejeita H0" if p_cl < ALPHA else "não rejeita H0")
''')
code(r'''
# Verificação: proporção ponderada por classe, na população e entre internautas (camada por desenho)
classe = df['classe_desc'].astype(str).to_numpy()
y_ap = df['C14_AGREG'].to_numpy(float)
h3 = {}
for universo, mascara in [('População 10+', np.ones(len(df), bool)), ('Internautas', INTERNAUTA)]:
    est = estimativa_com_replicas(lambda w, m=mascara: [prop_ponderada(y_ap, w, m & (classe == cl)) for cl in classe_order])
    h3[universo] = (est, wald_igualdade(est['estimativa'], est['replicas']))
    sub = df[mascara]
    t = pd.crosstab(sub['classe_desc'], sub['status_aposta']).loc[classe_order]
    h3[universo] += (stats.chi2_contingency(t),)

h3_tabela = pd.DataFrame({f"{u} (pond.)": h3[u][0]['estimativa'] for u in h3} |
                         {f"{u}: IC 95%": [f"[{fmt_pct(i)}; {fmt_pct(s)}]" for i, s in zip(h3[u][0]['ic_inf'], h3[u][0]['ic_sup'])] for u in h3},
                         index=classe_order)
for u in h3:
    X2, gl, p = h3[u][1]
    print(f"{u}: Wald por desenho X² = {X2:.1f} ({gl} g.l.), p = {p:.2e} | qui-quadrado sem peso = {h3[u][2].statistic:.1f}")
h3_tabela.style.format({c: (lambda v: fmt_pct(v)) for c in h3_tabela.columns if 'pond' in c})
''')
code(r'''
pi, pp = h3['Internautas'][0]['estimativa'], h3['População 10+'][0]['estimativa']
p_h3_pop, p_h3_int = h3['População 10+'][1][2], h3['Internautas'][1][2]
h3_int_rejeita = p_h3_int < ALPHA
mostrar(f"""
* **Resultado:** na análise original, χ² = {fmt(chi2_cl, 1)} ({dof_cl} g.l.), {fmt_p(p_cl)}. A classe DE tem resíduo
  {fmt(df_residuos.loc['Classe DE', 'Apostador'], 1)} (menos apostadores que o esperado) e as classes C, B e A têm resíduos positivos.
  Com pesos, a prevalência vai de {fmt_pct(pp[0])} (DE) a {fmt_pct(pp[3])} (A) na população, e o Wald por desenho rejeita a igualdade
  ({fmt_p(p_h3_pop)}). Entre internautas, vai de {fmt_pct(pi[0])} a {fmt_pct(pi[3])}, com Wald por desenho {fmt_p(p_h3_int)}.
* **Interpretação:** **rejeitamos $H_0$ na população de 10 anos ou mais.** {'Entre internautas, a diferença também é significativa, embora menor.' if h3_int_rejeita else '**Entre internautas, com o plano amostral, não há evidência suficiente de diferença entre as classes.** Os ICs por classe são largos, sobretudo na classe A, e se sobrepõem.'}
  Ou seja, boa parte do gradiente de classe da análise original reflete o **acesso à Internet**, e não uma propensão maior a apostar
  entre os mais ricos que já estão online. Os resíduos de Haberman dependem do tamanho da amostra: com n ≈ 24 mil, valores muito altos
  indicam detecção segura, não necessariamente uma diferença grande.
""")
''')
md(r"""
### 7.5. H4: idade de apostadores × não apostadores (Mann-Whitney U)
$$r = \frac{|Z|}{\sqrt{N}}, \qquad Z = \frac{U - \mu_U}{\sigma_U}$$
""")
code(r'''
# Análise original
idade_apostador = df[df['status_aposta'] == 'Apostador']['IDADE'].dropna()
idade_nao_apostador = df[df['status_aposta'] == 'Não Apostador']['IDADE'].dropna()

# Teste de Mann-Whitney U (unilateral: apostadores mais jovens)
u_stat, p_mann = stats.mannwhitneyu(idade_apostador, idade_nao_apostador, alternative='less')

# Cálculo do Z e do tamanho do efeito r
n1 = len(idade_apostador)
n2 = len(idade_nao_apostador)
mu_u = (n1 * n2) / 2
sigma_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
z_mann = (u_stat - mu_u) / sigma_u
effect_size_r = abs(z_mann) / np.sqrt(n1 + n2)

print(f"n: {n1:,} apostadores × {n2:,} não apostadores")
print(f"Mediana: {idade_apostador.median():.1f} × {idade_nao_apostador.median():.1f} anos | Média: {idade_apostador.mean():.2f} × {idade_nao_apostador.mean():.2f}")
print(f"U = {u_stat:,.2f} | Z = {z_mann:.4f} | p (unilateral) = {p_mann:.4e} | r = {effect_size_r:.4f}")
print("Decisão:", "rejeita H0" if p_mann < ALPHA else "não rejeita H0")
''')
code(r'''
# Verificação 1: o mesmo teste só entre internautas
sub = df[df['internauta'] == 1]
ia, ina = sub.loc[sub['C14_AGREG'] == 1, 'IDADE'], sub.loc[sub['C14_AGREG'] == 0, 'IDADE']
u_i, p_i = stats.mannwhitneyu(ia, ina, alternative='less')
z_i = (u_i - len(ia) * len(ina) / 2) / np.sqrt(len(ia) * len(ina) * (len(ia) + len(ina) + 1) / 12)
r_i = abs(z_i) / np.sqrt(len(ia) + len(ina))

# Verificação 2: diferença de idade média ponderada (camada por desenho)
idade = df['IDADE'].to_numpy(float)
ap = df['C14_AGREG'].to_numpy() == 1
h4 = {}
for universo, m in [('População 10+', np.ones(len(df), bool)), ('Internautas', INTERNAUTA)]:
    h4[universo] = estimativa_com_replicas(lambda w, m=m: prop_ponderada(idade, w, m & ap) - prop_ponderada(idade, w, m & ~ap))

h4_tabela = pd.DataFrame({
    'Mediana apostadores': [idade_apostador.median(), ia.median()],
    'Mediana não apostadores': [idade_nao_apostador.median(), ina.median()],
    'r (Mann-Whitney)': [effect_size_r, r_i],
    'p (unilateral)': [p_mann, p_i],
    'Dif. média ponderada (anos)': [h4[u]['estimativa'][0] for u in h4],
    'IC 95% (réplicas)': [f"[{fmt(h4[u]['ic_inf'][0], 1)}; {fmt(h4[u]['ic_sup'][0], 1)}]" for u in h4],
}, index=['População 10+ (análise original)', 'Internautas'])
h4_tabela
''')
code(r'''
d_pop, d_int = -h4['População 10+']['estimativa'][0], -h4['Internautas']['estimativa'][0]
mostrar(f"""
* **Resultado:** na análise original, a mediana é {idade_apostador.median():.0f} anos entre apostadores e {idade_nao_apostador.median():.0f} entre
  não apostadores (r = {fmt(effect_size_r, 3)}, {fmt_p(p_mann)}). Entre internautas, a mediana dos não apostadores cai para {ina.median():.0f} anos
  e o efeito diminui (r = {fmt(r_i, 3)}, {fmt_p(p_i)}). Com pesos, apostadores são em média {fmt(d_pop, 1)} anos mais jovens na população
  e {fmt(d_int, 1)} anos mais jovens entre internautas (IC [{fmt(-h4['Internautas']['ic_sup'][0], 1)}; {fmt(-h4['Internautas']['ic_inf'][0], 1)}]).
* **Interpretação:** **rejeitamos $H_0$**: apostadores são mais jovens. Cerca de {fmt_pct(1 - d_int / d_pop, 0)} da diferença de idade média vem dos
  não internautas, que são mais velhos e por definição não apostam online. O efeito entre internautas é o que descreve o perfil de quem aposta.
""")
''')
md(r"""
### 7.6. H5: idade entre as quatro modalidades (Kruskal-Wallis e pós-teste de Bonferroni)
""")
code(r'''
# Análise original
idades_cassino = df[df['aposta_cassino'] == 1]['IDADE'].dropna()
idades_rifa = df[df['aposta_rifa'] == 1]['IDADE'].dropna()
idades_bets = df[df['aposta_esportiva'] == 1]['IDADE'].dropna()
idades_loteria = df[df['aposta_loteria'] == 1]['IDADE'].dropna()

# Teste de Kruskal-Wallis
h_stat, p_kw = stats.kruskal(idades_cassino, idades_rifa, idades_bets, idades_loteria)

for nome, s in [('Cassino', idades_cassino), ('Bets', idades_bets), ('Rifas', idades_rifa), ('Loteria', idades_loteria)]:
    print(f"{nome:8s} (n={len(s)}): mediana = {s.median():.1f} | média = {s.mean():.2f}")
print(f"\nH = {h_stat:.4f} | p = {p_kw:.4e}")

# Comparações aos pares com Mann-Whitney e correção de Bonferroni (α ajustado = 0,05 / 6)
pares = [
    ('Cassino vs Bets', idades_cassino, idades_bets),
    ('Cassino vs Loteria', idades_cassino, idades_loteria),
    ('Cassino vs Rifas', idades_cassino, idades_rifa),
    ('Bets vs Loteria', idades_bets, idades_loteria),
    ('Bets vs Rifas', idades_bets, idades_rifa),
    ('Rifas vs Loteria', idades_rifa, idades_loteria)
]
pos_teste = pd.DataFrame([{'Par': nome, 'p-valor': stats.mannwhitneyu(g1, g2).pvalue} for nome, g1, g2 in pares])
pos_teste['Significante (p < 0,0083)'] = pos_teste['p-valor'] < ALPHA / 6
pos_teste
''')
code(r'''
# Verificação: idade média ponderada por modalidade, com Wald de igualdade por réplicas
mods_h5 = {'Cassino': 'aposta_cassino', 'Bets': 'aposta_esportiva', 'Rifas': 'aposta_rifa', 'Loteria': 'aposta_loteria'}
mascaras_h5 = {k: (df[c] == 1).to_numpy() for k, c in mods_h5.items()}
est_h5 = estimativa_com_replicas(lambda w: [prop_ponderada(idade, w, m) for m in mascaras_h5.values()])
X2_h5, gl_h5, p_h5_des = wald_igualdade(est_h5['estimativa'], est_h5['replicas'])
em_varios = int((df[list(mods_h5.values())].sum(axis=1) >= 2).sum())
print(f"Pessoas em 2 ou mais grupos do Kruskal-Wallis: {em_varios}")
print(f"Wald por desenho (H0: mesma idade média nas 4 modalidades): X² = {X2_h5:.1f} ({gl_h5} g.l.), p = {p_h5_des:.2e}")
pd.DataFrame({'Idade média ponderada': est_h5['estimativa'], 'IC 95% inf': est_h5['ic_inf'], 'IC 95% sup': est_h5['ic_sup']},
             index=list(mods_h5)).round(1)
''')
code(r'''
mostrar(f"""
* **Resultado:** H = {fmt(h_stat, 1)} (3 g.l.), {fmt_p(p_kw)}. A loteria tem público mais velho (mediana {idades_loteria.median():.0f} anos) que
  cassino ({idades_cassino.median():.0f}), bets ({idades_bets.median():.0f}) e rifas ({idades_rifa.median():.0f}). No pós-teste de Bonferroni, as três
  comparações com a loteria são significantes; cassino × bets não é. O Wald por desenho confirma a diferença ({fmt_p(p_h5_des)}).
* **Interpretação:** **rejeitamos $H_0$**: a loteria se destaca das demais. Atenção: {fmt(em_varios, 0)} pessoas aparecem em mais de um grupo,
  o que viola a independência suposta pelo Kruskal-Wallis. A conclusão se mantém na camada por desenho, que não depende dessa suposição.
""")
''')
md(r"""
### 7.7. H6: regressão logística multivariada (IRLS)
$$\text{logit}(p_i) = \ln\left( \frac{p_i}{1 - p_i} \right) = \beta_0 + \sum_{j=1}^k \beta_j x_{ij}, \qquad aOR_j = e^{\hat{\beta}_j}$$

A log-verossimilhança $\ln L(\boldsymbol{\beta}) = \sum_i [y_i \ln p_i + (1 - y_i) \ln(1 - p_i)]$ é maximizada por Newton-Raphson (IRLS). A covariância dos estimadores é a inversa da informação de Fisher, $(\mathbf{X}^T \mathbf{W} \mathbf{X})^{-1}$, com $\mathbf{W} = \text{diag}(p_i(1 - p_i))$.

#### 7.7.1. Ajuste (análise original)
""")
code(r'''
# Análise original
# Construção da Matriz de Variáveis Explicativas
features_df = pd.DataFrame(index=df.index)
features_df['intercepto'] = 1.0

# Demográficas
features_df['idade_padronizada'] = (df['IDADE'] - df['IDADE'].mean()) / df['IDADE'].std()
features_df['sexo_masculino'] = (df['sexo_desc'] == 'Masculino').astype(float)
features_df['ensino_medio'] = (df['escolaridade_desc'] == 'Médio').astype(float)
features_df['ensino_superior'] = (df['escolaridade_desc'] == 'Superior').astype(float)

# Socioeconômicas (Referência: Classe DE)
features_df['classe_C'] = (df['classe_desc'] == 'Classe C').astype(float)
features_df['classe_B'] = (df['classe_desc'] == 'Classe B').astype(float)
features_df['classe_A'] = (df['classe_desc'] == 'Classe A').astype(float)
features_df['trabalho_informal'] = (df['tipo_ocupacao_desc'].fillna('Outro') == 'Informal').astype(float)

# Tecnológicas e Hábitos Digitais
features_df['usa_pix'] = df['C8_I'].fillna(0).astype(float)
features_df['assiste_influenciadores'] = df['TC4B_I'].fillna(0).astype(float)
features_df['assiste_videos_esporte'] = df['TC4B_B'].fillna(0).astype(float)
features_df['joga_games_online'] = df['C9_A'].fillna(0).astype(float)
features_df['mobile_only'] = (df['dispositivos_desc'] == 'Apenas Celular').astype(float)

X = features_df.values
y = df['C14_AGREG'].values.astype(float)
nomes_preditores = list(features_df.columns)

print(f"Matriz de desenho: {X.shape[0]:,} observações e {X.shape[1]} colunas.")

# Algoritmo de Otimização Newton-Raphson / IRLS
beta = np.zeros(X.shape[1])
p_bar = y.mean()
beta[0] = np.log(p_bar / (1.0 - p_bar))  # Inicialização pelo logito amostral

for it in range(25):
    eta = np.clip(X @ beta, -20, 20)
    mu = 1.0 / (1.0 + np.exp(-eta))
    w = mu * (1.0 - mu)
    grad = X.T @ (y - mu)
    H = (X.T * w) @ X
    step = np.linalg.solve(H, grad)
    beta += step
    diff = np.linalg.norm(step)
    if diff < 1e-6:
        print(f"Convergência na iteração {it+1} (norma do passo = {diff:.2e}).")
        break

# Cálculo de Erros Padrão, Estatísticas de Teste Wald e Odds Ratios
eta_final = np.clip(X @ beta, -20, 20)
mu_final = 1.0 / (1.0 + np.exp(-eta_final))
w_final = mu_final * (1.0 - mu_final)
H_final = (X.T * w_final) @ X
cov_matrix = np.linalg.inv(H_final)

se_beta = np.sqrt(np.diag(cov_matrix))
z_wald = beta / se_beta
p_values = 2.0 * (1.0 - stats.norm.cdf(np.abs(z_wald)))

odds_ratios = np.exp(beta)
or_ci_lower = np.exp(beta - 1.96 * se_beta)
or_ci_upper = np.exp(beta + 1.96 * se_beta)

# Tabela Consolidada de Resultados
tabela_regressao = pd.DataFrame({
    'Coeficiente (β)': beta,
    'Erro Padrão': se_beta,
    'z de Wald': z_wald,
    'p-valor': p_values,
    'aOR (Odds Ratio)': odds_ratios,
    'IC 95% Inf': or_ci_lower,
    'IC 95% Sup': or_ci_upper
}, index=nomes_preditores)

tabela_regressao.round(4)
''')
md(r"""
#### 7.7.2. Forest plot das razões de chances ajustadas (análise original)
""")
code(r'''
nomes_amigaveis = {
    'usa_pix': 'Usa Pix (vs não)',
    'joga_games_online': 'Joga games online (vs não)',
    'assiste_videos_esporte': 'Assiste vídeos de esporte (vs não)',
    'sexo_masculino': 'Masculino (vs feminino)',
    'mobile_only': 'Só celular (vs PC/ambos)',
    'assiste_influenciadores': 'Assiste influenciadores (vs não)',
    'ensino_medio': 'Ensino médio (vs até fundamental)',
    'ensino_superior': 'Ensino superior (vs até fundamental)',
    'trabalho_informal': 'Trabalho informal (vs formal/outro)',
    'classe_A': 'Classe A (vs DE)',
    'classe_C': 'Classe C (vs DE)',
    'idade_padronizada': 'Idade (+1 desvio-padrão)',
    'classe_B': 'Classe B (vs DE)'
}


def forest(tabela, titulo):
    t = tabela.iloc[1:].sort_values('aOR (Odds Ratio)')
    cores = [COR['principal'] if p < ALPHA else COR['neutro'] for p in t['p-valor']]
    fig = go.Figure(go.Scatter(
        x=t['aOR (Odds Ratio)'], y=[nomes_amigaveis[i] for i in t.index], mode='markers', marker=dict(color=cores, size=10),
        error_x=dict(type='data', symmetric=False, array=t['IC 95% Sup'] - t['aOR (Odds Ratio)'],
                     arrayminus=t['aOR (Odds Ratio)'] - t['IC 95% Inf'], color='#555'),
        text=[f"{fmt(o)} [{fmt(i)}; {fmt(s)}]" for o, i, s in zip(t['aOR (Odds Ratio)'], t['IC 95% Inf'], t['IC 95% Sup'])],
        hovertemplate='%{y}<br>aOR = %{text}<extra></extra>'))
    fig.add_vline(x=1, line_dash='dash', line_color=COR['destaque'])
    fig.update_layout(height=520, title=titulo, xaxis_title='Razão de chances ajustada (escala log)', xaxis_type='log', showlegend=False)
    return fig


forest(tabela_regressao, 'Fatores associados à aposta digital: modelo original (azul = p < 0,05)').show()
''')
md(r"""
#### 7.7.3. Qualidade do ajuste e discriminação (análise original)
* **Pseudo-$R^2$ de McFadden:** $R^2 = 1 - \ln L_{\text{modelo}} / \ln L_{\text{nulo}}$.
* **Razão de verossimilhança:** $G^2 = 2(\ln L_{\text{modelo}} - \ln L_{\text{nulo}}) \sim \chi^2_k$.
* **Curva ROC e AUC:** capacidade de ordenar apostadores acima de não apostadores, em todos os limiares.
* **Matriz de confusão:** limiar igual à proporção amostral de apostadores.
""")
code(r'''
# Análise original
# 1. Métricas de Verossimilhança e Pseudo-R2
ll_modelo = np.sum(y * np.log(mu_final + 1e-15) + (1.0 - y) * np.log(1.0 - mu_final + 1e-15))
p_nulo = np.mean(y)
ll_nulo = np.sum(y * np.log(p_nulo) + (1.0 - y) * np.log(1.0 - p_nulo))
mcfadden_r2 = 1.0 - (ll_modelo / ll_nulo)

lrt_stat = 2.0 * (ll_modelo - ll_nulo)
lrt_df = X.shape[1] - 1
lrt_p_value = 1.0 - stats.chi2.cdf(lrt_stat, lrt_df)

# 2. Curva ROC e AUC (AUC = U / (n1 * n0))
prob_preditas = mu_final
pos_scores = prob_preditas[y == 1]
neg_scores = prob_preditas[y == 0]
u_roc, _ = stats.mannwhitneyu(pos_scores, neg_scores, alternative='greater')
auc_roc = u_roc / (len(pos_scores) * len(neg_scores))

thresholds = np.linspace(0, 1, 200)
tpr_list, fpr_list = [], []
for t in thresholds:
    pred_t = (prob_preditas >= t).astype(int)
    tp = np.sum((pred_t == 1) & (y == 1))
    fp = np.sum((pred_t == 1) & (y == 0))
    fn = np.sum((pred_t == 0) & (y == 1))
    tn = np.sum((pred_t == 0) & (y == 0))
    tpr_list.append(tp / (tp + fn))
    fpr_list.append(fp / (fp + tn))

fig = go.Figure()
fig.add_scatter(x=fpr_list, y=tpr_list, mode='lines', name=f'Modelo logístico (AUC = {fmt(auc_roc, 3)})',
                line=dict(color=COR['principal'], width=3))
fig.add_scatter(x=[0, 1], y=[0, 1], mode='lines', name='Classificador aleatório', line=dict(color=COR['destaque'], dash='dash'))
fig.update_layout(height=470, width=620, title='Curva ROC do modelo original',
                  xaxis_title='Taxa de falsos positivos (1 − especificidade)', yaxis_title='Sensibilidade')
fig.show()

# 3. Matriz de Confusão (Limiar = Proporção Amostral)
limiar_otimo = y.mean()
pred_final = (prob_preditas >= limiar_otimo).astype(int)
conf_mat = pd.crosstab(pd.Series(y, name='Real').map({1: 'Apostador', 0: 'Não Apostador'}),
                       pd.Series(pred_final, name='Previsto').map({1: 'Apostador', 0: 'Não Apostador'}))
tp = np.sum((pred_final == 1) & (y == 1))
tn = np.sum((pred_final == 0) & (y == 0))
fp = np.sum((pred_final == 1) & (y == 0))
fn = np.sum((pred_final == 0) & (y == 1))
sensibilidade = tp / (tp + fn)
especificidade = tn / (tn + fp)
acuracia = (tp + tn) / len(y)

print(f"ln L modelo = {ll_modelo:.2f} | ln L nulo = {ll_nulo:.2f}")
print(f"Pseudo-R² de McFadden = {mcfadden_r2:.4f}")
print(f"Razão de verossimilhança: G² = {lrt_stat:.2f} ({lrt_df} g.l.), p = {stats.chi2.sf(lrt_stat, lrt_df):.2e}")
print(f"AUC = {auc_roc:.4f}")
print(f"\nMatriz de confusão (limiar = {limiar_otimo:.3f}):")
print(conf_mat)
print(f"\nSensibilidade = {sensibilidade:.2%} | Especificidade = {especificidade:.2%} | Acurácia = {acuracia:.2%}")
''')
md(r"""
#### 7.7.4. Verificação: o modelo entre internautas e com o plano amostral
Ajustamos a **mesma especificação** (mesmas 13 variáveis e mesmas codificações) de duas formas:
1. **Internautas, sem peso:** remove a associação mecânica criada pelos não internautas, que são 0 em Pix, vídeos e jogos e 0 em aposta.
2. **Internautas, ponderado:** coeficientes estimados com `PESO` e erros-padrão pelas 200 réplicas. Cada réplica reajusta o modelo.
""")
code(r'''
# A função irls() reproduz a análise original (verificação de fidelidade)
beta_chk, _, _ = irls(montar_matriz(df).values, df['C14_AGREG'].to_numpy(float))
assert np.allclose(beta_chk, tabela_regressao['Coeficiente (β)'].values, atol=1e-6)

base_int = df[df['internauta'] == 1]
X_int = montar_matriz(base_int).values
y_int = base_int['C14_AGREG'].to_numpy(float)

# 1. Internautas, sem peso
beta_int, cov_int, mu_int = irls(X_int, y_int)
tabela_internautas = tabela_logit(beta_int, np.sqrt(np.diag(cov_int)), nomes_preditores)

# 2. Internautas, ponderado, EP por réplicas
beta_pond, _, mu_pond = irls(X_int, y_int, W_PESO[INTERNAUTA])
reps_beta = np.array([irls(X_int, y_int, W_REP[INTERNAUTA, r])[0] for r in range(W_REP.shape[1])])
ep_rep = reps_beta.std(axis=0, ddof=1)
tabela_ponderada = tabela_logit(beta_pond, ep_rep, nomes_preditores)

# Ajuste e discriminação do modelo entre internautas (mesmas métricas da análise original)
ll_i = np.sum(y_int * np.log(mu_int) + (1 - y_int) * np.log(1 - mu_int))
ll0_i = np.sum(y_int * np.log(y_int.mean()) + (1 - y_int) * np.log(1 - y_int.mean()))
mcfadden_int = 1 - ll_i / ll0_i
auc_int = stats.mannwhitneyu(mu_int[y_int == 1], mu_int[y_int == 0]).statistic / ((y_int == 1).sum() * (y_int == 0).sum())

comparacao_modelos = pd.DataFrame({
    'aOR original': tabela_regressao['aOR (Odds Ratio)'],
    'aOR internautas': tabela_internautas['aOR (Odds Ratio)'],
    'aOR internautas pond.': tabela_ponderada['aOR (Odds Ratio)'],
    'IC 95% pond. (réplicas)': [f"[{fmt(i)}; {fmt(s)}]" for i, s in zip(tabela_ponderada['IC 95% Inf'], tabela_ponderada['IC 95% Sup'])],
    'p pond.': tabela_ponderada['p-valor'],
    'EP réplicas / EP modelo': ep_rep / np.sqrt(np.diag(cov_int)),
}).drop(index='intercepto')
comparacao_modelos.index = [nomes_amigaveis[i] for i in comparacao_modelos.index]
print(f"Internautas: n = {len(y_int):,} | McFadden = {mcfadden_int:.4f} | AUC = {auc_int:.4f}")
comparacao_modelos.round(3)
''')
code(r'''
cm = comparacao_modelos
pix = cm.loc['Usa Pix (vs não)']
razao_ep = cm['EP réplicas / EP modelo']
classes_sig = [c for c in ['Classe C (vs DE)', 'Classe B (vs DE)', 'Classe A (vs DE)'] if cm.loc[c, 'p pond.'] < ALPHA]


def juntar(itens):
    """['C', 'B', 'A'] -> 'C, B e A'."""
    return itens[0] if len(itens) == 1 else ", ".join(itens[:-1]) + " e " + itens[-1]


nomes_classes_sig = juntar([c.split(' (')[0].replace('Classe ', '') for c in classes_sig]) if classes_sig else ''
mostrar(f"""
* **Resultado (análise original):** McFadden = {fmt(mcfadden_r2, 3)}, AUC = {fmt(auc_roc, 3)}, G² = {fmt(lrt_stat, 0)} ({lrt_df} g.l.), {fmt_p(lrt_p_value)}.
  Rejeitamos $H_0$ global. O maior aOR é o do Pix ({fmt(pix['aOR original'])}), seguido de games online ({fmt(cm.loc['Joga games online (vs não)', 'aOR original'])}).
* **Resultado (verificação):** entre internautas, o aOR do Pix cai para {fmt(pix['aOR internautas'])} e, com pesos, fica em
  {fmt(pix['aOR internautas pond.'])} {pix['IC 95% pond. (réplicas)']}. Os erros-padrão pelas réplicas são de {fmt(razao_ep.min(), 1)} a
  {fmt(razao_ep.max(), 1)} vezes os do modelo sem peso. No modelo ponderado,
  {('as classes ' + nomes_classes_sig + ' têm aOR significativamente diferente de 1 em relação à DE') if classes_sig else 'nenhuma classe difere significativamente da DE'}.
* **Interpretação:**
  * O McFadden de {fmt(mcfadden_r2, 2)} e a AUC de {fmt(auc_roc, 2)} indicam que o modelo **separa bem** apostadores de não apostadores. Na análise
    original, parte dessa separação é trivial, porque os não internautas são "previstos" facilmente. Entre internautas, a AUC é {fmt(auc_int, 2)}.
    Um bom ajuste não diz que os fatores **causam** a aposta.
  * O Pix segue como a associação mais forte, mas o valor original de ~9 é inflado pelo universo. O efeito realista fica perto de {fmt(pix['aOR internautas pond.'], 0)}.
    Como o estudo é transversal, não se sabe se o Pix facilita a aposta ou se quem já aposta passa a usar o Pix.
  * A conclusão original de que a classe **não** importa depois dos controles não se sustenta com os pesos amostrais: {('as classes ' + nomes_classes_sig + ' aparecem com chance **' + ('menor' if all(cm.loc[c, "aOR internautas pond."] < 1 for c in classes_sig) else 'diferente') + '** que a DE, mantidos os demais fatores') if classes_sig else 'os ICs, mais largos, continuam incluindo 1'}.
""")
''')
md(r"""
### 7.8. Quadro de decisões
""")
code(r'''
pix_o = tabela_regressao.loc['usa_pix']
quadro_decisoes = pd.DataFrame([
    {'Hipótese': 'H1: gênero × modalidade', 'Estatística (original)': f"χ² = {fmt(chi2_val, 1)} ({dof} g.l.)",
     'Efeito (original)': f"V = {fmt(cramer_v, 3)}", 'p (original)': fmt_p(p_val),
     'Verificação': f"Wald por desenho {fmt_p(p_h1_des)}", 'Decisão': 'Rejeita H0 (robusto)'},
    {'Hipótese': 'H2: influenciadores × cassino', 'Estatística (original)': f"χ² = {fmt(chi2_infl, 1)} (1 g.l.)",
     'Efeito (original)': f"OR = {fmt(odds_ratio)}", 'p (original)': fmt_p(p_infl),
     'Verificação': f"OR internautas pond. = {fmt(h2_tabela.iloc[2, 0])}", 'Decisão': 'Rejeita H0 (efeito menor)'},
    {'Hipótese': 'H3: classe × aposta', 'Estatística (original)': f"χ² = {fmt(chi2_cl, 1)} ({dof_cl} g.l.)",
     'Efeito (original)': f"z(DE) = {fmt(df_residuos.loc['Classe DE', 'Apostador'], 1)}", 'p (original)': fmt_p(p_cl),
     'Verificação': f"Wald internautas {fmt_p(p_h3_int)}",
     'Decisão': 'Rejeita H0' if h3_int_rejeita else 'Rejeita H0 na população; não entre internautas (desenho)'},
    {'Hipótese': 'H4: idade apostador × não', 'Estatística (original)': f"U, Z = {fmt(z_mann, 1)}",
     'Efeito (original)': f"r = {fmt(effect_size_r, 3)}", 'p (original)': fmt_p(p_mann),
     'Verificação': f"r internautas = {fmt(r_i, 3)}", 'Decisão': 'Rejeita H0 (efeito menor)'},
    {'Hipótese': 'H5: idade entre modalidades', 'Estatística (original)': f"H = {fmt(h_stat, 1)} (3 g.l.)",
     'Efeito (original)': 'Loteria mais velha', 'p (original)': fmt_p(p_kw),
     'Verificação': f"Wald por desenho {fmt_p(p_h5_des)}", 'Decisão': 'Rejeita H0'},
    {'Hipótese': 'H6: modelo (Pix)', 'Estatística (original)': f"G² = {fmt(lrt_stat, 0)} ({lrt_df} g.l.)",
     'Efeito (original)': f"aOR Pix = {fmt(pix_o['aOR (Odds Ratio)'])}", 'p (original)': fmt_p(lrt_p_value),
     'Verificação': f"aOR Pix internautas pond. = {fmt(tabela_ponderada.loc['usa_pix', 'aOR (Odds Ratio)'])}", 'Decisão': 'Rejeita H0 (efeito menor)'},
])
quadro_decisoes
''')

# =============================================================================
# 8. VISUALIZAÇÃO
# =============================================================================
md(r"""
## 8. Visualização e comunicação dos resultados

**O que vamos fazer?** Reunir os achados principais em gráficos que possam ir para uma apresentação.

### 8.1. Painel de síntese (análise original, com valores calculados)
No estudo original, o quarto painel tinha os aOR digitados à mão. Aqui eles vêm de `tabela_regressao`.
""")
code(r'''
mods_labels = ['Cassino', 'Rifas', 'Bets', 'Loteria']
mods_mils = [(df[c] * df['PESO']).sum() / 1e6 for c in ['aposta_cassino', 'aposta_rifa', 'aposta_esportiva', 'aposta_loteria']]
labels_idades = ['Cassino', 'Bets', 'Rifas', 'Loteria', 'Não apostador']
mods_idades = [idades_cassino.median(), idades_bets.median(), idades_rifa.median(), idades_loteria.median(), idade_nao_apostador.median()]
taxas_bets = [df[df['sexo_desc'] == s]['aposta_esportiva'].mean() * 100 for s in ['Masculino', 'Feminino']]
taxas_cassino = [df[df['sexo_desc'] == s]['aposta_cassino'].mean() * 100 for s in ['Masculino', 'Feminino']]
top = tabela_regressao.drop(index='intercepto').nlargest(4, 'aOR (Odds Ratio)')

fig = make_subplots(rows=2, cols=2, vertical_spacing=0.16, horizontal_spacing=0.14, subplot_titles=(
    'Milhões de pessoas por modalidade (ponderado)', 'Idade mediana (anos)', 'Participação por gênero (%)', 'Maiores aOR do modelo original'))
fig.add_bar(y=mods_labels[::-1], x=mods_mils[::-1], orientation='h', marker_color=[COR_MOD[m] for m in mods_labels[::-1]],
            text=[fmt(v, 1) + ' mi' for v in mods_mils[::-1]], textposition='outside', showlegend=False, row=1, col=1)
fig.add_bar(x=labels_idades, y=mods_idades, marker_color=[COR_MOD['Cassino'], COR_MOD['Bets'], COR_MOD['Rifas'], COR_MOD['Loteria'], COR['neutro']],
            text=[f"{v:.0f}" for v in mods_idades], textposition='outside', showlegend=False, row=1, col=2)
fig.add_bar(x=['Masculino', 'Feminino'], y=taxas_bets, name='Bets', marker_color=COR_MOD['Bets'],
            text=rotulos_pct(taxas_bets), textposition='outside', row=2, col=1)
fig.add_bar(x=['Masculino', 'Feminino'], y=taxas_cassino, name='Cassino', marker_color=COR_MOD['Cassino'],
            text=rotulos_pct(taxas_cassino), textposition='outside', row=2, col=1)
fig.add_bar(y=[nomes_amigaveis[i] for i in top.index[::-1]], x=top['aOR (Odds Ratio)'][::-1], orientation='h',
            marker_color=COR['quaternaria'], text=[fmt(v) + '×' for v in top['aOR (Odds Ratio)'][::-1]], textposition='outside',
            showlegend=False, row=2, col=2)
fig.add_vline(x=1, line_dash='dash', line_color=COR['destaque'], row=2, col=2)
fig.update_xaxes(range=[0, max(mods_mils) * 1.3], row=1, col=1)
fig.update_yaxes(range=[0, max(mods_idades) * 1.2], row=1, col=2)
fig.update_yaxes(range=[0, max(taxas_bets + taxas_cassino) * 1.25], row=2, col=1)
fig.update_xaxes(range=[0, top['aOR (Odds Ratio)'].max() * 1.3], row=2, col=2)
fig.update_layout(height=820, barmode='group', title='O perfil do apostador digital em quatro painéis',
                  legend=dict(y=0.42, x=0.0, yanchor='bottom'))
fig.show()
''')
md(r"""
### 8.2. Quanto as razões de chances mudam com o universo e o plano amostral
""")
code(r'''
modelos = {'Original (população 10+)': (tabela_regressao, COR['neutro']),
           'Internautas': (tabela_internautas, COR['secundaria']),
           'Internautas, ponderado (réplicas)': (tabela_ponderada, COR['principal'])}
ordem = tabela_ponderada.drop(index='intercepto').sort_values('aOR (Odds Ratio)').index
posicao = {v: i for i, v in enumerate(ordem)}
fig = go.Figure()
for k, (nome, (tab, cor)) in enumerate(modelos.items()):
    t = tab.loc[ordem]
    fig.add_scatter(x=t['aOR (Odds Ratio)'], y=[posicao[v] + (k - 1) * 0.25 for v in ordem], mode='markers', name=nome,
                    marker=dict(color=cor, size=9),
                    error_x=dict(type='data', symmetric=False, array=t['IC 95% Sup'] - t['aOR (Odds Ratio)'],
                                 arrayminus=t['aOR (Odds Ratio)'] - t['IC 95% Inf'], color=cor, thickness=1.5),
                    hovertemplate='aOR = %{x:.2f}<extra>' + nome + '</extra>')
fig.add_vline(x=1, line_dash='dash', line_color=COR['destaque'])
fig.update_layout(height=640, title='Razões de chances ajustadas nos três ajustes do mesmo modelo',
                  xaxis_title='aOR (escala log)', xaxis_type='log',
                  yaxis=dict(tickvals=list(range(len(ordem))), ticktext=[nomes_amigaveis[v] for v in ordem], showgrid=False))
fig.show()
''')
code(r'''
mostrar(f"""
* **Como ler:** cada linha é um preditor. Os três pontos são o mesmo modelo em três ajustes. Quando o ponto azul (ponderado) cruza a
  linha tracejada, a associação deixa de ser significativa com o plano amostral.
* **O que muda:** o Pix se aproxima de {fmt(tabela_ponderada.loc['usa_pix', 'aOR (Odds Ratio)'], 1)} e os intervalos ponderados ficam mais largos
  em todos os preditores. A ordem dos efeitos digitais (Pix, games, vídeos de esporte) se mantém.
""")
''')

# =============================================================================
# 9. DISCUSSÃO
# =============================================================================
md(r"""
## 9. Discussão
""")
code(r'''
mostrar(f"""
### 9.1. O que os dados sustentam
* **Magnitude.** Cerca de {fmt_pct(t_d['Ponderada'])} da população de 10 anos ou mais apostou online em 3 meses, IC de 95% por desenho
  [{fmt_pct(t_d['IC 95% inf'])}; {fmt_pct(t_d['IC 95% sup'])}]. O cassino online é a modalidade mais comum.
* **Gênero muda a modalidade.** Homens se concentram nas bets esportivas e mulheres no cassino online (V = {fmt(cramer_v, 2)}). É o resultado mais
  robusto: vale com e sem pesos e não depende da questão do universo.
* **Juventude.** Apostadores são mais jovens, mas a distância real entre internautas é menor do que a análise original sugere.
* **Ecossistema digital.** Pix, jogos online e vídeos de esporte seguem associados à aposta em todas as versões do modelo. Isso é
  compatível com a ideia de que pagamentos instantâneos reduzem o atrito entre o impulso e a aposta (Thaler, 1999), mas o desenho
  transversal não permite separar causa de seleção.

### 9.2. O que precisou ser corrigido
| Ponto do estudo original | Problema | Efeito da correção |
|---|---|---|
| Código 99 = "não" em Pix, vídeos e jogos | Não internautas entram como 0 no preditor e no desfecho | aOR do Pix: {fmt(pix_o['aOR (Odds Ratio)'])} → {fmt(tabela_internautas.loc['usa_pix', 'aOR (Odds Ratio)'])} (internautas) |
| IC da prevalência por amostra aleatória simples | Ignora pesos e conglomerados (deff ≈ {fmt(t_d['deff'], 1)}) | IC por réplicas, centrado na estimativa ponderada |
| Testes sem peso | Descrevem a amostra, não o Brasil | Camada por desenho em todas as hipóteses |
| Classe × aposta na população toda | Mistura acesso à Internet com propensão a apostar | Entre internautas, Wald por desenho {fmt_p(p_h3_int)} |
| Kruskal-Wallis com grupos sobrepostos | {fmt(em_varios, 0)} pessoas em mais de um grupo | Conclusão confirmada pelo Wald por desenho |
| "Classe não importa no modelo" | Válido só no modelo sem peso | Com pesos, {('as classes ' + nomes_classes_sig + ' diferem da DE') if classes_sig else 'a conclusão se mantém'} |
| "Ajuste excelente" (McFadden > 0,20) | Parte da separação vem dos não internautas | Entre internautas: AUC = {fmt(auc_int, 2)}, McFadden = {fmt(mcfadden_int, 2)} |
| Números digitados nos textos | Divergiam das saídas (ex.: OR 2,80 no texto e {fmt(odds_ratio)} na execução) | Todos os números agora são gerados pelas células |

### 9.3. Implicações
* **Impacto regressivo.** Qualquer que seja a propensão por classe, o mesmo valor apostado pesa mais no orçamento das famílias de baixa
  renda. O questionário, porém, não mede valores, então esse ponto é uma hipótese para estudos futuros, não um resultado.
* **Publicidade e influenciadores.** A associação entre vídeos de influenciadores e cassino online persiste entre internautas e com pesos.
  Isso dá base empírica ao debate sobre a regulação da publicidade de apostas, embora não prove que a exposição cause a aposta.

### 9.4. Limitações
1. **Transversal:** só associações, sem direção causal.
2. **Autodeclaração:** apostas podem ser subnotificadas por estigma.
3. **Ocorrência, não intensidade:** o C14 não mede frequência nem valor apostado.
4. **Testes clássicos numa amostra complexa:** mitigado pela camada por desenho, mas a camada usa estatísticas análogas (médias e
   proporções ponderadas), não versões por desenho dos mesmos testes de postos.
5. **Hipóteses formuladas após a exploração da mesma base:** os p-valores devem ser lidos como evidência descritiva forte, não como confirmação.
""")
''')

# =============================================================================
# 10. CONCLUSÃO
# =============================================================================
md(r"""
## 10. Conclusão
""")
code(r'''
mostrar(f"""
**P1. A modalidade preferida depende do gênero?** **Sim** (χ² = {fmt(chi2_val, 1)}, V = {fmt(cramer_v, 2)}, {fmt_p(p_val)}; Wald por desenho {fmt_p(p_h1_des)}).

**P2. Influenciadores estão associados ao cassino online?** **Sim**, com OR = {fmt(h2_tabela.iloc[2, 0])} entre internautas, com pesos
(IC [{fmt(h2_tabela.iloc[2, 1])}; {fmt(h2_tabela.iloc[2, 2])}]). É menor que o OR = {fmt(odds_ratio)} da análise original.

**P3. A proporção de apostadores é igual entre classes?** **Não, na população de 10 anos ou mais** ({fmt_p(p_cl)}; Wald por desenho {fmt_p(p_h3_pop)}).
{'Entre internautas, a diferença persiste' if h3_int_rejeita else 'Entre internautas, com o plano amostral, a diferença deixa de ser significativa'} ({fmt_p(p_h3_int)}):
o gradiente de classe vem sobretudo do acesso à Internet.

**P4. Apostadores são mais jovens?** **Sim** (r = {fmt(effect_size_r, 2)} na população e {fmt(r_i, 2)} entre internautas).

**P5. A idade difere entre modalidades?** **Sim** (H = {fmt(h_stat, 1)}, {fmt_p(p_kw)}). A loteria tem o público mais velho.

**P6. Quais fatores seguem associados no modelo?** Uso de Pix, jogos online, vídeos de esporte, gênero masculino e vídeos de influenciadores
em todas as versões. O aOR do Pix fica em {fmt(tabela_ponderada.loc['usa_pix', 'aOR (Odds Ratio)'])} no modelo ponderado entre internautas.

**Contribuição desta reorganização.** As conclusões qualitativas do estudo original se mantêm em gênero, idade e hábitos digitais,
com duas ressalvas. As magnitudes das associações digitais são menores do que as originais. E o papel da classe econômica muda:
o gradiente bivariado vem sobretudo do acesso à Internet{(' e, no modelo ponderado, as classes ' + nomes_classes_sig + ' aparecem com chance diferente da DE') if classes_sig else ''}.
""")
''')

# =============================================================================
# 11. REFERÊNCIAS E REPRODUTIBILIDADE
# =============================================================================
md(r"""
## 11. Referências e reprodutibilidade

### 11.1. Dados e estudo de origem
* Cetic.br/NIC.br. *Pesquisa sobre o uso das tecnologias de informação e comunicação nos domicílios brasileiros: TIC Domicílios 2025*. Base de microdados de indivíduos v1.0 e dicionário de variáveis v1.0. São Paulo: Comitê Gestor da Internet no Brasil. Disponível em: https://cetic.br/pt/arquivos/domicilios/2025/individuos/
* Estudo de origem: `projeto_apostas_digitais.ipynb`, branch `lucas-dataset-bets` do repositório TIC-Kids-Online.
* ABEP. *Critério de Classificação Econômica Brasil* (2015).

### 11.2. Métodos estatísticos
* Agresti, A. (2013). *Categorical Data Analysis* (3ª ed.). Wiley.
* Cramér, H. (1946). *Mathematical Methods of Statistics*. Princeton University Press.
* Haberman, S. J. (1973). The analysis of residuals in cross-classified tables. *Biometrics*, 29(1), 205–220.
* Mann, H. B., & Whitney, D. R. (1947). On a test of whether one of two random variables is stochastically larger than the other. *Annals of Mathematical Statistics*, 18(1), 50–60.
* Kruskal, W. H., & Wallis, W. A. (1952). Use of ranks in one-criterion variance analysis. *Journal of the American Statistical Association*, 47(260), 583–621.
* McFadden, D. (1974). Conditional logit analysis of qualitative choice behavior. In P. Zarembka (Ed.), *Frontiers in Econometrics*. Academic Press.
* Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic Regression* (3ª ed.). Wiley.

### 11.3. Amostras complexas
* Rao, J. N. K., & Wu, C. F. J. (1988). Resampling inference with complex survey data. *Journal of the American Statistical Association*, 83(401), 231–241.
* Lumley, T. (2010). *Complex Surveys: A Guide to Analysis Using R*. Wiley.
* Kish, L. (1965). *Survey Sampling*. Wiley.

### 11.4. Tema
* Kahneman, D., & Tversky, A. (1979). Prospect theory: an analysis of decision under risk. *Econometrica*, 47(2), 263–291.
* Thaler, R. H. (1999). Mental accounting matters. *Journal of Behavioral Decision Making*, 12(3), 183–206.

### 11.5. Ambiente de execução
""")
code(r'''
pd.Series({"Python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
           "scipy": scipy.__version__, "plotly": plotly.__version__, "Arquivo de dados": ARQ_DADOS.name,
           "Registros lidos": len(df_raw), "Pesos replicados": W_REP.shape[1]}).to_frame("valor")
''')
md(r"""
### 11.6. Como executar
1. Instale as dependências, a partir da pasta `PROJ 2`:
   ```bash
   pip install -r requirements.txt
   ```
2. Execute **todas as células em ordem** (*Run All*). Na linha de comando:
   ```bash
   jupyter nbconvert --to notebook --execute --inplace notebooks/04_tic_domicilios_2025_apostas.ipynb
   ```
3. Se `data/tic_domicilios_2025/` não tiver a base, a Seção 0 baixa os microdados e o dicionário oficiais do site do Cetic.br.
4. Este notebook é gerado por `scripts/build_notebook_apostas.py`. Edite o script e regenere o notebook para manter os dois sincronizados.
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"Notebook escrito em {OUT} ({len(cells)} células)")
