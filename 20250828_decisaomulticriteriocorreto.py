# app_mcda_cd_tese.py
# --------------------------------------------------
# Requisitos:
#   pip install streamlit pandas numpy plotly
# Execução:
#   streamlit run app_mcda_cd_tese.py
# --------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Localização de CD — MCDA (tese + app)", layout="wide")

# =========================================================
# Utilidades visuais e funções de apoio
# =========================================================
PLOT_BG = "#F3F4F6"   # cinza claro (board-friendly)
FONT = {"color": "#111827", "size": 16}

def fig_barras(scores: pd.Series, titulo: str):
    """
    Barra horizontal com paleta viridis e eixos legíveis.
    - Eixos X e Y com linha, ticks e grade.
    - Margens maiores para os rótulos fora das barras.
    - Intervalo de X com 'padding' automático para evitar corte.
    """
    s = scores.sort_values(ascending=True)

    # Intervalo do eixo X com folga (padding) automática
    x0 = float(s.min())
    x1 = float(s.max())
    if x0 == x1:
        x0, x1 = 0.0, 1.0
    pad = 0.05 * max(1e-6, (x1 - x0))
    xr = (x0 - pad, x1 + pad)

    fig = px.bar(
        s,
        x=s.values, y=s.index, orientation="h",
        text=[f"{v:.3f}" for v in s.values],
        color=s.values, color_continuous_scale="viridis",
    )

    fig.update_traces(textposition="outside", cliponaxis=False)

    # Layout: fundo, fonte e margens
    fig.update_layout(
        title=titulo,
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG, font=FONT,
        margin=dict(l=100, r=30, t=70, b=70),
        xaxis_title="Score (maior = melhor)", yaxis_title="Cidades",
        coloraxis_showscale=False,
    )

    AX_COLOR = "#111827"  # quase preto (alto contraste)
    GRID = "#CBD5E1"      # cinza para grid

    # Eixo X visível
    fig.update_xaxes(
        range=xr,
        showline=True, linewidth=1.4, linecolor=AX_COLOR,
        ticks="outside", tickcolor=AX_COLOR, ticklen=6,
        tickfont=dict(size=14, color=AX_COLOR),
        title_font=dict(size=16, color=AX_COLOR),
        gridcolor=GRID, zeroline=False, mirror=True,
    )

    # Eixo Y visível
    fig.update_yaxes(
        showline=True, linewidth=1.4, linecolor=AX_COLOR,
        tickfont=dict(size=14, color=AX_COLOR),
        title_font=dict(size=16, color=AX_COLOR),
        ticks="outside", tickcolor=AX_COLOR, ticklen=6,
        gridcolor=GRID, zeroline=False, mirror=True,
        automargin=True,
    )
    return fig

def minmax_series(s: pd.Series, benefit_flag: bool):
    """Normalização min–max → [0,1]. Em custo, inverte (1 = melhor)."""
    s = s.astype(float)
    if s.max() == s.min():
        return pd.Series(1.0, index=s.index)
    v = (s - s.min()) / (s.max() - s.min())
    return v if benefit_flag else (1 - v)

def ahp_weights_from_pcm(pcm: np.ndarray):
    """Pesos AHP (autovetor principal) + Consistency Ratio (CR)."""
    eigvals, eigvecs = np.linalg.eig(pcm)
    k = np.argmax(eigvals.real)
    w = np.abs(eigvecs[:, k].real)
    w = w / w.sum()
    n = pcm.shape[0]
    lam = eigvals[k].real
    CI = (lam - n) / (n - 1) if n > 2 else 0.0
    RI_table = {1:0.0, 2:0.0, 3:0.58, 4:0.90, 5:1.12, 6:1.24, 7:1.32, 8:1.41, 9:1.45}
    RI = RI_table.get(n, 1.49)
    CR = CI / RI if RI > 0 else 0.0
    return w.real, float(CR)

def gaussian_utility(s: pd.Series, mu: float, sigma: float, benefit_flag: bool):
    """Utilidade gaussiana (pico em μ). Para custo, reflete antes."""
    x = s.astype(float)
    sigma = max(1e-9, sigma)
    if not benefit_flag:
        x = (x.max() + x.min()) - x
    u = np.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2))
    return pd.Series(u, index=s.index)

def topsis(scores_01: pd.DataFrame, weights: np.ndarray, is_benefit: dict):
    """
    TOPSIS sobre matriz já em [0,1] (maior=melhor).
    Ideal = 1 para benefício e 0 para custo; anti-ideal o oposto.
    """
    V = scores_01.values.astype(float)
    w = weights / weights.sum()
    cols = scores_01.columns
    ideal = np.array([1.0 if is_benefit[c] else 0.0 for c in cols])
    anti  = np.array([0.0 if is_benefit[c] else 1.0 for c in cols])
    Dp = np.sqrt(((V - ideal) ** 2 * w).sum(axis=1))
    Dm = np.sqrt(((V - anti ) ** 2 * w).sum(axis=1))
    C  = Dm / (Dp + Dm + 1e-12)  # closeness
    return pd.Series(C, index=scores_01.index, name="TOPSIS")

# =========================================================
# ABAS
# =========================================================
aba1, aba2 = st.tabs(["📚 Tese (introdução + formulação MCDA)", "🏗️ Dados reais — MCDA (1 CD)"])

# =========================================================
# ABA 1 — Tese de mestrado (linguagem clara + LaTeX)
# =========================================================
with aba1:
    st.title("📚 Escolha de uma cidade para o CD via MCDA — Introdução e Formulação")

    st.header("1. Contexto do problema")
    st.markdown("""
Temos até **5 cidades candidatas** (ex.: Vitória, Vila Velha, Serra, Cariacica, Guarapari) e precisamos
escolher **apenas uma** para abrir um **Centro de Distribuição (CD)**.  
A decisão envolve **múltiplos critérios** que atuam em direções diferentes:

- **Benefício (quanto maior, melhor):** Demanda, PIB, População.  
- **Custo/Risco (quanto menor, melhor):** Aluguel/m², Roubo de cargas (índice), Diesel, **Distância média** (km) às demais praças.

Usar **apenas um** critério levaria a decisões **parciais**.  
Por isso adotamos **MCDA — Multi-Criteria Decision Analysis**, que combina critérios em uma **régua comum** e produz um **ranking** de cidades.
""")

    st.header("2. Conjuntos, dados e notação")
    st.markdown("Considere:")
    st.latex(r"""
\begin{aligned}
&\mathcal{A}=\{a_1,\dots,a_m\} &&\text{: conjunto de alternativas (cidades).}\\
&\mathcal{C}=\{c_1,\dots,c_n\} &&\text{: conjunto de critérios.}\\
&x_{ij} &&\text{: desempenho bruto da cidade } a_j \text{ no critério } c_i.\\
&b_i\in\{\text{benefício},\text{custo}\} &&\text{: tipo de critério.}\\
&w_i\ge 0,\ \sum_{i=1}^n w_i=1 &&\text{: peso do critério } c_i.\\
&r_{ij}\in[0,1] &&\text{: desempenho normalizado (maior = melhor).}
\end{aligned}
""")

    # ---------- NOVO: Glossário dos termos de modelagem ----------
    st.subheader("Glossário rápido dos termos")
    st.markdown("""
- **Alternativa (cidade)**: opção de localização do CD.  
- **Critério**: atributo avaliado (ex.: aluguel, demanda).  
- **Benefício**: quanto **maior**, **melhor** (ex.: demanda).  
- **Custo**: quanto **menor**, **melhor** (ex.: aluguel, distância).  
- **Normalização**: coloca tudo na mesma régua [0,1].  
- **Peso (wᵢ)**: importância relativa de cada critério (soma = 1).  
- **Ideal/anti-ideal**: melhores/piores referências por critério em um método.  
- **Fluxo (PROMETHEE)**: medida de quanto uma cidade “vence” as outras.  
- **Utilidade**: transformação de valor bruto em satisfação (0 a 1).  
- **CR (AHP)**: Consistency Ratio; ≤ 0,10 indica julgamentos consistentes.
""")

    st.header("3. Normalização e pesos")
    st.markdown("**Normalização min–max** em [0,1] (régua comum):")
    st.latex(r"""
r_{ij}=
\begin{cases}
\dfrac{x_{ij}-\min_j x_{ij}}{\max_j x_{ij}-\min_j x_{ij}}, & \text{se } c_i \text{ é benefício}\\[10pt]
1-\dfrac{x_{ij}-\min_j x_{ij}}{\max_j x_{ij}-\min_j x_{ij}}, & \text{se } c_i \text{ é custo}
\end{cases}
""")
    st.markdown("**Pesos** (importância relativa):")
    st.latex(r"\quad w_i\ge 0,\qquad \sum_{i=1}^{n} w_i=1.")

    # ---------- NOVO: exemplo numérico de normalização ----------
    with st.expander("Exemplo numérico simples — Normalização", expanded=False):
        st.markdown("""
**Cenário:** 2 cidades (A,B) e 2 critérios — `c1` (benefício) e `c2` (custo).  
Valores brutos:
- c1 (↑): A=80, B=60  → min=60, max=80  
- c2 (↓): A=30, B=10  → min=10, max=30  
""")
        st.latex(r"""
\begin{aligned}
r_{A,c1}&=\frac{80-60}{80-60}=1,\quad r_{B,c1}=\frac{60-60}{20}=0.\\
r_{A,c2}&=1-\frac{30-10}{20}=0,\quad r_{B,c2}=1-\frac{10-10}{20}=1.
\end{aligned}
""")
        st.markdown("**Matriz normalizada (maior=melhor):**  \nA: (1, 0) • B: (0, 1)")

    st.subheader("Hipóteses/pressupostos usuais")
    st.markdown("""
- **Mensurabilidade e não duplicidade** dos critérios.  
- **Monotonicidade** (maior melhor em benefícios; menor melhor em custos).  
- **Independência preferencial** (aprox.) para modelos aditivos (SAW/TOPSIS).  
- **Pesos estáveis** durante a decisão.  
- Em **AHP**, julgamentos com **consistência aceitável** (CR ≤ 0,10).
""")

    st.header("4. Métodos MCDA usados e formulações")
    st.subheader("4.1 SAW — Soma Ponderada (aditivo)")
    st.latex(r"S(a_j)=\sum_{i=1}^{n} w_i\, r_{ij}\quad\Longrightarrow\quad \text{maior } S(a_j) \text{ é melhor.}")

    # ---------- NOVO: exemplo SAW ----------
    with st.expander("Exemplo numérico — SAW", expanded=False):
        st.markdown("""
**Pesos:** `w = (0,6; 0,4)` para (c1, c2).  
**Matriz r:** A=(1,0), B=(0,1).  
""")
        st.latex(r"""
S(A)=0{,}6\cdot 1 + 0{,}4\cdot 0 = 0{,}6\qquad
S(B)=0{,}6\cdot 0 + 0{,}4\cdot 1 = 0{,}4
""")
        st.markdown("**Ranking:** A (0,60) ≻ B (0,40).")

    st.subheader("4.2 TOPSIS — proximidade ao ideal")
    st.markdown("Como a matriz já está em [0,1], definimos **ideal** e **anti-ideal** por critério:")
    st.latex(r"""
r_i^{+}=
\begin{cases}
1,& c_i \text{ benefício}\\
0,& c_i \text{ custo}
\end{cases}
\qquad
r_i^{-}=
\begin{cases}
0,& c_i \text{ benefício}\\
1,& c_i \text{ custo}
\end{cases}
""")
    st.markdown("Distâncias ponderadas e coeficiente de proximidade:")
    st.latex(r"""
D_j^{+}=\sqrt{\sum_{i=1}^{n} w_i\,(r_{ij}-r_i^{+})^2},\qquad
D_j^{-}=\sqrt{\sum_{i=1}^{n} w_i\,(r_{ij}-r_i^{-})^2},\qquad
C_j=\frac{D_j^{-}}{D_j^{+}+D_j^{-}}.
""")

    # ---------- NOVO: exemplo TOPSIS ----------
    with st.expander("Exemplo numérico — TOPSIS (mesmo cenário)", expanded=False):
        st.markdown("""
**Atenção didática:** aqui seguimos **a formulação desta seção** (ideal do **custo** = 0).  
**Pesos:** `w = (0,6; 0,4)`; **r:** A=(1,0), B=(0,1).  
**Ideais:** para c1 (benefício) r⁺=1, r⁻=0; para c2 (custo) r⁺=0, r⁻=1.
""")
        st.latex(r"""
\begin{aligned}
D_A^+&=\sqrt{0{,}6\,(1-1)^2 + 0{,}4\,(0-0)^2}=0, \quad
D_A^-=\sqrt{0{,}6\,(1-0)^2 + 0{,}4\,(0-1)^2}=\sqrt{1{,}0}\approx1.000.\\
C_A&=\frac{D_A^-}{D_A^+ + D_A^-}=\frac{1}{1}=1{,}000.\\[4pt]
D_B^+&=\sqrt{0{,}6\,(0-1)^2 + 0{,}4\,(1-0)^2}=\sqrt{1{,}0}\approx1.000,\quad
D_B^-=\sqrt{0{,}6\,(0-0)^2 + 0{,}4\,(1-1)^2}=0.\\
C_B&=0.
\end{aligned}
""")
        st.markdown("**Ranking:** A (1,000) ≻ B (0,000).")

    st.subheader("4.3 VIKOR — solução de compromisso")
    st.markdown("Com **v**∈[0,1] (peso da estratégia), sobre a matriz **r** em [0,1]:")
    st.latex(r"""
\begin{aligned}
S_j &= \sum_{i=1}^{n} w_i\,(1-r_{ij}),\\
R_j &= \max_{i}\big[w_i\,(1-r_{ij})\big],\\
Q_j &= v\,\frac{S_j - S^{*}}{S^{-}-S^{*}} + (1-v)\,\frac{R_j - R^{*}}{R^{-}-R^{*}}.
\end{aligned}
""")
    st.latex(r"S^{*}=\min_j S_j,\quad S^{-}=\max_j S_j,\quad R^{*}=\min_j R_j,\quad R^{-}=\max_j R_j.")
    st.markdown("**Menor** \(Q_j\) **é melhor.**")

    # ---------- NOVO: exemplo VIKOR ----------
    with st.expander("Exemplo numérico — VIKOR (v=0,5)", expanded=False):
        st.markdown("**w = (0,6;0,4)**; **r:** A=(1,0), B=(0,1); **v=0,5**.")
        st.latex(r"""
\begin{aligned}
S_A&=0{,}6(1-1)+0{,}4(1-0)=0{,}4,\quad &R_A&=\max\{0{,}6(0),\,0{,}4(1)\}=0{,}4.\\
S_B&=0{,}6(1-0)+0{,}4(1-1)=0{,}6,\quad &R_B&=\max\{0{,}6(1),\,0{,}4(0)\}=0{,}6.\\
S^*&=0{,}4,\ S^- =0{,}6,\ R^* =0{,}4,\ R^- =0{,}6.\\
Q_A&=0{,}5\cdot\frac{0{,}4-0{,}4}{0{,}2}+0{,}5\cdot\frac{0{,}4-0{,}4}{0{,}2}=0.\\
Q_B&=0{,}5\cdot\frac{0{,}6-0{,}4}{0{,}2}+0{,}5\cdot\frac{0{,}6-0{,}4}{0{,}2}=1.
\end{aligned}
""")
        st.markdown("**Ranking (menor Q)**: A (0,000) ≺ B (1,000) → **A melhor**.")

    st.subheader("4.4 PROMETHEE II — fluxos de preferência")
    st.markdown("Preferência **usual** por critério (1 se ganha, 0 caso contrário):")
    st.latex(r"P_i(a,b)=\begin{cases}1,& r_{ia}>r_{ib}\\[3pt] 0,& \text{caso contrário}\end{cases}")
    st.markdown("Fluxos de saída/entrada e fluxo líquido (ranking):")
    st.latex(r"""
\phi^{+}(a)=\frac{1}{m-1}\sum_{b\ne a}\sum_{i=1}^{n} w_i\,P_i(a,b),\quad
\phi^{-}(a)=\frac{1}{m-1}\sum_{b\ne a}\sum_{i=1}^{n} w_i\,P_i(b,a),\quad
\phi(a)=\phi^{+}(a)-\phi^{-}(a).
""")

    # ---------- NOVO: exemplo PROMETHEE II ----------
    with st.expander("Exemplo numérico — PROMETHEE II (função usual)", expanded=False):
        st.markdown("""
**w = (0,6;0,4)**; **r:** A=(1,0), B=(0,1).  
Comparando A vs B por critério:  
- c1: A (1) > B (0) → **A vence** (peso 0,6)  
- c2: A (0) < B (1) → **B vence** (peso 0,4)
""")
        st.latex(r"""
\phi^{+}(A)=0{,}6,\ \phi^{-}(A)=0{,}4 \Rightarrow \phi(A)=+0{,}2;\quad
\phi^{+}(B)=0{,}4,\ \phi^{-}(B)=0{,}6 \Rightarrow \phi(B)=-0{,}2.
""")
        st.markdown("**Ranking:** A ( +0,20 ) ≻ B ( −0,20 ).")

    st.subheader("4.5 AHP + utilidade gaussiana (AHP-Gauss)")
    st.markdown("**Pesos por AHP** e **avaliação por utilidade** centrada numa faixa-alvo:")
    st.latex(r"""
\text{AHP: } \mathbf{A}\in\mathbb{R}^{n\times n},\ a_{ij}=\text{importância de }c_i\text{ sobre }c_j,\quad
\mathbf{w}=\text{autovetor principal normalizado.}
""")
    st.latex(r"""
\text{CR}=\frac{\lambda_{\max}-n}{(n-1)\cdot RI}\ \ \ (\text{aceitável se } \text{CR}\le 0{,}10).
""")
    st.latex(r"""
\text{Utilidade gaussiana: } u_i(x)=\exp\!\Big(-\frac{(x-\mu_i)^2}{2\,\sigma_i^2}\Big),
\quad U(a_j)=\sum_{i=1}^{n} w_i\,u_i(x_{ij}).
""")
    
    # ---------- NOVO: exemplo TOPSIS ----------
    with st.expander("Exemplo numérico — AHP-Gauss (2 critérios)", expanded=False):
        st.markdown("**AHP (2 critérios):** suponha c1 é **3×** mais importante que c2.")
        st.markdown("Matriz par-a-par:")
        st.latex(r"A=\begin{bmatrix}1 & 3\\ \tfrac{1}{3} & 1\end{bmatrix}")
        st.markdown("→ pesos aproximados **w ≈ (0,75; 0,25)** e **CR = 0** (n=2).")
        st.markdown("---")
        st.markdown("**Utilidade (alvos):**")
        st.markdown("- **c1 (benefício):** alvo μ₁ = 1 com σ₁ = 0,2")
        st.latex(r"u_1(A)=\exp(0)=1,\quad u_1(B)=\exp\!\left(-\frac{(0-1)^2}{2\cdot 0.2^2}\right)\approx 4\times10^{-6}")
        st.markdown("- **c2 (custo):** alvo μ₂ = 0 com σ₂ = 0,2")
        st.latex(r"u_2(A)=\exp(0)=1,\quad u_2(B)=\exp\!\left(-\frac{(1-0)^2}{2\cdot 0.2^2}\right)\approx 4\times10^{-6}")
        st.markdown("---")
        st.markdown("**Agregação:**")
        st.latex(r"U(A)=0.75\cdot 1 + 0.25\cdot 1 = 1.00")
        st.latex(r"U(B)\approx 0.75\cdot 0.000004 + 0.25\cdot 0.000004 \approx 0.000004")
        st.markdown("**Ranking:** A ≻ B.")



    st.header("5. Como aplicar ao nosso caso (5 cidades → 1 CD)")
    st.markdown("""
1) **Preencher os dados** por cidade (Demanda, PIB, População, Aluguel/m², Roubo, Diesel, Distância média).  
2) **Normalizar** para [0,1]; **ajustar pesos** conforme a estratégia.  
3) Rodar os **5 métodos** e comparar **rankings** e **vencedores**.  
4) Se vários métodos apontarem a **mesma cidade**, a recomendação é **robusta**.
""")

# =========================================================
# ABA 2 — App MCDA (sem Excel, tudo via inputs do board)
# =========================================================
with aba2:
    st.title("🏗️ MCDA — Escolha de 1 CD (até 5 cidades candidatas)")
    st.caption("Preencha os critérios por cidade; o app calcula os 5 rankings.")

    # Quantidade e nomes das cidades
    n_cidades = st.slider("Quantas cidades candidatas?", 1, 5, 5)
    nomes_default = ["Vitória", "Vila Velha", "Serra", "Cariacica", "Guarapari"][:n_cidades]
    nomes = st.text_input("Nomes das cidades (separe por vírgula)", ", ".join(nomes_default))
    cidades = [s.strip() for s in nomes.split(",") if s.strip()]
    if len(cidades) != n_cidades:
        st.warning("O número de nomes deve bater com a quantidade de cidades.")
        st.stop()

    # --- valores fixos pré-preenchidos (da planilha) ---
    # Obs.: use ponto como separador decimal.
    dados_fixos = {
    "Vitória"   : {"PIB": 1866.0, "populacao": 322869.0, "aluguel_m2": 23.0, "roubo_ind": 0.0, "diesel": 6.58, "dist_media": 29.0},
    "Vila Velha": {"PIB": 1211.0, "populacao": 467722.0, "aluguel_m2": 16.0, "roubo_ind": 0.0, "diesel": 6.05, "dist_media": 29.0},
    "Serra"     : {"PIB": 1052.0, "populacao": 520653.0, "aluguel_m2": 30.0, "roubo_ind": 0.0, "diesel": 5.98, "dist_media": 38.0},
    "Cariacica" : {"PIB": 1715.0, "populacao": 353491.0, "aluguel_m2": 17.0, "roubo_ind": 0.0, "diesel": 6.23, "dist_media": 28.0},
    "Guarapari" : {"PIB":  991.0, "populacao": 124656.0, "aluguel_m2": 36.0, "roubo_ind": 0.0, "diesel": 6.29, "dist_media": 56.0},
    }

    # ordem das cidades
    cidades = list(dados_fixos.keys())

    # Coleta dos dados (critérios)
    st.subheader("Preencha os critérios por cidade")
    base = pd.DataFrame(
    {
        "cidade": cidades,
        "demanda": [np.nan]*len(cidades),  # ↑ (campo a ser digitado)
        "PIB": [dados_fixos[c]["PIB"] for c in cidades],  # ↑ Renda mensal por pessoa
        "populacao": [dados_fixos[c]["populacao"] for c in cidades],  # ↑
        "aluguel_m2": [dados_fixos[c]["aluguel_m2"] for c in cidades],  # ↓
        "roubo_ind": [dados_fixos[c]["roubo_ind"] for c in cidades],    # ↓ (0–1)
        "diesel": [dados_fixos[c]["diesel"] for c in cidades],          # ↓
        "dist_media": [dados_fixos[c]["dist_media"] for c in cidades],  # ↓ (km)
        }
    )

    cidades_df = st.data_editor(
    base, num_rows="fixed", use_container_width=True,
    column_config={
        "demanda":    st.column_config.NumberColumn("Demanda (unid/mês)", step=50.0, format="%.0f"),
        "PIB":        st.column_config.NumberColumn("Renda Mensal Pessoa", step=1.0, format="%.2f"),
        "populacao":  st.column_config.NumberColumn("População (hab.)", step=100.0, format="%.0f"),
        "aluguel_m2": st.column_config.NumberColumn("Aluguel (R$/m²/mês)", step=0.1, format="%.2f"),
        "roubo_ind":  st.column_config.NumberColumn("Roubo (0–1)", step=0.01, format="%.2f"),
        "diesel":     st.column_config.NumberColumn("Diesel (R$/L)", step=0.01, format="%.2f"),
        "dist_media": st.column_config.NumberColumn("Distância média (km)", step=1.0, format="%.2f"),
        }
    )
    
    # Validações
    if cidades_df.isna().any().any():
    # Apenas a demanda pode ficar em branco? -> exija preenchimento só dela.
        faltas = cidades_df.columns[cidades_df.isna().any()].tolist()
        st.warning(f"Preencha os campos faltantes: {', '.join(faltas)}.")
        st.stop()
    if (cidades_df["roubo_ind"] < 0).any() or (cidades_df["roubo_ind"] > 1).any():
        st.error("`roubo_ind` deve estar no intervalo [0,1].")
        st.stop()

    # Matriz de desempenho (valores brutos)
    perf = cidades_df.set_index("cidade").copy()
    perf = perf.rename(columns={"demanda": "demanda_cidade"})
    perf = perf[["demanda_cidade","PIB","populacao","aluguel_m2","roubo_ind","diesel","dist_media"]]

    benefit = {
        "demanda_cidade": True,
        "PIB": True,
        "populacao": True,
        "aluguel_m2": False,
        "roubo_ind": False,
        "diesel": False,
        "dist_media": False,
    }

    st.markdown("**Matriz de desempenho (valores brutos):**")
    st.dataframe(
        perf.style.format({
            "demanda_cidade":"{:,.0f}", "PIB":"{:,.2f}", "populacao":"{:,.0f}",
            "aluguel_m2":"{:,.2f}", "roubo_ind":"{:.2f}", "diesel":"{:,.2f}", "dist_media":"{:,.2f}"
        }),
        use_container_width=True
    )

    # Normalização (0–1)
    cols = list(benefit.keys())
    norm = pd.DataFrame(index=perf.index)
    for c in cols:
        norm[c] = minmax_series(perf[c], benefit[c])

    # Pesos (sliders)
    st.subheader("Pesos dos critérios (somam 1)")
    weight_defaults = {"demanda_cidade":0.10, "PIB":0.20, "populacao":0.10,
                       "aluguel_m2":0.20, "roubo_ind":0.15, "diesel":0.15, "dist_media":0.10}
    w = []
    cols_w = st.columns(len(cols))
    for k, c in enumerate(cols):
        w.append(cols_w[k].slider(f"Peso — {c}", 0.0, 1.0, float(weight_defaults.get(c, 0.14)), 0.05))
    w = np.array(w, dtype=float)
    if w.sum() == 0: w = np.ones_like(w)
    w = w / w.sum()

    st.header("Resultados — 5 métodos MCDA")

    # ---------- M1: SAW ----------
    M1 = pd.Series(norm[cols].values.dot(w), index=norm.index, name="SAW")
    st.subheader("M1 — SAW (Soma Ponderada)")
    st.plotly_chart(fig_barras(M1, "Ranking — M1 (SAW)"), use_container_width=True)

    # -----------------------
    # Explicação para o usuário final (sem jargão)
    # -----------------------
    with st.expander("O que é o SAW (Soma Ponderada)?", expanded=True):
        st.markdown("""
**Ideia em 1 frase:** cada alternativa recebe pontos em cada critério; a gente **coloca tudo na mesma escala**, multiplica pelo **peso** de cada critério (o que é mais importante vale mais) e **soma**. Quem somar mais, fica melhor no ranking.

**Por que precisa normalizar?** Porque critérios têm unidades/escala diferentes (ex.: PIB vs. tempo). Normalizar só **coloca tudo de 0 a 1**, para comparar maçã com laranja de forma justa.

**O que fazem os pesos?** Mostram o que importa mais (ex.: “Qualidade” pode pesar 50%, “Custo” 30%, “Prazo” 20%). Se um critério tem peso alto, ele puxa mais a pontuação final.
""")

    with st.expander("Como calculamos (modelagem por trás)", expanded=False):
        st.markdown("""
**Passo 1 — Normalização:** transformamos cada critério para **0 a 1**.  
- **Benefício (quanto maior, melhor):** o melhor vira 1 e o pior vira 0.  
- **Custo (quanto menor, melhor):** o menor vira 1 e o maior vira 0.

**Passo 2 — Pesos:** garantimos que os pesos **somem 1** (ex.: 0,5 + 0,3 + 0,2 = 1).

**Passo 3 — Soma ponderada:** para cada alternativa:  
`pontuação = (critério1_normalizado × peso1) + (critério2_normalizado × peso2) + ...`

> Resultado: um número entre 0 e 1 que permite ordenar (ranking).
""")

    # -----------------------
    # Exemplo numérico simples (auto-explicativo)
    # -----------------------
    st.subheader("Exemplo numérico (didático)")
    st.caption("3 alternativas (A, B, C) × 3 critérios. C1 e C3 = benefício (↑); C2 = custo (↓). Pesos w = [0,5, 0,3, 0,2].")

    # 1) Matriz original (não normalizada)
    dados = pd.DataFrame({
        "C1 (benefício)": [70, 90, 60],
        "C2 (custo)"    : [400, 450, 300],
        "C3 (benefício)": [3, 4, 5],
    }, index=["A", "B", "C"])
    st.markdown("**1) Dados originais**")
    st.dataframe(dados, use_container_width=True)

    # Fórmulas gerais (min–max)
    with st.expander("Como transformamos a Matriz 1 na Matriz 2 (normalizada)", expanded=True):
        st.markdown("**Benefício (↑)** — quanto maior, melhor:")
        st.latex(r"\tilde{x}_{ij}=\frac{x_{ij}-\min(x_j)}{\max(x_j)-\min(x_j)}")
        st.markdown("**Custo (↓)** — quanto menor, melhor:")
        st.latex(r"\tilde{x}_{ij}=\frac{\max(x_j)-x_{ij}}{\max(x_j)-\min(x_j)}")
    
    # 2) Normalização min–max (0 a 1)
    def minmax_beneficio(col: pd.Series):
        mn, mx = float(col.min()), float(col.max())
        den = mx - mn
        if np.isclose(den, 0):
            return pd.Series(0.0, index=col.index), mn, mx, den
        return (col - mn) / den, mn, mx, den

    def minmax_custo(col: pd.Series):
        mn, mx = float(col.min()), float(col.max())
        den = mx - mn
        if np.isclose(den, 0):
            return pd.Series(0.0, index=col.index), mn, mx, den
        return (mx - col) / den, mn, mx, den
    
    # Função utilitária para exibir a conta célula a célula
    def mostra_conta(col_nome_curto, sentido, alt, val, mn, mx, den):
        if np.isclose(den, 0):
            st.markdown(f"- **{alt}**: denominador = 0 ⇒ valor normalizado definido como **0.0000**")
            return
        if sentido == "benefit":
            num = val - mn
            st.latex(
            rf"\tilde{{x}}_{{{alt},{col_nome_curto}}}="
            rf"\frac{{{val:.4g}-{mn:.4g}}}{{{mx:.4g}-{mn:.4g}}}"
            rf"=\frac{{{num:.4g}}}{{{den:.4g}}}"
            rf"={num/den:.4f}"
        )
        else:  # cost
            num = mx - val
            st.latex(
            rf"\tilde{{x}}_{{{alt},{col_nome_curto}}}="
            rf"\frac{{{mx:.4g}-{val:.4g}}}{{{mx:.4g}-{mn:.4g}}}"
            rf"=\frac{{{num:.4g}}}{{{den:.4g}}}"
            rf"={num/den:.4f}"
        )
    
    norm_demo = pd.DataFrame(index=dados.index)
    explicacao = {}

    # C1 — benefício
    st.markdown("### 2) Normalização passo a passo — **C1 (benefício)**")
    col = dados["C1 (benefício)"].astype(float)
    norm_demo["C1"], mn1, mx1, den1 = minmax_beneficio(col)
    explicacao["C1"] = (mn1, mx1)
    st.markdown(f"min = **{mn1}**, max = **{mx1}**, denominador = **{den1}**")
    for alt, val in col.items():
        mostra_conta("C1", "benefit", alt, float(val), mn1, mx1, den1)

    # C2 — custo
    st.markdown("### 2) Normalização passo a passo — **C2 (custo)**")
    col = dados["C2 (custo)"].astype(float)
    norm_demo["C2"], mn2, mx2, den2 = minmax_custo(col)
    explicacao["C2"] = (mn2, mx2)
    st.markdown(f"min = **{mn2}**, max = **{mx2}**, denominador = **{den2}**")
    for alt, val in col.items():
        mostra_conta("C2", "cost", alt, float(val), mn2, mx2, den2)

    # C3 — benefício
    st.markdown("### 2) Normalização passo a passo — **C3 (benefício)**")
    col = dados["C3 (benefício)"].astype(float)
    norm_demo["C3"], mn3, mx3, den3 = minmax_beneficio(col)
    explicacao["C3"] = (mn3, mx3)
    st.markdown(f"min = **{mn3}**, max = **{mx3}**, denominador = **{den3}**")
    for alt, val in col.items():
        mostra_conta("C3", "benefit", alt, float(val), mn3, mx3, den3)

    st.markdown("**Matriz 2 — normalizada (0 a 1, sempre no sentido de benefício)**")
    st.dataframe(norm_demo.round(4), use_container_width=True)

    # 3) Pesos e soma ponderada
    w_demo = np.array([0.5, 0.3, 0.2])  # soma = 1
    cols_demo = ["C1", "C2", "C3"]
    saw_demo = pd.Series(
    norm_demo[cols_demo].values.dot(w_demo),
    index=norm_demo.index,
    name="Pontuação SAW"
        )

    st.markdown("**3) Pontuação final (soma ponderada com os pesos)**")
    st.dataframe(saw_demo.to_frame().round(4), use_container_width=True)

    # 4) Ranking do exemplo
    ranking_demo = saw_demo.sort_values(ascending=False).to_frame("Pontuação SAW")
    st.markdown("**4) Ranking do exemplo**")
    st.dataframe(ranking_demo.round(4), use_container_width=True)

    with st.expander("Como ler esse resultado?"):
        st.markdown("""
- **B** ficou em 1º porque foi muito bem no critério mais importante (**C1**).
- **C** mandou bem em **C2 (custo)** e **C3**, mas perdeu força no **C1** (que pesa mais).
- **A** não liderou nenhum critério relevante, então ficou atrás.
""")

    # ---------- M2: TOPSIS ----------
    M2 = topsis(norm[cols], w, benefit)
    st.subheader("M2 — TOPSIS (proximidade ao ideal)")
    st.plotly_chart(fig_barras(M2, "Ranking — M2 (TOPSIS)"), use_container_width=True)

        # -----------------------
    # Explicação simples (intuição)
    # -----------------------
    with st.expander("O que é o TOPSIS (em linguagem simples)?", expanded=True):
        st.markdown("""
**Ideia em 1 frase:** o TOPSIS escolhe a alternativa **mais perto do cenário ideal** (o melhor possível em todos os critérios) e **mais longe do cenário anti-ideal** (o pior possível).

Como ele faz isso:
1. Colocamos cada critério numa **escala comparável** (normalização) e aplicamos os **pesos**.
2. Montamos dois pontos de referência:
   - **Ideal**: o melhor valor ponderado em cada critério.
   - **Anti-ideal**: o pior valor ponderado em cada critério.
3. Calculamos a **distância** de cada alternativa ao ideal e ao anti-ideal.
4. A pontuação final é a **proximidade relativa ao ideal**: quanto mais perto do ideal (e longe do anti-ideal), **melhor o ranking**.
""")
        
        # Modelagem (passo a passo sem jargão pesado)
    # -----------------------
    with st.expander("Como calculamos (modelagem por trás)", expanded=False):
        st.markdown("""
**Passo 1 — Normalização** (deixar comparável): usamos normalização **vetorial** (dividir pela raiz da soma dos quadrados) ou outra equivalente que você defina no pipeline.

**Passo 2 — Pesos:** multiplicamos cada coluna normalizada pelo **peso** do critério.

**Passo 3 — Ideal e Anti-ideal:**
- Para **benefícios (↑)**: ideal = **máximo** ponderado; anti-ideal = **mínimo**.
- Para **custos (↓)**: ideal = **mínimo** ponderado; anti-ideal = **máximo**.

**Passo 4 — Distâncias:** medimos a distância (Euclidiana) de cada alternativa ao **ideal** e ao **anti-ideal**.

**Passo 5 — Proximidade relativa:**""")
        # >>> AQUI está a correção: usar st.latex para renderizar a fórmula
        st.latex(r"C_i = \frac{D_i^-}{D_i^+ + D_i^-}")
        st.markdown("onde **D+** é a distância ao ideal e **D-** é a distância ao anti-ideal.")
    
        # Vantagens e Limitações
    # -----------------------
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ✅ Vantagens")
        st.markdown("""
- **Intuitivo**: “mais perto do ideal, mais longe do anti-ideal”
- **Considera melhor e pior cenário** simultaneamente
- **Escala comparável** via normalização vetorial (estável para unidades distintas)
- **Dá noção de separação** entre alternativas (distâncias)
""")
    with col2:
        st.markdown("### ⚠️ Limitações")
        st.markdown("""
- **Depende da normalização/pesos** (escolhas influenciam o resultado)
- **Distância Euclidiana** assume critérios independentes
- **Pode sofrer rank reversal** (como outros métodos com normalização global)
- **Critérios muito correlacionados** podem exagerar a influência de um aspecto
""")
        
    # -----------------------
    # Exemplo numérico (didático e auto-contido)
    # -----------------------
    st.subheader("Exemplo numérico (passo a passo)")
    st.caption("Mesmo exemplo do SAW para facilitar comparação. C1 e C3 = benefício (↑); C2 = custo (↓). Pesos w = [0,5, 0,3, 0,2].")

    # 1) Matriz original
    X = pd.DataFrame({
        "C1": [70, 90, 60],   # benefício
        "C2": [400, 450, 300],# custo
        "C3": [3, 4, 5],      # benefício
    }, index=["A", "B", "C"])
    sense_demo = {"C1": True, "C2": False, "C3": True}  # True=benefício, False=custo
    w_demo = np.array([0.5, 0.3, 0.2], dtype=float)      # soma=1

    st.markdown("**1) Dados originais**")
    st.dataframe(X, use_container_width=True)

    # Fórmulas gerais TOPSIS
    with st.expander("Fórmulas usadas no TOPSIS", expanded=True):
        st.markdown("**Normalização vetorial (por coluna):**")
        st.latex(r"r_{ij} = \frac{x_{ij}}{\sqrt{\sum_{i=1}^{n} x_{ij}^{2}}}")
        st.markdown("**Ponderação:**")
        st.latex(r"v_{ij} = w_j \cdot r_{ij}")
        st.markdown("**Pontos de referência (por critério \(j\)):**")
        st.markdown("- **Benefício (↑):**")
        st.latex(r"A_j^{+}=\max_i v_{ij},\qquad A_j^{-}=\min_i v_{ij}")
        st.markdown("- **Custo (↓):**")
        st.latex(r"A_j^{+}=\min_i v_{ij},\qquad A_j^{-}=\max_i v_{ij}")
        st.markdown("**Distâncias (Euclidiana):**")
        st.latex(r"D_i^+ = \sqrt{\sum_{j=1}^{m} (v_{ij}-A_j^+)^2}, \quad D_i^- = \sqrt{\sum_{j=1}^{m} (v_{ij}-A_j^-)^2}")
        st.markdown("**Proximidade relativa (pontuação TOPSIS):**")
        st.latex(r"C_i = \frac{D_i^-}{D_i^+ + D_i^-}")
    
    # ---------------------------------------------
    # 2) Matriz 2 (normalização vetorial) — contas coluna a coluna
    st.markdown("### 2) Normalização vetorial — **Matriz 1 → Matriz 2 (R)**")

    def col_norm_l2(col: pd.Series):
        denom = float(np.sqrt((col.astype(float)**2).sum()))
        if np.isclose(denom, 0.0):
            return pd.Series(0.0, index=col.index), denom
        return col.astype(float) / denom, denom
    
    def mostra_normalizacao_coluna(col_name, col, denom):
        st.markdown(f"**{col_name}** — denominador da normalização:")
        st.latex(rf"\sqrt{{\sum x^2}} = {denom:.6f}")
        for alt, val in col.items():
            if np.isclose(denom, 0.0):
                st.markdown(f"- **{alt}**: denominador = 0 ⇒ **0.0000**")
            else:
                st.latex(
                rf"r_{{{alt},{col_name}}} = \frac{{{float(val):.6f}}}{{{denom:.6f}}} = {float(val)/denom:.6f}"
            )

    R = pd.DataFrame(index=X.index)
    R["C1"], den1 = col_norm_l2(X["C1"]); mostra_normalizacao_coluna("C1", X["C1"], den1)
    R["C2"], den2 = col_norm_l2(X["C2"]); mostra_normalizacao_coluna("C2", X["C2"], den2)
    R["C3"], den3 = col_norm_l2(X["C3"]); mostra_normalizacao_coluna("C3", X["C3"], den3)
    
    st.markdown("**Matriz 2 — R (normalizada vetorialmente)**")
    st.dataframe(R.round(6), use_container_width=True)

    # ---------------------------------------------
    # 3) Matriz ponderada V = R * w — contas célula a célula
    # ---------------------------------------------

    st.markdown("### 3) Ponderação — **V = R × w**")
    V = R.copy()
    for j, c in enumerate(V.columns):
        V[c] = V[c] * w[j]
        # Mostrar as contas da coluna c
        st.markdown(f"**{c}** — peso = **{w[j]:.4f}**")
        for alt, rij in R[c].items():
            st.latex(
            rf"v_{{{alt},{c}}} = w_{{{c}}} \cdot r_{{{alt},{c}}} = {w[j]:.6f} \cdot {rij:.6f} = {w[j]*rij:.6f}"
            )
    
    st.markdown("**Matriz ponderada — V**")
    st.dataframe(V.round(6), use_container_width=True)

    # ---------------------------------------------
    # 4) Ideal (A⁺) e Anti-ideal (A⁻) — escolha por benefício/custo
    # ---------------------------------------------

    st.markdown("### 4) Pontos de referência — **Ideal (A⁺)** e **Anti-ideal (A⁻)**")

    def mostra_ideal_anti(c: str, serie: pd.Series, is_benefit: bool):
        # lista os v_ij por alternativa
        st.markdown(f"**{c} — valores ponderados (v_ij):**")
        for alt, val in serie.items():
            st.latex(rf"v_{{{alt},{c}}} = {float(val):.6f}")
        # escolhe A+ e A− conforme benefício/custo e mostra a conta
        valores = ", ".join([f"{float(v):.6f}" for v in serie.values])
        if is_benefit:
            vmax = float(serie.max()); imax = serie.idxmax()
            vmin = float(serie.min()); imin = serie.idxmin()
            st.markdown("**Benefício (↑):**")
            st.latex(rf"A_{{{c}}}^+ = \max\{{{valores}\}} = {vmax:.6f}")
            st.caption(f"Atingido por **{imax}**.")
            st.latex(rf"A_{{{c}}}^- = \min\{{{valores}\}} = {vmin:.6f}")
            st.caption(f"Atingido por **{imin}**.")
        else:
            vmin = float(serie.min()); imin = serie.idxmin()
            vmax = float(serie.max()); imax = serie.idxmax()
            st.markdown("**Custo (↓):**")
            st.latex(rf"A_{{{c}}}^+ = \min\{{{valores}\}} = {vmin:.6f}")
            st.caption(f"Atingido por **{imin}**.")
            st.latex(rf"A_{{{c}}}^- = \max\{{{valores}\}} = {vmax:.6f}")
            st.caption(f"Atingido por **{imax}**.")
            st.markdown("---")
    
    A_plus, A_minus = {}, {}
    for c in V.columns:
        mostra_ideal_anti(c, V[c], sense_demo[c])
        if sense_demo[c]:  # BENEFÍCIO
            A_plus[c]  = float(V[c].max()); A_minus[c] = float(V[c].min())
        else:             # CUSTO
            A_plus[c]  = float(V[c].min()); A_minus[c] = float(V[c].max())
    
    A_plus  = pd.Series(A_plus)
    A_minus = pd.Series(A_minus)

    st.write("**Resumo — Ideal (A⁺)**")
    st.dataframe(A_plus.to_frame("valor").T.round(6), use_container_width=True)
    st.write("**Resumo — Anti-ideal (A⁻)**")
    st.dataframe(A_minus.to_frame("valor").T.round(6), use_container_width=True)

    # ---------------------------------------------
    # 5) Distâncias D+ e D− — com substituição numérica
    # ---------------------------------------------
    st.markdown("### 5) Distâncias até A⁺ e A⁻ — **D⁺** e **D⁻**")

    # Mostra a soma dos quadrados antes da raiz
    def dist_explica(alt, row, target, name):
        termos = []
        for c in V.columns:
            diff = row[c] - target[c]
            termos.append((c, diff, diff**2))
        # expressão Latex
        soma = sum(t[2] for t in termos)    
        st.markdown(f"**{alt} → {name}**")
        partes = " + ".join([rf"({t[1]:.6f})^2" for t in termos])
        st.latex(rf"\sqrt{{{partes}}} = \sqrt{{{soma:.6f}}} = {np.sqrt(soma):.6f}")

    D_plus  = pd.Series(index=V.index, dtype=float)
    D_minus = pd.Series(index=V.index, dtype=float)

    for alt, row in V.iterrows():
        # D+ (até A+)
        dist_explica(alt, row, A_plus,  "D^+")
        D_plus[alt] = np.sqrt(((row - A_plus)**2).sum())
        # D- (até A-)
        dist_explica(alt, row, A_minus, "D^-")
        D_minus[alt] = np.sqrt(((row - A_minus)**2).sum())

    # ---------------------------------------------
    # 6) Proximidade relativa C = D- / (D+ + D-)
    # ---------------------------------------------
    st.markdown("### 6) Proximidade relativa — **pontuação TOPSIS**")
    C = D_minus / (D_plus + D_minus)
    C.name = "TOPSIS"

    resumo = pd.DataFrame({
    "D+ (dist. ao ideal)": D_plus,
    "D- (dist. ao anti-ideal)": D_minus,
    "Pontuação TOPSIS": C
    })
    st.dataframe(resumo.round(6), use_container_width=True)

    # ---------------------------------------------
    # 7) Ranking final
    # ---------------------------------------------

    st.markdown("**7) Ranking do exemplo (TOPSIS)**")
    ranking_topsis = C.sort_values(ascending=False).to_frame("Pontuação TOPSIS")
    st.dataframe(ranking_topsis.round(6), use_container_width=True)

    with st.expander("Como ler esse resultado?"):
        st.markdown("""
    - A alternativa com **maior pontuação TOPSIS** é a que fica **mais próxima do ideal (A⁺)** e **mais distante do anti-ideal (A⁻)**.
    - Se você ajustar **pesos** ou mudar a **normalização**, os pontos de referência mudam e o ranking pode mudar.
    - Para critérios de **custo (↓)**, verifique se o `sense` está correto — é isso que inverte quem é ideal/anti-ideal.
    """)

    # -----------------------
    # Nota prática sobre dados reais
    # -----------------------
    st.info("""
**Nota prática:** se existir critério de **custo (↓)**, garanta que o vetor `benefit` esteja correto.
Para dados com **outliers** ou **assimetria** (ex.: PIB), considere normalização **robusta** (quantis/log) ANTES do TOPSIS ou ajuste a normalização vetorial para reduzir o efeito de extremos.
""")
    
    # ---------- M3: VIKOR ----------
    st.subheader("M3 — VIKOR (solução de compromisso)")
    v_param = st.slider("Parâmetro v (peso da estratégia)", 0.0, 1.0, 0.5, 0.05)
    V = norm[cols].values
    S = (w * (1.0 - V)).sum(axis=1)
    R = (w * (1.0 - V)).max(axis=1)
    S_star, S_minus = S.min(), S.max()
    R_star, R_minus = R.min(), R.max()
    Q = v_param * (S - S_star) / (S_minus - S_star + 1e-12) + (1 - v_param) * (R - R_star) / (R_minus - R_star + 1e-12)
    M3 = pd.Series(1.0 - Q, index=norm.index, name="VIKOR")  # maior melhor
    st.plotly_chart(fig_barras(M3, "Ranking — M3 (VIKOR)"), use_container_width=True)

    # -----------------------
    # Dados (iguais aos modelos anteriores)
    # -----------------------
    X = pd.DataFrame({
    "C1": [70, 90, 60],   # benefício
    "C2": [400, 450, 300],# custo
    "C3": [3, 4, 5],      # benefício
    }, index=["A", "B", "C"])

    sense_demo = {"C1": True, "C2": False, "C3": True}      # True=benefício (↑) / False=custo (↓)
    w_demo = np.array([0.5, 0.3, 0.2], dtype=float)

    st.subheader("Exemplo numérico (passo a passo)")
    st.caption("Mesmos dados dos modelos anteriores. C1 e C3 = benefício (↑); C2 = custo (↓). Pesos w = [0,5, 0,3, 0,2].")
    st.markdown("**1) Matriz original (valores brutos)**")
    st.dataframe(X, use_container_width=True)

    # ============================================================
    # ------------------------  M3: VIKOR  -----------------------
    # ============================================================
    st.header("M3 — VIKOR (solução de compromisso)")

    with st.expander("Fórmulas (VIKOR)", expanded=True):
        st.markdown("**Melhor e pior por critério:**")
        st.latex(r"\text{Benefício: } f_j^{*}=\max_i x_{ij},\quad f_j^{-}=\min_i x_{ij}\qquad"
             r"\text{Custo: } f_j^{*}=\min_i x_{ij},\quad f_j^{-}=\max_i x_{ij}")
        st.markdown("**Gap normalizado ponderado:**")
        st.latex(r"d_{ij}=w_j\cdot\frac{|f_j^{*}-x_{ij}|}{|f_j^{*}-f_j^{-}|}")
        st.markdown("**Índices de utilidade e arrependimento:**")
        st.latex(r"S_i=\sum_j d_{ij},\qquad R_i=\max_j d_{ij}")
        st.markdown("**Índice de compromisso (Q) com parâmetro \(v\):**")
        st.latex(r"Q_i= v\frac{S_i-S^*}{S^- - S^*} + (1-v)\frac{R_i-R^*}{R^- - R^*}")

    # Passo 2 — f* e f− por critério, com contas
    st.markdown("### 2) Melhores (f*) e piores (f−) por critério")
    f_star, f_minus = {}, {}
    for c in X.columns:
        col = X[c].astype(float)
        valores = ", ".join([f"{float(v):.6f}" for v in col.values])
        if sense_demo[c]:  # benefício
            f_star[c]  = float(col.max());  f_minus[c] = float(col.min())
            st.markdown(f"**{c} (benefício)**")
            st.latex(rf"f_{{{c}}}^*=\max\{{{valores}\}}={f_star[c]:.6f}")
            st.latex(rf"f_{{{c}}}^-=\min\{{{valores}\}}={f_minus[c]:.6f}")
        else:               # custo
            f_star[c]  = float(col.min());  f_minus[c] = float(col.max())
            st.markdown(f"**{c} (custo)**")
            st.latex(rf"f_{{{c}}}^*=\min\{{{valores}\}}={f_star[c]:.6f}")
            st.latex(rf"f_{{{c}}}^-=\max\{{{valores}\}}={f_minus[c]:.6f}")
        st.markdown("---")

    f_star  = pd.Series(f_star)
    f_minus = pd.Series(f_minus)
    den = (f_star - f_minus).abs().replace(0, np.nan)

    # Passo 3 — d_ij célula a célula; S e R
    st.markdown("### 3) Cálculo de \(d_{ij}\) (com substituições), \(S_i\) e \(R_i\)")
    D = pd.DataFrame(index=X.index, columns=X.columns, dtype=float)
    for c in X.columns:
        st.markdown(f"**{c}** — denominador:")
        st.latex(rf"|f_{{{c}}}^*-f_{{{c}}}^-| = {abs(f_star[c]-f_minus[c]):.6f}")
        wj = float(w_demo[list(X.columns).index(c)])
        for alt, xij in X[c].items():
            if pd.isna(den[c]) or np.isclose(den[c], 0.0):
                D.loc[alt, c] = 0.0
                st.markdown(f"- **{alt}**: denominador = 0 ⇒ \(d_{{{alt},{c}}}=0\)")
            else:
                gap = abs(f_star[c] - float(xij)) / den[c]
                dij = wj * gap
                D.loc[alt, c] = dij
                st.latex(
                rf"d_{{{alt},{c}}}={wj:.6f}\cdot\frac{{|{f_star[c]:.6f}-{float(xij):.6f}|}}{{|{f_star[c]:.6f}-{f_minus[c]:.6f}|}}={dij:.6f}"
            )
        st.markdown("---")

    st.markdown("**Matriz \\(d_{ij}\\)** (desvios normalizados e ponderados)")
    st.dataframe(D.round(6).style.format("{:.6f}"), use_container_width=True)

    st.markdown("### 3.1) Como cada \\(S_i\\) e \\(R_i\\) foram calculados (passo a passo)")
    S = pd.Series(dtype=float)
    R = pd.Series(dtype=float)

    def peso(c:str)->float:
        return float(w_demo[list(X.columns).index(c)])
    
    for alt in X.index:
        st.markdown(f"**Alternativa {alt}**")
        termos = []  # (c, wj, num, den, frac, dij)
        for c in X.columns:
            wj = peso(c)
            xij  = float(X.loc[alt, c])
            num  = abs(float(f_star[c]) - xij)
            denj = float(den[c]) if not pd.isna(den[c]) else np.nan
            frac = 0.0 if (np.isnan(denj) or np.isclose(denj,0.0)) else (num/denj)
            dij  = wj * frac
            termos.append((c, wj, num, denj, frac, dij))
            st.latex(
            rf"d_{{{alt},{c}}} = {wj:.6f}\cdot\frac{{|{float(f_star[c]):.6f}-{xij:.6f}|}}{{|{float(f_star[c]):.6f}-{float(f_minus[c]):.6f}|}}"
            rf" = {wj:.6f}\cdot\frac{{{num:.6f}}}{{{denj:.6f}}} = {dij:.6f}"
        )
        S_i = sum(t[5] for t in termos)
        R_i = max(t[5] for t in termos)    
        S.loc[alt] = S_i
        R.loc[alt] = R_i

        # linha de soma S_i
        soma_txt = " + ".join([f"{t[5]:.6f}" for t in termos])
        st.latex(rf"S_{{{alt}}} = " + soma_txt + rf" = {S_i:.6f}")

        # destaque do máximo para R_i
        c_max = max(termos, key=lambda t: t[5])[0]
        st.latex(rf"R_{{{alt}}} = \max\{{" + ", ".join([f"d_{{{alt},{t[0]}}}" for t in termos]) + rf"\}} = d_{{{alt},{c_max}}} = {R_i:.6f}")
        st.markdown("---")

        st.markdown("**Resumo — índices \\(S\\) (soma) e \\(R\\) (máximo)**")
        st.dataframe(pd.DataFrame({"S (soma)": S, "R (máximo)": R}).round(6).style.format("{:.6f}"), use_container_width=True)

    # ---------------------------------------------
    # Passo 4 — Q_i com normalização EXPLÍCITA
    # ---------------------------------------------
    st.markdown("### 4) Índice de compromisso \\(Q_i\\) — normalização numérica de S e R")
    v_param = st.slider("Escolha v (peso da estratégia de maioria)", 0.0, 1.0, 0.5, 0.05, key="vikor_v")

    S_min, S_max = float(S.min()), float(S.max())
    R_min, R_max = float(R.min()), float(R.max())
    S_rng = S_max - S_min if not np.isclose(S_max - S_min, 0.0) else 1.0
    R_rng = R_max - R_min if not np.isclose(R_max - R_min, 0.0) else 1.0

    st.markdown("**Constantes da normalização (neste exemplo):**")
    st.latex(rf"S^* = \min_i S_i = {S_min:.6f},\quad S^- = \max_i S_i = {S_max:.6f},\quad S^- - S^* = {S_rng:.6f}")
    st.latex(rf"R^* = \min_i R_i = {R_min:.6f},\quad R^- = \max_i R_i = {R_max:.6f},\quad R^- - R^* = {R_rng:.6f}")

    Q = pd.Series(index=X.index, dtype=float)
    for alt in X.index:
        numS = float(S.loc[alt] - S_min)
        numR = float(R.loc[alt] - R_min)
        QS   = numS / S_rng
        QR   = numR / R_rng
        Qi   = v_param * QS + (1 - v_param) * QR
        Q.loc[alt] = Qi

        # mostra a substituição para este alt
        st.markdown(f"**{alt} — cálculo de \\(Q_{alt}\\)**")
        st.latex(
        rf"Q_{{{alt}}} = {v_param:.2f}\cdot\frac{{S_{{{alt}}}-{S_min:.6f}}}{{{S_max:.6f}-{S_min:.6f}}}"
        rf" + {1-v_param:.2f}\cdot\frac{{R_{{{alt}}}-{R_min:.6f}}}{{{R_max:.6f}-{R_min:.6f}}}"
        rf" = {v_param:.2f}\cdot{QS:.6f} + {1-v_param:.2f}\cdot{QR:.6f} = {Qi:.6f}"
    )

    st.markdown("**Ranking VIKOR (menor Q é melhor)**")
    st.dataframe(Q.sort_values(ascending=True).to_frame("Q (VIKOR)").round(6).style.format("{:.6f}"), use_container_width=True)

    # ---------- M4: PROMETHEE II ----------
    st.subheader("M4 — PROMETHEE II (fluxo líquido)")
    Vmat = norm[cols].copy()
    alts = Vmat.index.tolist()
    n_alt = len(alts)
    pref_plus = np.zeros((n_alt, n_alt))
    pref_minus = np.zeros((n_alt, n_alt))
    for a in range(n_alt):
        for b in range(n_alt):
            if a == b: 
                continue
            delta = (Vmat.iloc[a].values - Vmat.iloc[b].values)
            P_ab = (delta > 0).astype(float)  # função usual
            pref_plus[a, b] = (w * P_ab).sum()
            pref_minus[a, b] = (w * (delta < 0).astype(float)).sum()
    phi = (pref_plus.sum(axis=1) - pref_minus.sum(axis=1)) / (n_alt - 1)
    M4 = pd.Series(phi, index=alts, name="PROMETHEE_II")
    st.plotly_chart(fig_barras(M4, "Ranking — M4 (PROMETHEE II)"), use_container_width=True)
    
    # -------------------------
    # Dados do exemplo (iguais aos outros modelos)
    # -------------------------
    X = pd.DataFrame({
    "C1": [70, 90, 60],   # benefício (↑)
    "C2": [400, 450, 300],# custo     (↓)
    "C3": [3,  4,  5],    # benefício (↑)
    }, index=["A","B","C"])

    sense = {"C1": True, "C2": False, "C3": True}       # True=benefício; False=custo
    w = np.array([0.5, 0.3, 0.2], dtype=float)     # pesos (soma=1); ordem = colunas de X

    st.subheader("PROMETHEE II — passo a passo (exemplo numérico)")
    st.caption("Critérios: C1 (↑), C2 (↓), C3 (↑). Pesos w = [0,5, 0,3, 0,2]. Função de preferência linear.")

    st.markdown("**1) Matriz original (valores brutos)**")
    st.dataframe(X, use_container_width=True)

    with st.expander("Fórmulas usadas", expanded=True):
        st.markdown("### Diferença por critério j (a vs b)")
        st.latex(r"\text{Benefício (↑):}\quad d_j = x_{aj} - x_{bj}")
        st.latex(r"\text{Custo (↓):}\quad d_j = x_{bj} - x_{aj}")
        st.markdown("### Função de preferência (linear):")
        st.latex(r"P_j(d) = \max\!\left(0,\; \min\!\left(1,\; \frac{d}{p_j}\right)\right)")
        st.markdown("### Preferência agregada:")
        st.latex(r"\Pi(a,b) = \sum_j w_j \, P_j(d_j)")
        st.markdown("### Fluxos:")
        st.latex(
        r"""\phi^+(a)=\frac{1}{n-1}\sum_{b\ne a}\Pi(a,b), \quad 
        \phi^-(a)=\frac{1}{n-1}\sum_{b\ne a}\Pi(b,a), \quad 
        \phi(a)=\phi^+(a)-\phi^-(a)"""
    )

    # -------------------------------------------------
    # Parâmetro da função de preferência: p_j = α·(max−min)
    # -------------------------------------------------
    st.markdown("### 2) Escala de preferência por critério")
    alpha = st.slider("α para p_j = α·(max−min)", 0.1, 2.0, 1.0, 0.1, key="prom_alpha_clean")
    ranges = (X.max() - X.min()).astype(float)
    p = (alpha * ranges).replace(0, 1.0)  # evita zero (se não houver variação)
    st.dataframe(pd.DataFrame({"range (max−min)": ranges, "p_j": p}).round(6).T, use_container_width=True)

    def P_linear(d, pj):
        if pj <= 0:
            return 0.0
        return float(np.clip(d / pj, 0.0, 1.0))
    
    # -------------------------------------------------
    # 3) Contas par-a-par (claras e com substituições)
    # -------------------------------------------------

    st.markdown("### 3) Comparações par-a-par e preferências agregadas Π(a,b)")

    alts = list(X.index)
    cols = list(X.columns)
    Pi = pd.DataFrame(0.0, index=alts, columns=alts, dtype=float)

    for i, a in enumerate(alts):
        for j, b in enumerate(alts):
            if a == b:
                continue
            st.markdown(f"**Par ({a}, {b})**")
            prefs = []  # guarda (c, wj, d, pj, Pj)
            for k, c in enumerate(cols):
                xa, xb = float(X.loc[a, c]), float(X.loc[b, c])
                # d com o sinal certo
                d = (xa - xb) if sense[c] else (xb - xa)
                pj = float(p[c])
                Pj = P_linear(d, pj)

            # Mostra a conta do d e do P_j(d)
                if sense[c]:
                    st.latex(rf"\text{{C{c[-1]}}}:\; d = x_{{{a},{c}}} - x_{{{b},{c}}} = {xa:.6f} - {xb:.6f} = {d:.6f}")
                else:
                    st.latex(rf"\text{{C{c[-1]}}}:\; d = x_{{{b},{c}}} - x_{{{a},{c}}} = {xb:.6f} - {xa:.6f} = {d:.6f}")
                st.latex(rf"P_{{{c}}}(d) = \max\!\Big(0,\min\big(1,\frac{{{d:.6f}}}{{{pj:.6f}}}\big)\Big) = {Pj:.6f}")

                prefs.append((c, float(w[k]), d, pj, Pj))

            # Soma ponderada Π(a,b)
            soma_terms = [rf"{w_k:.3f}\cdot{P_k:.6f}" for (_, w_k, _, _, P_k) in prefs]
            soma_str = " + ".join(soma_terms)
            pi_ab = sum(w_k * P_k for (_, w_k, _, _, P_k) in prefs)
            Pi.loc[a, b] = pi_ab
            st.latex(rf"\Pi({a},{b}) = {soma_str} = \mathbf{{{pi_ab:.6f}}}")
            st.markdown("---")

    # -------------------------------------------------
    # 4) Fluxos φ+, φ− e φ (ranking)
    # -------------------------------------------------
    st.markdown("### 4) Fluxos e ranking")
    n = len(alts)
    phi_plus  = Pi.sum(axis=1) / (n - 1)
    phi_minus = Pi.sum(axis=0) / (n - 1)
    phi = phi_plus - phi_minus

    tabela_fluxos = pd.DataFrame(
    {"φ+ (domina)": phi_plus, "φ− (dominado)": phi_minus, "φ (líquido)": phi}
    ).round(6)
    st.dataframe(tabela_fluxos, use_container_width=True)

    st.markdown("**Ranking PROMETHEE II (maior φ é melhor)**")
    st.dataframe(phi.sort_values(ascending=False).to_frame("φ (líquido)").round(6), use_container_width=True)

# ---------- M5: AHP-Gauss ----------
st.subheader("M5 — AHP-Gaussiano (pesos AHP + utilidades)")
st.caption("Matriz par-a-par de 7 critérios: demanda_cidade, PIB, populacao, aluguel_m2, roubo_ind, diesel, dist_media.")

import unicodedata
import numpy as np
import pandas as pd

# -------------------------
# Helpers de normalização e mapeamento de critérios
# -------------------------
def _norm(s: str) -> str:
    """normaliza: minúsculas, sem acentos, sem espaços/underscores/hífens."""
    s2 = unicodedata.normalize("NFKD", str(s))
    s2 = "".join(ch for ch in s2 if not unicodedata.combining(ch))
    s2 = s2.lower().strip().replace(" ", "").replace("_", "").replace("-", "")
    return s2

def _looks_like_Ck(name: str) -> bool:
    name = str(name).strip().upper()
    return len(name) >= 2 and name[0] == "C" and name[1:].isdigit()

def _numeric_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

def _auto_map_cols_to_perf(cols_labels: list, perf: pd.DataFrame) -> dict:
    """
    Tenta mapear cols_labels -> colunas do perf.
    Estratégias:
      1) se já existem em perf, mapeia identidade;
      2) se são C1..Ck e tamanho bate, mapeia por posição para colunas numéricas do perf;
      3) tenta casar por nome normalizado (case/acentos/underscore-insensitive).
    Retorna dict mapping {label_original: coluna_perf}.
    """
    perf_cols = list(perf.columns)
    mapping = {}

    # 1) Identidade para os que já existem
    for c in cols_labels:
        if c in perf_cols:
            mapping[c] = c

    # 2) Se rótulos são C1.. e faltam mapear, tentar por posição com colunas numéricas
    still = [c for c in cols_labels if c not in mapping]
    if still and all(_looks_like_Ck(c) for c in cols_labels):
        nc = _numeric_cols(perf)
        if len(nc) >= len(cols_labels):
            for i, c in enumerate(cols_labels):
                mapping[c] = nc[i]  # mapeia por ordem
            still = [c for c in cols_labels if c not in mapping]

    # 3) Nome normalizado
    if still:
        norm_perf = {_norm(p): p for p in perf_cols}
        for c in still:
            nc = _norm(c)
            if nc in norm_perf:
                mapping[c] = norm_perf[nc]

    return mapping

def _coerce_benefit_flags(crit_names, benefit_flags, default=True):
    """
    Retorna dict {crit: bool} cobrindo todos em crit_names.
    Aceita list/tuple/ndarray (posicional), Series (reindex) ou dict (.get).
    """
    if isinstance(benefit_flags, (list, tuple, np.ndarray)):
        if len(benefit_flags) != len(crit_names):
            raise ValueError(
                f"benefit_flags tem {len(benefit_flags)} itens; precisam ser {len(crit_names)}."
            )
        s = pd.Series(benefit_flags, index=crit_names, dtype="bool")
    elif isinstance(benefit_flags, pd.Series):
        s = benefit_flags.astype("bool").reindex(crit_names)
    elif isinstance(benefit_flags, dict):
        s = pd.Series({c: bool(benefit_flags.get(c, default)) for c in crit_names}, dtype="bool")
    else:
        raise TypeError("benefit_flags deve ser list/tuple/ndarray, pandas.Series ou dict.")
    if s.isna().any():
        missing = list(s[s.isna()].index)
        raise KeyError(f"benefit_flags sem chaves para: {missing}.")
    return s.astype(bool).to_dict()

def _align_cols_and_benefit(perf: pd.DataFrame, cols: list, benefit) -> tuple:
    """
    Resolve o alinhamento final de critérios:
      - tenta mapear automaticamente cols -> perf.columns
      - se sobrar algo sem mapear, abre UI p/ mapeamento manual
      - retorna (crit_final, benefit_map_final, mapping_usado)
    """
    perf_cols = list(perf.columns)
    mapping = _auto_map_cols_to_perf(cols, perf)

    not_mapped = [c for c in cols if c not in mapping]
    if not_mapped:
        st.warning("Alguns critérios não foram encontrados automaticamente. Faça o mapeamento manual abaixo.")
        with st.form("map_criterios"):
            st.write("**Mapeie cada critério da sua lista para uma coluna existente nos dados.**")
            choices = perf_cols
            sel = {}
            for c in cols:
                default_guess = mapping.get(c, None)
                idx_default = choices.index(default_guess) if default_guess in choices else 0
                sel[c] = st.selectbox(f"{c} →", choices=choices, index=idx_default, key=f"map_{c}")
            ok = st.form_submit_button("Usar esse mapeamento")
        if ok:
            mapping = sel.copy()
        else:
            # se o usuário não clicou, usar o melhor mapping parcial disponível
            for c in not_mapped:
                # fallback: primeira coluna numérica ainda não usada
                ncands = [p for p in _numeric_cols(perf) if p not in mapping.values()]
                mapping[c] = ncands[0] if ncands else perf_cols[0]

    # crit_final garantido existir no perf
    crit_final = [mapping[c] for c in cols]

    # benefit pode estar nas chaves antigas (ex.: C1) ou nas novas (ex.: demanda_cidade)
    # tentamos montar benefit por nomes FINAIS:
    if isinstance(benefit, dict):
        benefit_by_final = {}
        inv_map = {v: k for k, v in mapping.items()}
        for cf in crit_final:
            if cf in benefit:
                benefit_by_final[cf] = bool(benefit[cf])
            elif cf in inv_map and inv_map[cf] in benefit:
                benefit_by_final[cf] = bool(benefit[inv_map[cf]])
            else:
                # default: benefício True
                benefit_by_final[cf] = True
        benefit_map = _coerce_benefit_flags(crit_final, benefit_by_final)
    else:
        # lista/série/array posicional — reindexamos após o mapeamento
        benefit_map = _coerce_benefit_flags(crit_final, benefit)

    return crit_final, benefit_map, mapping

# ========= Alinhamento robusto (resolve definitivamente C1/C2/C3) =========
crit_labels = list(cols)  # rótulos informados
crit, benefit_map, name_mapping = _align_cols_and_benefit(perf, crit_labels, benefit)

# --- Guia rápido (AHP + Utilidade) ---
with st.expander("📘 Guia de ajuda — AHP (matriz) e Utilidade Gaussiana", expanded=False):
    st.markdown(r"""
**Como preencher a matriz (AHP):** use a escala de Saaty **1–9** (e intermediários 2,4,6,8).  
Preencha **apenas a metade superior** (inclua 1 na diagonal). O app completa os recíprocos.
- 1 = igual importância
- 3 = moderadamente mais importante
- 5 = fortemente mais importante
- 7 = muito fortemente mais importante
- 9 = extremamente mais importante  
*(opostos ficam como o inverso: se PIB vs Demanda = 3, então Demanda vs PIB = 1/3).*

**Utilidade Gaussiana:** defina **μ** (alvo) e **σ** (tolerância).  
Mais perto de μ ⇒ utilidade maior. Em **critérios de custo**, o app já reflete o valor antes de aplicar a gaussiana.
""")

# ===== Funções AHP =====
def pcm_from_weights(w_vec):
    w = np.asarray(w_vec, dtype=float)
    s = float(w.sum())
    w = (w / s) if s > 0 else np.ones_like(w)/len(w)
    A = (w[:, None] / w[None, :]).astype(float)
    np.fill_diagonal(A, 1.0)
    return A

def saaty_quantize_matrix(A, odd_only=True):
    """Odd-only: {1/9,1/7,1/5,1/3,1,3,5,7,9}."""
    if odd_only:
        scale = np.array([1/9, 1/7, 1/5, 1/3, 1, 3, 5, 7, 9], dtype=float)
    else:
        scale = np.array([1/9,1/8,1/7,1/6,1/5,1/4,1/3,1/2,1,2,3,4,5,6,7,8,9], dtype=float)
    n = A.shape[0]
    Aq = np.ones_like(A, dtype=float)
    for i in range(n):
        Aq[i, i] = 1.0
        for j in range(i+1, n):
            r = float(A[i, j])
            k = int(np.argmin(np.abs(np.log(scale) - np.log(max(1e-12, r)))))
            v = float(scale[k])
            Aq[i, j] = v
            Aq[j, i] = 1.0 / v
    return Aq

def _show_weights_and_A(w_vec, A_mat, crit_names):
    st.markdown("**Pesos AHP estimados:**")
    st.dataframe(pd.Series(w_vec/np.sum(w_vec), index=crit_names, name="peso")
                 .round(4).to_frame().T.style.format("{:.4f}"),
                 use_container_width=True)
    st.markdown("**Matriz A (gerada automaticamente):**")
    st.dataframe(pd.DataFrame(A_mat, index=crit_names, columns=crit_names)
                 .style.format("{:.4f}"), use_container_width=True)

def make_weights_within_groups(crit_names, benefit_flags, preset_name, differentiate: bool):
    """Retorna vetor de pesos (soma=1) conforme preset, com opção de diferenciar dentro dos grupos."""
    benefit_flags = _coerce_benefit_flags(crit_names, benefit_flags)

    idx    = {c: k for k, c in enumerate(crit_names)}
    custos = [c for c in crit_names if not benefit_flags[c]]
    benefs = [c for c in crit_names if benefit_flags[c]]

    w = np.zeros(len(crit_names), dtype=float)

    if preset_name == "Equilibrado":
        if differentiate:
            base = np.linspace(1.3, 0.7, len(crit_names))
            base = base / base.sum()
            for c, v in zip(crit_names, base): w[idx[c]] = v
        else:
            w[:] = 1.0 / len(crit_names)

    elif preset_name == "Foco em Custo":
        share_c, share_b = 0.70, 0.30
        if differentiate:
            cost_shape = {"aluguel_m2": 4, "roubo_ind": 3, "diesel": 2, "dist_media": 1}
            ben_shape  = {"demanda_cidade": 3, "PIB": 2, "populacao": 1}
        else:
            cost_shape = {c: 1 for c in custos}
            ben_shape  = {c: 1 for c in benefs}
        if custos:
            v = np.array([float(cost_shape.get(c, 1)) for c in custos])
            v = v / v.sum() * share_c
            for c, vv in zip(custos, v): w[idx[c]] = vv
        if benefs:
            v = np.array([float(ben_shape.get(c, 1)) for c in benefs])
            v = v / v.sum() * share_b
            for c, vv in zip(benefs, v): w[idx[c]] = vv

    elif preset_name == "Foco em Demanda":
        share_dem, share_others = 0.50, 0.50
        if "demanda_cidade" in idx: w[idx["demanda_cidade"]] = share_dem
        others = [c for c in crit_names if c != "demanda_cidade"]
        if differentiate:
            other_shape = {"PIB": 2, "roubo_ind": 2, "aluguel_m2": 1.5, "diesel": 1.3,
                           "dist_media": 1.2, "populacao": 1.0}
        else:
            other_shape = {c: 1 for c in others}
        if others:
            v = np.array([float(other_shape.get(c, 1)) for c in others])
            v = v / v.sum() * share_others
            for c, vv in zip(others, v): w[idx[c]] = vv

    elif preset_name == "Foco em PIB":
        share_pib, share_others = 0.50, 0.50
        if "PIB" in idx: w[idx["PIB"]] = share_pib
        others = [c for c in crit_names if c != "PIB"]
        if differentiate:
            other_shape = {"demanda_cidade": 2, "roubo_ind": 2, "aluguel_m2": 1.6,
                           "diesel": 1.3, "dist_media": 1.1, "populacao": 1.0}
        else:
            other_shape = {c: 1 for c in others}
        if others:
            v = np.array([float(other_shape.get(c, 1)) for c in others])
            v = v / v.sum() * share_others
            for c, vv in zip(others, v): w[idx[c]] = vv

    elif preset_name == "Foco em Custo + Demanda":
        share_c, share_dem, share_rest = 0.50, 0.30, 0.20
        rest = [c for c in benefs if c != "demanda_cidade"]
        if differentiate:
            cost_shape = {"aluguel_m2": 4, "roubo_ind": 3, "diesel": 2, "dist_media": 1}
            rest_shape = {"PIB": 2, "populacao": 1}
        else:
            cost_shape = {c: 1 for c in custos}
            rest_shape = {c: 1 for c in rest}
        if custos:
            v = np.array([float(cost_shape.get(c, 1)) for c in custos])
            v = v / v.sum() * share_c
            for c, vv in zip(custos, v): w[idx[c]] = vv
        if "demanda_cidade" in idx:
            w[idx["demanda_cidade"]] = share_dem
        if rest:
            v = np.array([float(rest_shape.get(c, 1)) for c in rest])
            v = v / v.sum() * share_rest
            for c, vv in zip(rest, v): w[idx[c]] = vv

    w = w / w.sum() if w.sum() > 0 else np.ones_like(w)/len(w)
    return w

# ===== UI =====
preset = st.selectbox(
    "Escolha um preset de estratégia:",
    ["Equilibrado", "Foco em Custo", "Foco em Demanda", "Foco em PIB", "Foco em Custo + Demanda"],
    index=1,
)
diff_intragroup = st.toggle("Diferenciar pesos dentro dos grupos (reduz '1' na PCM)", value=True)
forcar_saaty = st.toggle("Forçar escala de Saaty (apenas 1,3,5,7,9) e recalcular CR", value=True)

# Estado
if "ahp_A" not in st.session_state: st.session_state.ahp_A = None
if "ahp_ready" not in st.session_state: st.session_state.ahp_ready = False
if "ahp_preset_name" not in st.session_state: st.session_state.ahp_preset_name = None

# Botão para gerar A
if st.button("Gerar matriz A a partir do preset", use_container_width=True):
    w_base = make_weights_within_groups(crit, benefit_map, preset, differentiate=diff_intragroup)
    A = pcm_from_weights(w_base)           # consistente
    if forcar_saaty:
        A = saaty_quantize_matrix(A, odd_only=True)  # 1,3,5,7,9
    st.session_state.ahp_A = A
    st.session_state.ahp_ready = True
    st.session_state.ahp_preset_name = preset
    st.success(
        f"Preset aplicado: **{preset}** — "
        + ("Matriz projetada na escala (1,3,5,7,9)." if forcar_saaty else "Matriz consistente (CR = 0).")
    )

# Se ainda não gerou a matriz, interrompe
if not st.session_state.ahp_ready or st.session_state.ahp_A is None:
    st.info("Selecione um **preset** e clique em **Gerar matriz A** para calcular o M5 (AHP-Gauss).")
    st.stop()

# Matriz final (já quantizada se forcar_saaty=True)
A = st.session_state.ahp_A

# Sempre reestimamos os pesos e o CR a partir da matriz FINAL
w_ahp, CR_disp = ahp_weights_from_pcm(A)

# Exibe
_show_weights_and_A(w_ahp, A, crit)
st.write(
    f"**AHP — Consistency Ratio (CR):** `{CR_disp:.3f}` "
    + ("✅ (consistente/aceitável)" if CR_disp <= 0.10 else "⚠️ (revise preferências)")
    + f" — preset: **{st.session_state.ahp_preset_name}**."
)

# ===== Utilidade Gaussiana =====
with st.expander("Ajustar utilidade gaussiana (μ, σ) por critério", expanded=False):
    mu_sigma = {}
    for c in crit:
        x = perf[c]  # seguro: 'c' existe em perf.columns
        mu_default = float(x.max()) if benefit_map[c] else float(x.min())
        sigma_default = float(max(1e-6, (x.max() - x.min())/3.0))
        col1, col2 = st.columns(2)
        mu_sigma[c] = (
            col1.number_input(f"μ — {c}", value=mu_default, key=f"mu_{c}"),
            col2.number_input(f"σ — {c}", value=sigma_default, key=f"sig_{c}")
        )

util = pd.DataFrame(index=perf.index)
for c in crit:
    mu, sig = mu_sigma.get(
        c,
        (perf[c].max() if benefit_map[c] else perf[c].min(),
         max(1e-6, (perf[c].max() - perf[c].min())/3.0))
    )
    util[c] = gaussian_utility(perf[c], mu=mu, sigma=sig, benefit_flag=benefit_map[c])

# Agregação final do M5
w_ahp = w_ahp / w_ahp.sum()
M5 = pd.Series(util[crit].values.dot(w_ahp), index=util.index, name="AHP_Gauss")
st.plotly_chart(fig_barras(M5, "Ranking — M5 (AHP-Gaussiano)"), use_container_width=True)

# =========================================================
# PÓS-ANÁLISE PARA O BOARD
# - Prós & contras dos modelos
# - Correlação de rankings (Spearman)
# - Consenso de Borda
# - Importância (contribuição e sensibilidade)
# - Probabilidade de vitória (Monte Carlo)
# =========================================================

# ---------- 1) Prós & contras (texto executivo) ----------
st.subheader("Como ler os resultados — prós & contras dos modelos")
st.markdown("""
- **SAW (Soma Ponderada)** → simples e transparente; **compensatório** (excelente em um critério pode compensar outro ruim).  
- **TOPSIS** → favorece quem fica **perto do ideal e longe do anti-ideal**; penaliza “calcanhar de Aquiles”.  
- **VIKOR** → **solução de compromisso**; parâmetro **v** controla estratégia (consenso vs. satisfaça o pior critério).  
- **PROMETHEE II** → **outranking** (comparações par-a-par); **pouco compensatório**; bom para diferenças pequenas entre cidades.  
- **AHP-Gauss** → traz **julgamento estruturado** (AHP, checa **CR**) e **alvos (μ,σ)** por critério (útil quando há metas/zonas de conforto).
""")

# ---------- 2) Correlação de rankings (Spearman) ----------
st.subheader("Concordância entre métodos — correlação de rankings (Spearman ρ)")
rank_sources = {}
if 'M1' in locals(): rank_sources["SAW"] = M1.rank(ascending=False, method="min")
if 'M2' in locals(): rank_sources["TOPSIS"] = M2.rank(ascending=False, method="min")
if 'M3' in locals(): rank_sources["VIKOR"] = M3.rank(ascending=False, method="min")
if 'M4' in locals(): rank_sources["PROMETHEE_II"] = M4.rank(ascending=False, method="min")
rank_sources["AHP_Gauss"] = M5.rank(ascending=False, method="min")

rank_df = pd.DataFrame(rank_sources).loc[M5.index]  # sincroniza índice
rho = rank_df.corr(method="spearman")
st.dataframe(rho.style.format("{:.2f}"), use_container_width=True)
st.caption("ρ≈1 indica forte acordo de ranking entre os métodos; ρ≈0 indica baixa concordância.")

# ---------- 3) Consenso de Borda ----------
st.subheader("Ranking de consenso — método de Borda")
m = len(rank_df)  # nº de alternativas
borda_points = (m - rank_df + 1).sum(axis=1)  # 1º m pts, 2º m-1, ..., último 1
borda_rank = borda_points.sort_values(ascending=False)
st.dataframe(
    pd.DataFrame({
        "Pontos_Borda": borda_points,
        "Posição": borda_rank.rank(ascending=False, method="min").astype(int)
    }).loc[borda_rank.index].style.format({"Pontos_Borda": "{:.0f}"}),
    use_container_width=True
)
st.caption("Borda agrega os rankings dos 5 métodos em um ranking de consenso.")

# ---------- 6) Vencedor por modelo ----------
st.subheader("Vencedor por modelo (5 métodos)")

def _winners_of(series: pd.Series):
    if series is None or not isinstance(series, pd.Series) or series.empty:
        return "N/D", np.nan
    s = series.dropna()
    if s.empty:
        return "N/D", np.nan
    m = s.max()
    v = s[s == m].index.tolist()
    return ", ".join(map(str, v)), float(m)

models = [
    ("SAW",           M1 if 'M1' in locals() else None),
    ("TOPSIS",        M2 if 'M2' in locals() else None),
    ("VIKOR",         M3 if 'M3' in locals() else None),
    ("PROMETHEE II",  M4 if 'M4' in locals() else None),
    ("AHP-Gauss",     M5),  # M5 existe neste bloco
]

rows = []
for name, series in models:
    winner, score = _winners_of(series)
    rows.append({"Modelo": name, "Vencedor": winner, "Score": score})

df_winners = pd.DataFrame(rows)
st.dataframe(
    df_winners.style.format({"Score": "{:.4f}"}),
    use_container_width=True
)


