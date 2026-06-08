# -*- coding: utf-8 -*-
"""
app_streamlit.py

Interface Streamlit para previsao/simulacao de vazao com modelo Random Forest salvo em .pkl.

Rodar localmente:
    streamlit run app_streamlit.py

No GitHub/Streamlit Cloud, deixe no repositorio:
    - app_streamlit.py
    - funcoes_previsao.py
    - rf_chuva_vazao_com_memoria.pkl
    - requirements.txt
"""

from __future__ import annotations

from pathlib import Path
from datetime import date

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from funcoes_previsao import carregar_modelo, listar_sites, simular_vazao_futura, DEFAULT_MODEL_PATH

st.set_page_config(
    page_title="Previsao de vazao - Random Forest",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def carregar_modelo_cache(path: str):
    return carregar_modelo(path)


def ler_tabela_upload(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    raise ValueError("Formato nao reconhecido. Use .csv, .xlsx ou .xls.")


def criar_historico_padrao(data_referencia: date, n_dias: int = 30) -> pd.DataFrame:
    datas = pd.date_range(end=pd.to_datetime(data_referencia), periods=n_dias, freq="D")
    return pd.DataFrame({"Data": datas.date, "P_mm": [0.0] * n_dias})


def criar_futuro_padrao(data_referencia: date, horizonte: int) -> pd.DataFrame:
    datas = pd.date_range(start=pd.to_datetime(data_referencia) + pd.Timedelta(days=1), periods=horizonte, freq="D")
    return pd.DataFrame({"Data": datas.date, "P_prevista_mm": [0.0] * horizonte})


def fig_resultados(df_res: pd.DataFrame):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=df_res["Data"],
            y=df_res["P_prevista_mm"],
            name="Precipitacao prevista (mm)",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df_res["Data"],
            y=df_res["Q_prevista_m3s"],
            mode="lines+markers",
            name="Vazao prevista (m³/s)",
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title="Precipitacao prevista e vazao simulada",
        xaxis_title="Data",
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Precipitacao (mm)", secondary_y=False)
    fig.update_yaxes(title_text="Vazao (m³/s)", secondary_y=True)
    return fig


st.title("Previsão de vazão com Random Forest")
st.caption(
    "O modelo usa a relacao chuva-vazao aprendida na série histórica e aplica essa relação "
    "às condições atuais: chuva recente, vazão atual e chuva futura prevista."
)

with st.sidebar:
    st.header("Modelo")
    usar_modelo_repo = st.checkbox(
        "Usar modelo do repositorio",
        value=Path(DEFAULT_MODEL_PATH).exists(),
        help="Marque esta opcao se o arquivo .pkl estiver junto ao app no GitHub/repo local.",
    )
    uploaded_model = None
    if not usar_modelo_repo:
        uploaded_model = st.file_uploader("Subir arquivo .pkl do modelo", type=["pkl"])

try:
    if usar_modelo_repo:
        if not Path(DEFAULT_MODEL_PATH).exists():
            st.error(f"Nao encontrei o arquivo {DEFAULT_MODEL_PATH} no repositorio/pasta do app.")
            st.stop()
        bundle = carregar_modelo_cache(DEFAULT_MODEL_PATH)
    else:
        if uploaded_model is None:
            st.info("Suba o arquivo .pkl do modelo para continuar.")
            st.stop()
        bundle = carregar_modelo(uploaded_model)
except Exception as exc:
    st.error(f"Erro ao carregar o modelo: {exc}")
    st.stop()

sites = listar_sites(bundle)

col1, col2, col3 = st.columns([1.2, 1, 1])
with col1:
    site = st.selectbox("Serie/bacia para simular", sites)
with col2:
    data_ref = st.date_input("Data atual / data de referencia", value=date.today())
with col3:
    q_atual = st.number_input("Vazão atual observada [m³/s]", min_value=0.0, value=10.0, step=0.1)

meta_site = bundle["meta"]["sites"][site]
st.info(
    f"Serie selecionada: **{site}** | Chuva usada no treinamento: `{meta_site['p_col']}` | "
    f"Vazao usada no treinamento: `{meta_site['q_col']}` | "
    f"Periodo treinado: {meta_site.get('period_start', '01/08/1967')} a {meta_site.get('period_end', '30/09/2022')}"
)

st.subheader("1) Chuva recente observada")
st.write(
    "Informe pelo menos 30 dias de precipitação recente. "
    "Esses valores sao usados para calcular os acumulados de 3, 5, 7, 14 e 30 dias."
)

modo_hist = st.radio(
    "Como deseja informar a chuva recente?",
    ["Digitar/editar tabela", "Subir CSV/Excel"],
    horizontal=True,
)

if modo_hist == "Subir CSV/Excel":
    up_hist = st.file_uploader("Arquivo com colunas Data e P_mm", type=["csv", "xlsx", "xls"], key="hist")
    if up_hist is not None:
        try:
            hist_df = ler_tabela_upload(up_hist)
        except Exception as exc:
            st.error(f"Erro ao ler historico: {exc}")
            st.stop()
    else:
        hist_df = criar_historico_padrao(data_ref, 30)
else:
    hist_df = criar_historico_padrao(data_ref, 30)

hist_df = st.data_editor(
    hist_df,
    num_rows="dynamic",
    use_container_width=True,
    key="hist_editor",
    column_config={
        "Data": st.column_config.DateColumn("Data"),
        "P_mm": st.column_config.NumberColumn("P_mm", min_value=0.0, step=0.1, format="%.2f"),
    },
)

st.subheader("2) Chuva futura prevista")
horizonte = st.number_input("Numero de dias futuros para simular", min_value=1, max_value=30, value=7, step=1)

fut_default = criar_futuro_padrao(data_ref, int(horizonte))
fut_df = st.data_editor(
    fut_default,
    num_rows="dynamic",
    use_container_width=True,
    key=f"future_editor_{horizonte}_{data_ref}",
    column_config={
        "Data": st.column_config.DateColumn("Data"),
        "P_prevista_mm": st.column_config.NumberColumn("P_prevista_mm", min_value=0.0, step=0.1, format="%.2f"),
    },
)

st.subheader("3) Simulacao")

if st.button("Simular vazao futura", type="primary"):
    try:
        res = simular_vazao_futura(
            bundle=bundle,
            site=site,
            df_hist_precip=hist_df,
            q_atual=q_atual,
            df_fut_precip=fut_df,
        )
        st.success("Simulacao concluida.")
        st.dataframe(res, use_container_width=True)
        st.plotly_chart(fig_resultados(res), use_container_width=True)

        csv = res.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Baixar resultados em CSV",
            data=csv,
            file_name=f"previsao_vazao_{site.replace(' ', '_')}.csv",
            mime="text/csv",
        )

        with st.expander("Como interpretar"):
            st.write(
                "Para o primeiro dia futuro, o modelo usa a vazao atual observada como Q(t-1). "
                "Para os dias seguintes, usa a vazao prevista do dia anterior. "
                "Por isso, quanto maior o horizonte, maior tende a ser a incerteza acumulada."
            )
    except Exception as exc:
        st.error(f"Nao foi possivel simular: {exc}")

with st.expander("Observacoes tecnicas"):
    st.markdown(
        """
- Este app nao recalibra o modelo. Ele carrega o `.pkl` ja treinado.
- O modelo atual usa apenas `Q(t-1)` como memoria direta de vazao.
- A chuva recente e necessaria porque o modelo usa acumulados moveis de chuva.
- Para previsoes de varios dias, a simulacao e recursiva: a vazao prevista de um dia vira a memoria do dia seguinte.
- O modelo nao deve ser extrapolado para condicoes muito diferentes das observadas na serie historica de treinamento.
"""
    )
