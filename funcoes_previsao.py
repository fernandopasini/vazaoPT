# -*- coding: utf-8 -*-
"""
funcoes_previsao.py

Funcoes para carregar o modelo .pkl e simular vazao futura com base em:
- precipitacao recente observada;
- vazao atual observada;
- precipitacao futura prevista.

Este arquivo e importado pelo app_streamlit.py.
"""

from __future__ import annotations

from typing import Dict, Any, List

import joblib
import numpy as np
import pandas as pd

DEFAULT_MODEL_PATH = "rf_chuva_vazao_com_memoria.pkl"


def carregar_modelo(path_or_file=DEFAULT_MODEL_PATH) -> Dict[str, Any]:
    """Carrega o bundle salvo pelo script de treinamento."""
    bundle = joblib.load(path_or_file)
    if not isinstance(bundle, dict) or "models" not in bundle or "meta" not in bundle:
        raise ValueError("O arquivo .pkl nao parece ter o formato esperado: {'models', 'meta'}.")
    return bundle


def listar_sites(bundle: Dict[str, Any]) -> List[str]:
    """Lista as series/bacias disponiveis no modelo."""
    return list(bundle["models"].keys())


def obter_config_site(bundle: Dict[str, Any], site: str) -> Dict[str, str]:
    """Retorna nomes das colunas de chuva e vazao de uma serie."""
    try:
        return bundle["meta"]["sites"][site]
    except KeyError as exc:
        raise KeyError(f"Serie '{site}' nao encontrada no modelo.") from exc


def _normalizar_df_precipitacao(
    df: pd.DataFrame,
    date_col: str = "Data",
    p_candidates=("P", "P_mm", "Precipitacao", "Precipitação", "precipitacao", "chuva", "Chuva", "P_prevista_mm"),
) -> pd.DataFrame:
    """Padroniza um dataframe para colunas Data e P_mm."""
    d = df.copy()

    if date_col not in d.columns:
        possible_dates = [c for c in d.columns if "data" in str(c).lower() or "date" in str(c).lower()]
        if possible_dates:
            d = d.rename(columns={possible_dates[0]: date_col})
        else:
            raise ValueError("A tabela precisa ter uma coluna de data chamada 'Data'.")

    p_col = None
    for c in p_candidates:
        if c in d.columns:
            p_col = c
            break

    if p_col is None:
        other_cols = [c for c in d.columns if c != date_col]
        if len(other_cols) == 1:
            p_col = other_cols[0]
        else:
            raise ValueError("A tabela precisa ter uma coluna de precipitacao, por exemplo 'P_mm' ou 'P_prevista_mm'.")

    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d[p_col] = pd.to_numeric(d[p_col], errors="coerce").fillna(0.0)
    d = d.dropna(subset=[date_col]).sort_values(date_col)
    d = d[[date_col, p_col]].rename(columns={p_col: "P_mm"})
    return d


def preparar_historico_precipitacao(df_hist: pd.DataFrame) -> pd.DataFrame:
    """Prepara chuva recente observada. Recomenda-se ao menos 30 dias."""
    d = _normalizar_df_precipitacao(df_hist)
    d = d.drop_duplicates(subset=["Data"], keep="last").sort_values("Data")
    return d


def preparar_precipitacao_futura(df_fut: pd.DataFrame) -> pd.DataFrame:
    """Prepara chuva futura prevista."""
    d = _normalizar_df_precipitacao(df_fut)
    d = d.drop_duplicates(subset=["Data"], keep="last").sort_values("Data")
    return d.rename(columns={"P_mm": "P_prevista_mm"})


def _montar_linha_features(
    p_series: pd.Series,
    q_lag_val: float,
    p_col_modelo: str,
    q_col_modelo: str,
    feature_cols: List[str],
    lags_p: List[int],
    windows_p: List[int],
    q_lags: List[int],
) -> pd.DataFrame:
    """Monta uma linha de features com os nomes esperados pelo modelo treinado."""
    row = {}

    for L in lags_p:
        nome = f"{p_col_modelo}_lag{L}"
        if nome in feature_cols:
            row[nome] = float(p_series.iloc[-1 - L]) if len(p_series) > L else np.nan

    for w in windows_p:
        nome = f"{p_col_modelo}_sum{w}"
        if nome in feature_cols:
            row[nome] = float(p_series.iloc[-w:].sum()) if len(p_series) >= w else np.nan

    for ql in q_lags:
        nome = f"{q_col_modelo}_lag{ql}"
        if nome in feature_cols:
            # O modelo atual usa apenas Q(t-1). Para q_lags maiores, seria necessario
            # passar historico de vazao. Aqui mantemos Q(t-1) como memoria operacional.
            row[nome] = float(q_lag_val)

    X = pd.DataFrame([row]).reindex(columns=feature_cols)

    if X.isna().any(axis=None):
        missing = X.columns[X.isna().any()].tolist()
        raise ValueError(
            "Nao ha dados suficientes para montar todas as variaveis do modelo. "
            f"Variaveis incompletas: {missing}. Informe ao menos 30 dias de precipitacao recente."
        )

    return X


def simular_vazao_futura(
    bundle: Dict[str, Any],
    site: str,
    df_hist_precip: pd.DataFrame,
    q_atual: float,
    df_fut_precip: pd.DataFrame,
) -> pd.DataFrame:
    """
    Simula vazao futura de forma sequencial/recursiva.

    Para o primeiro dia futuro, usa Q atual observada como Q(t-1).
    Para os dias seguintes, usa a vazao prevista do dia anterior como Q(t-1).
    """
    if site not in bundle["models"]:
        raise KeyError(f"Serie '{site}' nao existe no modelo.")

    model = bundle["models"][site]
    meta = bundle["meta"]
    site_cfg = meta["sites"][site]

    p_col_modelo = site_cfg["p_col"]
    q_col_modelo = site_cfg["q_col"]
    feature_cols = meta["features"][site]
    lags_p = list(meta.get("lags_p", [0, 1, 2, 3, 5, 7, 10, 14]))
    windows_p = list(meta.get("windows_p", [3, 5, 7, 14, 30]))
    q_lags = list(meta.get("q_lags", [1]))

    hist = preparar_historico_precipitacao(df_hist_precip)
    fut = preparar_precipitacao_futura(df_fut_precip)

    if len(hist) < max(windows_p):
        raise ValueError(
            f"Foram informados {len(hist)} dias de chuva recente. "
            f"O modelo precisa de pelo menos {max(windows_p)} dias para calcular acumulados."
        )

    p_all = hist[["Data", "P_mm"]].copy()
    p_all = p_all.drop_duplicates(subset=["Data"], keep="last").sort_values("Data")
    p_all = p_all.set_index("Data")["P_mm"].astype(float)

    q_lag = float(q_atual)
    results = []

    for _, row_f in fut.iterrows():
        data = pd.to_datetime(row_f["Data"])
        p_prev = float(row_f["P_prevista_mm"])

        p_all.loc[data] = p_prev
        p_all = p_all.sort_index()

        X = _montar_linha_features(
            p_series=p_all,
            q_lag_val=q_lag,
            p_col_modelo=p_col_modelo,
            q_col_modelo=q_col_modelo,
            feature_cols=feature_cols,
            lags_p=lags_p,
            windows_p=windows_p,
            q_lags=q_lags,
        )

        q_pred = float(model.predict(X)[0])

        results.append({
            "Data": data.date(),
            "P_prevista_mm": p_prev,
            "Q_lag_usada_m3s": q_lag,
            "Q_prevista_m3s": q_pred,
        })

        q_lag = q_pred

    return pd.DataFrame(results)
