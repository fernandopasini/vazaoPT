# Aplicativo Streamlit - Previsao de Vazao com Random Forest

Este projeto aplica um modelo Random Forest chuva-vazao com memoria hidrologica `Q(t-1)`.

## Arquivos principais

- `treinar_modelo_rf.py`: treina os modelos com a serie historica e gera o `.pkl`.
- `funcoes_previsao.py`: funcoes usadas para carregar o modelo e simular vazao futura.
- `app_streamlit.py`: interface amigavel em Streamlit.
- `requirements.txt`: dependencias para rodar no Streamlit Cloud.
- `rf_chuva_vazao_com_memoria.pkl`: arquivo gerado depois do treinamento; deve ser colocado no repositorio para o app funcionar diretamente.

## Como treinar no Colab

1. Suba `treinar_modelo_rf.py` no Colab.
2. Rode o script.
3. Envie a planilha historica com as colunas configuradas no script, por padrao:
   - `Data`
   - `TricolorP`
   - `TricolorV`
   - `PirayP`
   - `PirayV`
4. O script gerara `rf_chuva_vazao_com_memoria.pkl` e `metricas_modelo_rf.xlsx`.

## Como rodar localmente

```bash
pip install -r requirements.txt
streamlit run app_streamlit.py
```

## Como publicar no Streamlit Community Cloud

1. Crie um repositorio no GitHub.
2. Suba estes arquivos:
   - `app_streamlit.py`
   - `funcoes_previsao.py`
   - `requirements.txt`
   - `rf_chuva_vazao_com_memoria.pkl`
3. Entre no Streamlit Community Cloud.
4. Conecte sua conta GitHub.
5. Escolha o repositorio, branch e o arquivo principal `app_streamlit.py`.
6. Clique em Deploy.

## Observacao

O modelo nao preve chuva. O usuario deve informar a chuva futura prevista. O app simula a vazao futura usando a relacao chuva-vazao aprendida na serie historica.
