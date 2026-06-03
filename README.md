# Predicción de la dirección de acciones tecnológicas con ensembles + sentimiento FinBERT

Proyecto final de la *Tecnicatura en Ciencia de Datos* (módulo de Machine Learning: clasificación y regresión).

Sistema end-to-end que predice la **dirección diaria** (sube / baja) del precio de acciones tecnológicas
combinando **indicadores técnicos**, **datos macroeconómicos (FRED)**, **fundamentales (FMP)** y
**sentimiento de noticias procesado con FinBERT**, mediante un **ensemble por regímenes + stacking**.
Incluye un **backtest forward** out-of-sample y un **backtest de portafolio** con asignación equal-risk.

> ⚠️ **Aviso**: proyecto educativo / de investigación. **No** constituye asesoramiento financiero.

---

## 🎯 Qué demuestra este proyecto

- **Pipeline de ML completo**: ingesta de datos por API → feature engineering → entrenamiento → calibración → evaluación → backtesting.
- **Ensembles avanzados**: modelos por régimen temporal (2000–2019 vs 2020–2025) combinados por *weighted vote* y *stacking*, con **umbrales de decisión optimizados** por F1.
- **NLP financiero**: extracción de noticias (Finnhub → Google News RSS → Yahoo) y scoring de sentimiento con **FinBERT** (`yiyanghkust/finbert-tone`).
- **Integración de múltiples fuentes**: precios y fundamentales de *Financial Modeling Prep (FMP)*, series macro de *FRED*, titulares de *Finnhub*.
- **Evaluación rigurosa**: holdout temporal, matrices de confusión, *permutation importance*, *drift* entre periodos.
- **Backtesting cuantitativo**: estrategias *Stacking puro* vs *Stacking + filtro ADX*, métricas de win-rate, Sharpe, max drawdown y construcción de portafolio multi-activo.

---

## 📓 Notebooks

| Notebook | Descripción |
|---|---|
| [`modeloparapresentar_finbertreal.ipynb`](modeloparapresentar_finbertreal.ipynb) | Caso profundo sobre **NVDA**: EDA, sentimiento **FinBERT real**, ensemble por régimen, matrices de confusión e importancia de variables. |
| [`modelomejoradoparalinkedin.ipynb`](modelomejoradoparalinkedin.ipynb) | Pipeline **multi-ticker** (AAPL, AMD, META, MSFT, NVDA) + backtest de portafolio. |
| [`modelomejoradoparalinkedin_mejoras.ipynb`](modelomejoradoparalinkedin_mejoras.ipynb) | Versión refactorizada del anterior (manejo robusto de claves vía variables de entorno y helpers de red). |

---

## 🧠 Arquitectura del modelo

```
Precios (FMP) ─┐
Macro (FRED) ──┤
Fundamentales ─┼─► Feature engineering ─► Modelo P1 (2000–2019) ─┐
   (FMP)       │   (técnicos: retornos,     Modelo P2 (2020–2025) ─┼─► Weighted vote / Stacking
Noticias ──────┘    EMA, volatilidad, ADX)                         │   (umbrales óptimos por F1)
 (Finnhub →                                                        ▼
  FinBERT)                                              Señal direccional + Backtest
```

Los modelos base incluyen **XGBoost**, **LightGBM**, **RandomForest** y **LogisticRegression**,
combinados en un meta-modelo de *stacking*.

---

## 📊 Resultados

### Clasificación (holdout temporal — *period vote*)

| Ticker | F1 | Accuracy |
|--------|------|------|
| AAPL | 0.882 | 0.885 |
| AMD  | 0.851 | 0.857 |
| META | 0.903 | 0.876 |
| MSFT | 0.871 | 0.863 |
| NVDA | 0.868 | 0.859 |

### Backtest forward (sep–dic 2025, *Stacking + ADX*)

| Ticker | Trades | Win-rate | Sharpe | Retorno total | Max DD |
|--------|:------:|:--------:|:------:|:-------------:|:------:|
| AMD  | 26 | 50.0% | **1.72** | **+27.7%** | -11.7% |
| AAPL | 56 | 48.2% | 0.37 | +1.5% | -6.5% |
| NVDA | 32 | 43.8% | -0.81 | -7.1% | -14.9% |
| MSFT | 27 | 48.2% | -0.77 | -2.6% | -5.3% |
| META | 20 | 40.0% | -1.75 | -12.1% | -16.6% |

> El forward (out-of-sample real) muestra resultados **mixtos** frente a la fuerte performance en
> holdout — un recordatorio honesto de la diferencia entre validación histórica y mercado real.
> Ver `multi_ticker_pnl_summary.csv` y `portfolio_backtest_aggregated.csv` para el detalle.

---

## 📁 Estructura del repositorio

```
.
├── modeloparapresentar_finbertreal.ipynb     # NVDA + FinBERT (caso profundo)
├── modelomejoradoparalinkedin.ipynb          # Multi-ticker + portafolio
├── modelomejoradoparalinkedin_mejoras.ipynb  # Versión refactorizada
├── sp500.csv / sp500_test.csv                # Benchmark de mercado
├── <TICKER>_ensemble.json                    # Config del ensemble (pesos + umbrales)
├── <TICKER>_results_summary.json             # Métricas de clasificación
├── <TICKER>_forward_*_ADXdyn.csv             # Señales/retornos forward por ticker
├── multi_ticker_pnl_summary.csv              # Resumen P&L de la estrategia
├── per_ticker_strategy_metrics.csv           # Métricas detalladas por ticker
├── portfolio_backtest_*.csv                  # Backtest de portafolio
├── portfolio_weights_equal_risk.csv          # Pesos equal-risk
├── features_*_2periods.csv                   # Features unión / intersección entre periodos
├── imp_permutation_*.csv                      # Permutation importance
├── cm_*.png                                   # Matrices de confusión
├── requirements.txt
└── .env.example                               # Plantilla de claves de API
```

> Los modelos entrenados (`*.joblib`) se excluyen del repositorio por tamaño; se regeneran al ejecutar los notebooks.

---

## 🚀 Cómo ejecutarlo

1. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configurar las claves de API** (gratuitas en [FMP](https://site.financialmodelingprep.com/), [FRED](https://fred.stlouisfed.org/docs/api/api_key.html) y [Finnhub](https://finnhub.io/)).
   Copiá `.env.example` y exportá las variables. En **Windows PowerShell**:
   ```powershell
   $env:FMP_API_KEY     = "tu_clave_fmp"
   $env:FRED_API_KEY    = "tu_clave_fred"
   $env:FINNHUB_API_KEY = "tu_clave_finnhub"
   ```
   Los notebooks leen las claves con `os.getenv(...)`; nunca se versionan claves en el código.

3. **Abrir y ejecutar** cualquiera de los notebooks en Jupyter / VS Code.

---

## 🛠️ Stack técnico

`Python` · `pandas` · `numpy` · `scikit-learn` · `XGBoost` · `LightGBM` ·
`PyTorch` + `Transformers` (FinBERT) · `matplotlib` · APIs `FMP` / `FRED` / `Finnhub`

---

## 👤 Autor

**Fabrizio Sola** — Tecnicatura en Ciencia de Datos.
