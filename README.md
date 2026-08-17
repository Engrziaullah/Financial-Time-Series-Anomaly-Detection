# Financial Time Series Anomaly Detection

Detects anomalous price behavior in historical stock data by combining classic
technical indicators with two different anomaly-detection approaches: an
unsupervised **Isolation Forest** over indicator features, and a
**Prophet forecast**-based residual check. Implemented as a single Jupyter
notebook, originally authored and run on Kaggle.

## Data

- **Source:** [Massive Yahoo Finance Dataset](https://www.kaggle.com/datasets/iveeaten3223times/massive-yahoo-finance-dataset) (Kaggle), file `stock_details_5_years.csv`.
- **Content:** daily OHLCV records for many NASDAQ/NYSE-listed companies, with columns `Date, Open, High, Low, Close, Volume, Dividends, Stock Splits, Company`. Tickers observed in the notebook include `AAPL`, `WCN`, `FERG`, `CSGP`, `TTD`, `MELI`, `BKNG`, `NVR`, `AZO`, `CMG`, `MTD`, `FCNCA`, among others.
- The notebook was written for the Kaggle runtime and reads the CSV from `/kaggle/input/massive-yahoo-finance-dataset/stock_details_5_years.csv`. To run it outside Kaggle, download the dataset from the link above and update `file_path` in the notebook to point at your local copy.

## Technical indicators

Computed per company (grouped by `Company`, then sorted by `Date`):

| Indicator | Definition used in the notebook |
|---|---|
| `SMA_10` | 10-period simple moving average of `Close` |
| `EMA_10` | 10-period exponential moving average of `Close` |
| `RSI_14` | 14-period Relative Strength Index, computed from rolling average gain/loss |
| `Bollinger_Upper` / `Bollinger_Lower` | 20-period rolling mean of `Close` ± 2 rolling standard deviations |

## Anomaly detection methods

**1. Isolation Forest (scikit-learn)**
- Features: `Close`, `SMA_10`, `EMA_10`, `RSI_14`, `Bollinger_Upper`, `Bollinger_Lower` (missing values filled with 0).
- Parameters: `n_estimators=100`, `contamination=0.01`, `random_state=42`.
- Rows are labeled `is_anomaly = 1` when the model predicts `-1` (outlier), else `0`.

**2. Prophet forecast residuals**
- A `Prophet` model is fit on a single company's `Close` price series (the notebook uses `MELI`) and forecasts 60 days ahead.
- A point is flagged anomalous when the actual price falls outside the forecast's confidence interval: `forecast_anomaly = 1` if `y > yhat_upper` (higher than expected), `-1` if `y < yhat_lower` (lower than expected), else `0`.

## Visualization

- Close price with Isolation Forest anomalies overlaid as red markers, for a selected company (default `MELI`).
- Prophet forecast plot with confidence band, actual price, and forecast-based anomalies marked (red = above upper bound, orange = below lower bound).

## Results (from the saved notebook run)

- The source CSV had no missing values in any column (`Date`, `Open`, `High`, `Low`, `Close`, `Volume`, `Dividends`, `Stock Splits`, `Company`).
- The Isolation Forest step flagged **6,028 rows** as anomalous across the full dataset (`contamination=0.01`).
- Per-company anomaly counts in the saved output are concentrated in a handful of tickers rather than spread evenly across all companies: `BKNG` 1257, `NVR` 1257, `AZO` 1109, `CMG` 873, `MTD` 758, `MELI` 631, `FCNCA` 143. Note: these happen to be among the higher-priced tickers in the set; since the model is fit on raw price-scaled features (`Close`, `SMA_10`, `EMA_10`, Bollinger bands) pooled across all companies rather than per-company, stocks with much higher absolute price levels are more likely to be flagged as outliers relative to the rest of the dataset. This is a known limitation of fitting one global Isolation Forest across companies with very different price scales, rather than evidence that these specific dates were unusual for those companies.
- For `MELI` specifically, the Isolation Forest split was close to even within that company's rows (`is_anomaly`: 631 vs. 627), which is consistent with the scale-driven concentration above rather than a genuine 1% anomaly rate for that ticker.
- The Prophet-based check is demonstrated only on `MELI` in the saved notebook; no aggregate anomaly count across companies is computed for this method.

## Libraries used

- `pandas`, `numpy` — data loading and indicator computation
- `scikit-learn` (`IsolationForest`) — unsupervised anomaly detection
- `prophet` — time series forecasting (via its `cmdstanpy` backend)
- `matplotlib` — plotting

## Setup

```bash
git clone https://github.com/Engrziaullah/Financial-Time-Series-Anomaly-Detection.git
cd Financial-Time-Series-Anomaly-Detection
pip install -r requirements.txt
```

Then download `stock_details_5_years.csv` from the [dataset page](https://www.kaggle.com/datasets/iveeaten3223times/massive-yahoo-finance-dataset), update the `file_path` variable in `financial-time-series-anomaly-detection.ipynb` to your local path, and run the notebook with Jupyter:

```bash
jupyter notebook financial-time-series-anomaly-detection.ipynb
```

## Repository contents

- `financial-time-series-anomaly-detection.ipynb` — the full analysis: data loading, indicator computation, Isolation Forest anomaly detection, Prophet forecasting, and plots.

## Known limitations

- Isolation Forest is fit once across all companies on raw (non-normalized) price-based features, so results are biased toward higher-priced tickers (see Results above). Normalizing per-company (e.g., z-scoring `Close` and the indicators within each `Company` group) before fitting would likely give more balanced, per-company anomaly rates.
- The Prophet-based anomaly check is only demonstrated for one ticker (`MELI`); it is not run across the full universe of companies in this notebook.
