# Empirical L2/L3 Order-Book Datasets & Calibrated Queue-Reactive Tables

This directory contains curated empirical limit order book (LOB) datasets, message streams,
and calibrated multi-level Queue-Reactive / State-Dependent Hawkes intensity tables used by
`FlatLadderBook`, `ReplayDriver`, `OfflineHawkesCalibrator`, `QueueReactiveProcess`, the CLI
modes (`./build/sim --mode=calibrate|parity|qr-stylized`), and the publication notebooks in `../notebooks/`.

---

## 1. Dataset Provenance & Academic Citations

1. **LOBSTER (Limit Order Book System - The Efficient Reconstructor) NASDAQ TotalView-ITCH Samples**:
   - **Tickers & Session**: `AAPL`, `AMZN`, `GOOG`, `INTC`, `MSFT` on `2012-06-21` (`09:30:00.000` to `16:00:00.000` EST, 5 price levels).
   - **Citation**: Huang, R., & Polak, T. (2011). *LOBSTER: Limit Order Book Reconstruction System*. Humboldt University of Berlin / SSRN 1977207 (`https://lobsterdata.com/info/DataSamples.php`).
   - **Mirror**: HuggingFace `datasets/Ayrus-1007/lobster-data` (official LOBSTER 5-level message and orderbook sample CSVs). Full 10,000-row aligned slices per ticker are archived in `/home/radu/code/tacit-mui/docs/investigation-garden/inbox/queue_reactive_literature/datasets/`, while 2,000-row aligned prefixes are tracked directly in `data/` for fast deterministic CI and notebook execution.
2. **FI-2010 Benchmark Limit Order Book Dataset**:
   - **Description**: High-frequency 10-level limit order book feature & mid-price prediction benchmark from Nasdaq Nordic (149 feature columns + target horizon labels).
   - **Citation**: Ntakaris, A., Magris, M., Kanniainen, J., Gabbouj, M., & Iosifidis, A. (2018). *Benchmark Dataset for Mid-Price Forecasting of Limit Order Book Data with Machine Learning Methods*. Journal of Forecasting, 37(8), 852-866 (`arXiv:1705.03233`).
   - **Mirror**: HuggingFace `datasets/shanehans/FI2010` (`fi2010_benchmark_sample.csv`).
3. **Calibrated 5-Level HLR Model IIb/III & Noble-Rosenbaum-Souilmi (2026) 3D Empirical Intensity Tables**:
   - **Files**: `hlr_calibrated_5level.json`, `hlr_calibrated_5level.csv`, `qr_simulator_intensities.json`, `calib_binance_btcusdt.json`, `lobster_calibration_summary.json`.
   - **Citations**:
     * Huang, W., Lehalle, C.-A., & Rosenbaum, M. (2015). *Simulating and Analyzing Order Book Data: The Queue-Reactive Model*. Journal of the American Statistical Association (JASA), 110(509), 107-122 (`arXiv:1312.0563`).
     * Wu, P., Rambaldi, M., Muzy, J.-F., & Bacry, E. (2019). *Queue-Reactive Hawkes Models for the Order Flow* (`arXiv:1901.08938`).
     * Morariu-Patrichi, M., & Pakkanen, M. S. (2022). *State-Dependent Hawkes Processes and Their Application to Limit Order Book Modelling*. Quantitative Finance, 22(3), 563-583 (`arXiv:1809.08060`).
     * Bodor, H., & Carlier, L. (2024). *The Queue-Reactive Model: Importance of Order Sizes* (`arXiv:2405.18594`).
     * Noble, A., Rosenbaum, M., & Souilmi, S. (2026). *Bridging the Reality Gap in Limit Order Book Simulation* (`arXiv:2603.24137`).

---

## 2. Schema Documentation

### 2.1 LOBSTER Message CSV (`lobster_<TICKER>_2012-06-21_message_5.csv`)
Headerless 6-column CSV per official LOBSTER specification:
- `col[0]` (`time_s`): Seconds after midnight with nanosecond precision (`34200.0` = `09:30:00`).
- `col[1]` (`event_type`):
  * `1`: Submission of a new limit order (`TraceEventKind::SubmitLimit`)
  * `2`: Partial cancellation of a resting limit order (`TraceEventKind::Cancel`)
  * `3`: Total deletion of a resting limit order (`TraceEventKind::Cancel`)
  * `4`: Execution of a visible resting limit order (`TraceEventKind::SubmitMarket`)
  * `5`: Execution of a hidden resting limit order
  * `7`: Trading halt indicator
- `col[2]` (`order_id`): Unique external order reference number (`uint64_t`).
- `col[3]` (`size`): Number of shares (`int32_t`).
- `col[4]` (`price`): Dollar price multiplied by `10000` (e.g., `$585.33` -> `5853300`; 1 cent tick = `100`).
- `col[5]` (`direction`): `+1` for Buy (`Side::Buy`), `-1` for Sell (`Side::Sell`).

### 2.2 LOBSTER Orderbook CSV (`lobster_<TICKER>_2012-06-21_orderbook_5.csv`)
Headerless 20-column CSV aligned row-for-row with the message file (`4 * K` columns for `K = 5` levels):
- For level `m in {1..5}`: `ask_price_m, ask_size_m, bid_price_m, bid_size_m`.

### 2.3 Calibrated 3D Empirical Lookup Table (`hlr_calibrated_5level.csv` / `.json`)
525-cell 3D state-space grid `Phi(LOB) = (imbalance_bin, spread_bin, depth_bin)` (`21 x 5 x 5`):
- `imbalance_bin` (`0..20`): 21 bins of width `0.1` over `[-1.0, +1.0]` with exact-zero bin `10` per Noble, Rosenbaum & Souilmi (2026) Section 2.3.
- `spread_bin` (`0..4`): Spread in ticks `s in {1, 2, 3, 4, >=5}`.
- `depth_bin` (`0..4`): Total L1 volume `q_1^bid + q_1^ask` quantized via Eytzinger bin edges `[100, 250, 500, 1000]`.
- `lambda_add_per_us`, `lambda_cancel_per_us`, `lambda_exec_per_us`, `lambda_create_per_us`: Calibrated conditional arrival intensities in events per microsecond.

---

## 3. File Inventory & SHA-256 Checksums

| File | Size (Bytes) | Rows | SHA-256 Checksum |
| :--- | ---: | ---: | :--- |
| `calib_binance_btcusdt.json` | 596308 | 49521 | `3ca87381a1a27b12856870ea0d4c97b71c7d29e56624af10d0ca83686b584b47` |
| `fi2010_benchmark_sample.csv` | 813410 | 500 | `f87fb32bbe4adcf1f3986ce3909152352555178bfb9d8b6cd0afd3e0b5640c13` |
| `hlr_calibrated_5level.csv` | 27156 | 526 | `847365bf364244d4e63e16605e319a398e871f399edfe6c1b373067f43205b2c` |
| `hlr_calibrated_5level.json` | 171356 | 6517 | `c165f359ef4d0cf2dac5bcaa7fd00da0634c6cc83e0869dac11e50064629228c` |
| `lobster_AAPL_2012-06-21_message_5.csv` | 80275 | 2000 | `6beced9e1c5b6ea964e85f593ce4940919510682f3198e29151d99267dc5a085` |
| `lobster_AAPL_2012-06-21_orderbook_5.csv` | 231730 | 2000 | `7ee775eea9f791f0709b76280038d633001cf0217328a4a62a362385641d2231` |
| `lobster_AMZN_2012-06-21_message_5.csv` | 80746 | 2000 | `3757a4f0c5f798e694060f7ecfa19742807534135b2d0a4a01e8179436da5568` |
| `lobster_AMZN_2012-06-21_orderbook_5.csv` | 227822 | 2000 | `f73df42f7880a490668404cefba5ca54ef1a6f24e60e33fb008c57a596b34903` |
| `lobster_GOOG_2012-06-21_message_5.csv` | 80935 | 2000 | `d59ab8688fa0f3c5d8e7d985cbba3c1824ea0771f205c0ab293c3473e1719efe` |
| `lobster_GOOG_2012-06-21_orderbook_5.csv` | 228606 | 2000 | `0e86a364f97d516daac65057aefca86e4f378aac7aff240ae20f4e609077e4f7` |
| `lobster_INTC_2012-06-21_message_5.csv` | 80307 | 2000 | `e840fc4f4fc7ea4e5689bd83286f413e00f9b5a28b9e8b9051bfc0488a75f465` |
| `lobster_INTC_2012-06-21_orderbook_5.csv` | 227541 | 2000 | `86987d071c8834fec9c87f0cb0b914bee9bbfa97e3234a79e55e8bd7a5aa385e` |
| `lobster_MSFT_2012-06-21_message_5.csv` | 80569 | 2000 | `aa87c7723f0df0cea832e9e04a0921731d76726e11e970dedccd9532cb0d5aa1` |
| `lobster_MSFT_2012-06-21_orderbook_5.csv` | 233373 | 2000 | `645fb40170ae480fb51ea570ba63b01bee251f6e0df30ea58f7f49fe0fe2584e` |
| `lobster_calibration_summary.json` | 6418 | 188 | `710a9d00fe9fdd7dd4f38f2598a0df6ecee4b555ca744af7fcdc5f1ad224d8b2` |
| `qr_simulator_intensities.json` | 38780 | 1514 | `4f1b305ac9ca457d4405f8a36f6f5b7be7ad8bb424829ab7684e5bcc39e066ca` |

### Extended Raw Archive (`/home/radu/code/tacit-mui/docs/investigation-garden/inbox/queue_reactive_literature/datasets/`)

| Archive File | Size (Bytes) | Rows | SHA-256 Checksum |
| :--- | ---: | ---: | :--- |
| `AAPL_2012-06-21_message_5_10k.csv` | 405419 | 10000 | `492583a50709b28d1b5fb377de88919e2d949a2bef7644a9ae8f70aad07dcec5` |
| `AAPL_2012-06-21_orderbook_5_10k.csv` | 1158816 | 10000 | `36057b2adf5bc320dd689e02026a6cb618b757470957201e07a2e7ea92ab7910` |
| `AMZN_2012-06-21_message_5_10k.csv` | 409417 | 10000 | `c5dd0f3e4c52841d3fb800d1d7c762066054c5d334fa029335bf6b2c62eed81f` |
| `AMZN_2012-06-21_orderbook_5_10k.csv` | 1168877 | 10000 | `400dbe5356a271ad4ad895279b7c05968fd034e47c31217284e6b96917cfcceb` |
| `FI2010_test_slice.csv` | 1998467 | 1226 | `7ab964ddc903ad93315582139eb26d65c3dbabe200c9a989979c41660216b7cb` |
| `GOOG_2012-06-21_message_5_10k.csv` | 407833 | 10000 | `4156920db053c965184e56094f1c5dd96e0c2a1b9be9d79555d4184537b6fa97` |
| `GOOG_2012-06-21_orderbook_5_10k.csv` | 1157378 | 10000 | `a53108480381781385fe73d44ddb32337c965d321884504427f800b72a6ba488` |
| `INTC_2012-06-21_message_5_10k.csv` | 400846 | 10000 | `8978eb255bf7b986050148eebb8644c8feee466dc0b88dce6a27cf2cd76a38f9` |
| `INTC_2012-06-21_orderbook_5_10k.csv` | 1151706 | 10000 | `1234ecb0cd0f03538e994e5cd0bdc92bbf8248e75ae2025fd97b0e4a08989421` |
| `LOBSTER_SampleFiles_ReadMe.txt` | 4790 | 140 | `fd97b5f49391e11ef52c023ef8f3ca156baac7441b4d52240339f60844244d5b` |
| `MSFT_2012-06-21_message_5_10k.csv` | 403452 | 10000 | `62ccaf02406c0f53ba91880b247ccaa747780454d0a37ba157f328db10b80763` |
| `MSFT_2012-06-21_orderbook_5_10k.csv` | 1186542 | 10000 | `feb833be0901425293dc846a74ec38dd157031924d69c95b82b557c1b26f034b` |
