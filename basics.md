Below is a “walk-through” of the whole file, written as if you were explaining it to a friend who knows what a spreadsheet is, has heard of “machine-learning” and “PyTorch”, but does **not** want to read source code line-by-line.

---

## 1. What is this file for?

Imagine you have a huge Excel workbook that contains daily stock information for hundreds of tickers.
Our end-goal is a deep-learning model that can look at the last 120 trading days of one ticker and predict how its price might change over several future horizons (1, 5, 20 days).

To get from the raw CSV to “ready-for-model” tensors, we need to:

1. Clean the data.
2. Fill in any missing trading days.
3. Engineer lots of extra columns (features and targets).
4. Turn each 120-day slice into something PyTorch understands.

Everything you see in `data.py` is doing one of those four jobs.

---

## 2. Key libraries used

• **Polars (`pl`)** – a super-fast alternative to Pandas for columnar data.
• **PyTorch (`torch`)** – the deep-learning library.
• **torch.utils.data.Dataset** – a PyTorch interface that says “you can ask me for the *n*-th training sample and I’ll give it to you”.
• **dataclasses.dataclass** – a neat way to store configuration (what columns to use, window size, etc.).

---

## 3. The configuration box – `DataConfig`

Think of `DataConfig` as a recipe card that tells later code:

• Which columns are:
  – constant for a ticker (`static_*`),
  – known in advance like the calendar (`time_varying_known_reals`),
  – only revealed day-by-day like yesterday’s return (`time_varying_unknown_reals`).

• Which columns are the targets (e.g. `target_1d`, `target_5d`, …).
• How many look-back days we want (120 here).

We pass this “recipe” into the Dataset so it knows what to grab.

---

## 4. The star of the show – `MarketDataset`

`MarketDataset` is the bridge between the processed Polars table and PyTorch.

### 4.1. Constructor (`__init__`)

1. **Adds an internal row index** – This acts like a permanent row number, so later filters can’t lose track of where we are.
2. **Finds “valid” rows** – A row is valid if:
   • we are at least 119 rows deep into a ticker’s history (so we *can* look back 120 days), **and**
   • we still have at least 20 future rows for targets (the longest horizon we’ll predict).
   The list of these row numbers is stored in `self.valid_indices`.
3. Prints out how many samples we can serve.

### 4.2. `__len__`

Returns the number of valid samples.

### 4.3. `__getitem__(idx)`

When PyTorch says “give me sample *idx*”, we:

1. **Locate the row** in the table that marks the *end* of the 120-day window.
2. Get its `ticker_id` and its calendar day (`time_idx`).
3. **Slice** exactly 120 rows **backwards** from that day for that ticker.
4. Sanity-check: if, for any reason, you don’t get 120 rows → raise an error (better explode than silently train garbage).
5. **Build tensors**:
   • `x_cat`  – static categorical stuff → one hot? (here still ints).
   • `x_known_reals` – numbers you already knew going in (calendar features).
   • `x_unknown_reals` – numbers you only get as the past unfolds (returns, rsi, …).
   • `y` – the target price changes sitting at the *end* of the window.
   • Plus helper items (`groups`, `time_idx_window`) that other libraries like [pytorch-forecasting](https://github.com/Nixtla/neuralforecast) like to see.

Returned as a tidy Python dict.

---

## 5. Helper utilities – from raw CSV to a pristine table

Below is the 30-second elevator pitch for each helper function.

### 5.1. `_clean_col_names(df)`

Standardises every header to `snake_case` so “Closing Price” → `closing_price`. Why? Because inconsistent column names are a nightmare.

### 5.2. `load_and_clean_data(csv_path)`

1. Reads your CSV with Polars.
2. Converts weird “N/A”, “#N/A”, empty strings, … into proper nulls.
3. Keeps only the columns we care about (`date`, `ticker`, `close`, `open`, `volume`, `industry`, `market_cap`).
4. Makes sure essential columns are not null.
5. Shows you a preview.

Result: a **clean but still gappy** daily table.

### 5.3. `create_mappings(df, output_dir)`

Stock tickers and industries are text strings. Neural networks prefer integers:

• Builds `ticker_map` – e.g. “AAPL” → 0, “AMZN” → 1, …
• Builds `industry_map` – e.g. “Tech” → 0, “Retail” → 1, …
• Saves them to disk (so later code can repeat the mapping) and merges the new IDs back into the DataFrame.

Now every row has a `ticker_id` (int) and `sector_id` (int).

### 5.4. `reindex_and_fill_gaps(df)`

Real stock data skips weekends and holidays, but some tickers also have *missing* trading days due to bad data.
We want each ticker’s timeline to be a perfect daily calendar with “holes” clearly marked.

Steps:

1. Sort by `ticker_id`, `date`.
2. For each ticker:
   • “Upsample” → create rows for every day, even if they didn’t exist.
   • Add a column `is_missing` that is `True` when the row was artificially created.
3. Forward-fill numeric columns up to `max_ffill_days` (5) so a single missing day just copies yesterday’s close/volume, but a long blackout stays null.
4. Throw away any rows that still have no `ticker_id` (shouldn’t happen, but double-safety).

After this, every ticker has a **continuous** date index. Missing days are known, tiny holes are patched.

### 5.5. `create_features_and_targets(df)`

This is the “feature kitchen”. For each row (ticker-day) it cooks up:

• **Targets:**
  – `target_1d`: % log-return from today to tomorrow.
  – `target_5d`: … over 5 days.
  – `target_20d`: … over 20 days.

• **Known-future features** (calendar stuff): day-of-week, month, quarter-end flag, …

• **Observed-past features**:
  – 1/5/20-day log returns,
  – 20-day rolling volatility, skewness, kurtosis,
  – 20-day z-score of volume,
  – RSI(14),
  – MACD (and signal line),
  – … etc.

Finally it:

1. Drops rows with *any* nulls (after engineering; you don’t want NaNs going into the network).
2. Creates `time_idx` – a simple 0,1,2,… counter per ticker (needed by many time-series models).

---

## 6. The `__main__` block – turning it into a pipeline

When you `python data.py`, this last portion runs:

1. Defines where the files live (`data/raw/stock_data.csv`, `data/processed/`).
2. Runs the full chain:

```
CSV → load_and_clean_data
     → create_mappings
     → reindex_and_fill_gaps
     → create_features_and_targets
     → save as Parquet
```

3. Prints a sample of the final DataFrame.
4. Writes `processed_data.parquet` so your training script can load *one* compact file instead of re-running the pipeline every time.

---

## 7. How the pieces fit together

```
┌───────────────┐
│  stock_data   │  (raw CSV)
└──────┬────────┘
       │ load_and_clean_data
       ▼
┌───────────────┐
│ cleaned table │
└──────┬────────┘
       │ create_mappings
       ▼
┌────────────────┐
│ mapped table   │  (ticker_id, sector_id)
└──────┬─────────┘
       │ reindex_and_fill_gaps
       ▼
┌─────────────────┐
│ gap-filled data │
└──────┬──────────┘
       │ create_features_and_targets
       ▼
┌───────────────────┐
│ final feature set │
└────────┬──────────┘
         │ MarketDataset (120-day windows)
         ▼
   PyTorch dataloader
```

---

## 8. TL;DR

1. **Load & tidy** the raw CSV.
2. **Turn ticker strings into ints.**
3. **Make the calendar continuous** for each ticker and patch small gaps.
4. **Engineer smart features** and future targets.
5. **Wrap** everything in a PyTorch-friendly Dataset that serves 120-day slices plus targets.

That’s it! 🎉
