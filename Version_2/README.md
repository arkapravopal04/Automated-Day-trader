# Version 2

> An updated automeated day trader , this time using more advanced topics and clear maths
> Will try to post an actual data for this project


data/
    | --- fetch_alpaca.py
          Pulls 5-minute OHLCV bars for all 14 tickers from Alpaca. Caches to parquet (not CSV) partitioned by ticker. Handles pagination for multi-year history pulls. Should support incremental re-fetch (only pull new bars since last cached timestamp) so it's reusable during live phase for warm-up context.
    | --- dataset.py
          Feature engineering: returns (log return, multi-horizon), realized volatility windows, volume z-score, time-of-day/session encoding, per-instrument tick size lookup table. Normalization stats (mean/std) computed strictly on the training split to avoid lookahead leakage. Outputs a clean per-ticker feature tensor plus a metadata file (feature names, normalization constants, tick sizes) that later stages depend on.
    | --- preprocess.py
          Windowing and batching into vectorized rollout-ready tensors, shape [n_envs=14, window, features]. Train/val/test split done by date range, never by row, to avoid temporal leakage. Includes a sanity-check routine: no NaNs, no gaps beyond expected market-closed periods, no future information in any feature column.


env/
      | --- 