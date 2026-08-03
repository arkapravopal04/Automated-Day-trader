# Version 2

> An updated automeated day trader , this time using more advanced topics and clear maths
> Will try to post an actual data for this project


data/
    | --- fetch_alpaca.py
          Pulls 5-minute OHLCV bars for all 14 tickers from Alpaca. Caches to parquet (not CSV) partitioned by ticker. Handles pagination for multi-year history pulls. Should support incremental re-fetch (only pull new bars since last cached timestamp) so it's reusable during live phase for warm-up context.
    | --- dataset.py
    | --- preprocess.py