'''
Pulls 5-minute OHLCV bars for all 14 tickers from Alpaca.
Caches to parquet (not CSV) partitioned by ticker.
Handles pagination for multi-year history pulls. 
Should support incremental re-fetch (only pull new bars since last cached timestamp) 
so it's reusable during live phase for warm-up context.
'''