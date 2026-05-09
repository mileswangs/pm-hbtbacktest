======================
Polymarket HftBacktest
======================

Key Features
============

* Polymarket tick-level backtesting
* Fast execution with `Numba <https://numba.pydata.org/>`_ JIT and Rust
* Designed for strategy research and execution simulation

Getting started
===============

Installation
------------

.. code-block:: console

   pip install pm-hbtbacktest

Examples
========

.. code-block:: python

   from hftbacktest import (
       init_orderbook,
       polymarket_to_hbt,
       BacktestAssetPoly,
       ROIVectorMarketDepthBacktest,
       Recorder,
       GTC,
       LIMIT,
   )
   from hftbacktest.stats import PolyAssetRecord
   import pandas as pd
   from numba import njit
   import numpy as np


   # Sweep end-of-market strategy.
   @njit
   def endline_trading(
       hbt,
       recorder,
       up_trigger: float,
       stop_long: float,
       order_qty: float,
   ):
       asset_no = 0

       if not init_orderbook(hbt, asset_no):
           return

       hbt_tick_size = hbt.depth(asset_no).tick_size
       price_tick_size = 0.01

       # Strategy state: trigger and stop only once to avoid repeated
       # open/close cycles in the same end-of-market move.
       activated = False
       # side=1 means buy UP; side=-1 means buy DOWN, represented by selling
       # the UP contract.
       side = 0
       submitted_once = False
       stop_submitted = False

       up_trigger = min(max(up_trigger, price_tick_size), 1.0 - price_tick_size)
       down_trigger = 1.0 - up_trigger

       # stop_long is the UP long stop line; DOWN uses the symmetric stop_short.
       stop_long = min(max(stop_long, price_tick_size), 1.0 - price_tick_size)
       stop_short = 1.0 - stop_long
       order_qty = np.float64(max(order_qty, 0.0))

       # Run strategy logic every 100ms.
       while hbt.elapse(100_000_000) == 0:
           hbt.clear_inactive_orders(asset_no)
           depth = hbt.depth(asset_no)

           # Use mid price as the trigger price to reduce one-sided book noise.
           bid, ask = depth.best_bid, depth.best_ask
           mid = (bid + ask) / 2.0

           # Enter the end-of-market certainty area: break up to buy UP, break
           # down to buy DOWN.
           if not activated:
               if mid >= up_trigger:
                   activated = True
                   side = 1
               elif mid <= down_trigger:
                   activated = True
                   side = -1

           # Submit only one entry order after trigger.
           if activated and (not submitted_once):
               if side > 0:
                   p = up_trigger
               else:
                   p = down_trigger

               p = round(p / price_tick_size) * price_tick_size
               p = max(price_tick_size, min(1.0 - price_tick_size, p))

               # For a single-submit scenario, use the price tick index as order id.
               oid = np.uint64(round(p / hbt_tick_size))

               if side > 0:
                   hbt.submit_buy_order(asset_no, oid, p, order_qty, GTC, LIMIT, False)
               else:
                   hbt.submit_sell_order(asset_no, oid, p, order_qty, GTC, LIMIT, False)
               submitted_once = True

           # position is UP net position: positive means holding UP; negative can
           # be understood as holding DOWN.
           pos = hbt.position(asset_no)
           if (not stop_submitted) and (pos != 0):
               need_stop = (pos > 0 and mid <= stop_long) or (
                   pos < 0 and mid >= stop_short
               )
               if need_stop:
                   close_qty = np.float64(np.abs(pos))
                   if pos > 0:
                       px = (
                           round((bid - price_tick_size) / price_tick_size)
                           * price_tick_size
                       )
                       px = max(price_tick_size, min(1.0 - price_tick_size, px))
                       oid = np.uint64(round(px / hbt_tick_size))
                       hbt.submit_sell_order(
                           asset_no, oid, px, close_qty, GTC, LIMIT, False
                       )
                   else:
                       px = (
                           round((ask + price_tick_size) / price_tick_size)
                           * price_tick_size
                       )
                       px = max(price_tick_size, min(1.0 - price_tick_size, px))
                       oid = np.uint64(round(px / hbt_tick_size))
                       hbt.submit_buy_order(
                           asset_no, oid, px, close_qty, GTC, LIMIT, False
                       )
                   stop_submitted = True

           recorder.record(hbt)

       recorder.record(hbt)


   slug = "btc-updown-15m-1778263200"
   df = pd.read_parquet(
       f"https://s.wangshuox.com/poly_l2/{slug}.parquet",
       storage_options={"User-Agent": "Mozilla/5.0"},
   )

   data = polymarket_to_hbt(df)

   asset = BacktestAssetPoly().data(data)
   hbt = ROIVectorMarketDepthBacktest([asset])
   recorder = Recorder(hbt.num_assets, 5_000_000)

   endline_trading(
       hbt,
       recorder.recorder,
       up_trigger=0.84,
       stop_long=0.4,
       order_qty=5,
   )
   _ = hbt.close()

   BOOK_SIZE = 100
   stats = PolyAssetRecord(recorder.get(0)).resample("1s").stats(book_size=BOOK_SIZE)
   print(f"earn: {stats.earn}")
   stats.plot()
