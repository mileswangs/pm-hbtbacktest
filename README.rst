======================
Polymarket HftBacktest
======================

核心特性
========

* Polymarket tick 级别回测
* 使用 `Numba <https://numba.pydata.org/>`_ JIT 和 Rust 实现快速执行
* 面向策略研究和撮合执行模拟

快速开始
========

安装
----

.. code-block:: console

   pip install pm-hftbacktest

示例
====

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


   # 扫尾盘策略。
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

       # 策略状态：只触发一次、止损一次，避免在同一段尾盘行情里反复开平仓。
       activated = False
       # side=1 表示买 UP；side=-1 表示买 DOWN，在 UP 合约上表现为卖出。
       side = 0
       submitted_once = False
       stop_submitted = False

       up_trigger = min(max(up_trigger, price_tick_size), 1.0 - price_tick_size)
       down_trigger = 1.0 - up_trigger

       # stop_long 是 UP 多头止损线；DOWN 方向使用对称的 stop_short。
       stop_long = min(max(stop_long, price_tick_size), 1.0 - price_tick_size)
       stop_short = 1.0 - stop_long
       order_qty = np.float64(max(order_qty, 0.0))

       # 每 100ms 运行一轮策略逻辑。
       while hbt.elapse(100_000_000) == 0:
           hbt.clear_inactive_orders(asset_no)
           depth = hbt.depth(asset_no)

           # 使用 mid 作为触发价，减少单边盘口噪声影响。
           bid, ask = depth.best_bid, depth.best_ask
           mid = (bid + ask) / 2.0

           # 进入尾盘确定性区域：向上突破买 UP，向下突破买 DOWN。
           if not activated:
               if mid >= up_trigger:
                   activated = True
                   side = 1
               elif mid <= down_trigger:
                   activated = True
                   side = -1

           # 触发后只提交一次开仓单。
           if activated and (not submitted_once):
               if side > 0:
                   p = up_trigger
               else:
                   p = down_trigger

               p = round(p / price_tick_size) * price_tick_size
               p = max(price_tick_size, min(1.0 - price_tick_size, p))

               # 单次提交场景里，直接用价格 tick 序号作为 order id。
               oid = np.uint64(round(p / hbt_tick_size))

               if side > 0:
                   hbt.submit_buy_order(asset_no, oid, p, order_qty, GTC, LIMIT, False)
               else:
                   hbt.submit_sell_order(asset_no, oid, p, order_qty, GTC, LIMIT, False)
               submitted_once = True

           # position 是 UP 合约净仓位：正数为持有 UP，负数可理解为持有 DOWN。
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
