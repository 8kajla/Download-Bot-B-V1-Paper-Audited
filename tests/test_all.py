import tempfile,time
from pathlib import Path
from unittest.mock import patch
from strategy import ConvergenceStrategy
from paper_ledger import PaperLedger
from market_discovery import _ts,_normalize,discover

def test_strategy_no_early(): assert ConvergenceStrategy().decide(80,.9,.1,.89,.09,[],[],0,1000) is None
def test_strategy_late():
 s=ConvergenceStrategy(); x=s.decide(220,.94,.06,.93,.05,[(time.time()-30,.91)],[(time.time()-30,.09)],0,1000,up_depth=20,down_depth=20); assert x and x.side=='Up' and x.notional>0
def test_cap(): assert ConvergenceStrategy(max_market_exposure=5).decide(220,.94,.06,.93,.05,[],[],5,1000) is None
def test_cash_cap(): assert ConvergenceStrategy(max_order=10).decide(220,.94,.06,.93,.05,[],[],0,3,up_depth=20,down_depth=20).notional==3
def test_ledger():
 with tempfile.TemporaryDirectory() as d:
  l=PaperLedger(Path(d)/'s.json',100); t=l.buy('c','u','m','Up',.8,10,1); assert abs(t['shares']-12.5)<1e-9; assert l.mark({'u':.9})['unrealized']>1; assert l.settle('c','u'); assert l.realized>2.4
def test_cash_never_negative():
 with tempfile.TemporaryDirectory() as d:
  l=PaperLedger(Path(d)/'s.json',5); l.buy('c','u','m','Up',.5,10,1); assert l.cash==0
def test_ts_utc(): assert abs(_ts('2026-08-28T00:00:00Z')-1787875200)<1
def test_normalize():
 x=_normalize({'id':'m1','conditionId':'c1','slug':'btc-updown-5m-1787899800','question':'Bitcoin Up or Down','clobTokenIds':'["u","d"]','outcomes':'["Up","Down"]'}); assert x['asset']=='BTC' and x['start_ts']==1787899800 and x['end_ts']==1787900100
def test_discovery_slug_path():
 with patch('market_discovery._get_market_by_slug') as get:
  get.return_value={'id':'m1','conditionId':'c1','slug':'btc-updown-5m-1787899800','question':'Bitcoin Up or Down','clobTokenIds':'["u","d"]','outcomes':'["Up","Down"]','enableOrderBook':True}
  x=discover(now=1787899900); assert any(m['asset']=='BTC' for m in x)
def test_resolution_pending_contract():
 from market_discovery import _winner
 assert _winner({'tokens':[{'token_id':'u','outcome':'Up','winner':True},{'token_id':'d','outcome':'Down','winner':False}]})==('u','Up')
def test_resolution_price_contract():
 from market_discovery import _winner
 assert _winner({'clobTokenIds':'["u","d"]','outcomes':'["Up","Down"]','outcomePrices':'["1","0"]'})==('u','Up')
def test_resolution_unresolved():
 from market_discovery import _winner
 assert _winner({'clobTokenIds':'["u","d"]','outcomes':'["Up","Down"]','outcomePrices':'["0.7","0.3"]'})==(None,None)

def test_book_depth_shape():
 from market_discovery import book
 with patch('market_discovery.SESSION.get') as g:
  r=g.return_value; r.status_code=200; r.raise_for_status=lambda:None; r.json=lambda:{'bids':[{'price':'0.80','size':'10'}],'asks':[{'price':'0.82','size':'5'}]}
  assert book('u')==(0.8,0.82,10.0,5.0)

def test_ledger_metadata_persists():
 with tempfile.TemporaryDirectory() as d:
  l=PaperLedger(Path(d)/'s.json',100); l.buy('c','u','m','Up',.8,10,1,{'slug':'btc-updown-5m-1','start_ts':1,'end_ts':301,'asset':'BTC','up_token':'u','down_token':'d'}); l2=PaperLedger(Path(d)/'s.json',100); assert l2.positions[next(iter(l2.positions))]['slug'].startswith('btc-')


def test_research_files_and_resolution():
 from research_logger import ResearchLogger
 from types import SimpleNamespace
 with tempfile.TemporaryDirectory() as d:
  r=ResearchLogger(d)
  m={'id':'m1','condition':'c1','slug':'btc-updown-5m-1','asset':'BTC','market':'BTC','start_ts':1,'end_ts':301}
  r.record_decision(ts=2,market=m,elapsed=1,left=299,up_bid=.49,up_ask=.51,up_depth=10,down_bid=.49,down_ask=.51,down_depth=10,signal=None,exposure=0,cash=100)
  sig=SimpleNamespace(side='Up',price=.9,score=.8,notional=5,reason='test momentum=+0.100')
  tr={'ts':3,'condition':'c1','market_id':'m1','slug':m['slug'],'asset':'BTC','market':'BTC','side':'Up','token':'u','price':.9,'shares':5/.9,'notional':5}
  r.record_trade(trade=tr,market=m,elapsed=2,left=298,up_bid=.89,up_ask=.9,up_depth=20,down_bid=.1,down_ask=.11,down_depth=20,score=.8,momentum=.1,reason='test',cash_after=95,exposure_after=5)
  r.record_resolution(ts=304,market=m,winner='Up',winner_token='u',closed=[('c1:u',5/.9-5)])
  for fn in ('decisions.jsonl','orderbooks.jsonl','trades.csv','markets.csv','resolutions.csv','pnl_1min.csv'):
   assert Path(d,fn).exists()
  assert 'RESOLVED' in Path(d,'resolutions.csv').read_text()

def test_research_rebuild():
 from research_logger import ResearchLogger
 with tempfile.TemporaryDirectory() as d:
  r=ResearchLogger(d)
  class L: trades=[{'action':'BUY','condition':'c','notional':4,'shares':5,'ts':2,'asset':'BTC','slug':'s','market_id':'m','start_ts':1,'end_ts':301}]; positions={}
  r.rebuild_from_ledger(L())
  assert r.market_stats['c']['cost']==4


def test_market_result_has_side_breakdown():
 from research_logger import ResearchLogger
 with tempfile.TemporaryDirectory() as d:
  r=ResearchLogger(d)
  m={'id':'m2','condition':'c2','slug':'btc-updown-5m-2','asset':'BTC','market':'BTC','start_ts':1,'end_ts':301}
  for side,token in [('Up','u'),('Down','d')]:
   tr={'ts':2 if side=='Up' else 3,'condition':'c2','market_id':'m2','slug':m['slug'],'asset':'BTC','market':'BTC','side':side,'token':token,'price':.9,'shares':5/.9,'notional':5}
   r.record_trade(trade=tr,market=m,elapsed=2,left=298,up_bid=.89,up_ask=.9,up_depth=20,down_bid=.1,down_ask=.11,down_depth=20,score=.8,momentum=0,reason='test',cash_after=90,exposure_after=10)
  r.record_resolution(ts=304,market=m,winner='Up',winner_token='u',closed=[('c2:u',5/.9-5),('c2:d',-5)])
  import csv
  rows=list(csv.DictReader(Path(d,'markets.csv').open()))
  assert float(rows[0]['up_cost'])==5 and float(rows[0]['down_cost'])==5
  assert float(rows[0]['winning_cost'])==5 and float(rows[0]['losing_cost'])==5


def test_research_logger_prune(tmp_path):
    import json, time
    from research_logger import ResearchLogger
    d=tmp_path/"data"; r=ResearchLogger(d)
    old=time.time()-10*86400
    new=time.time()
    (d/"decisions.jsonl").write_text(
        json.dumps({"t":old,"x":"old"})+"\n"+json.dumps({"t":new,"x":"new"})+"\n",
        encoding="utf-8")
    r._prune_jsonl("decisions.jsonl", 7)
    lines=(d/"decisions.jsonl").read_text().splitlines()
    assert len(lines)==1
    assert json.loads(lines[0])["x"]=="new"

def test_storage_files_created(tmp_path):
    from research_logger import ResearchLogger
    r=ResearchLogger(tmp_path/"data")
    for name in ["trades.csv","markets.csv","resolutions.csv","pnl_1min.csv",
                 "decisions.jsonl","orderbooks.jsonl"]:
        assert (tmp_path/"data"/name).exists()

def test_bot_b_ignores_weak_signal():
    s=ConvergenceStrategy(min_score=.70)
    x=s.decide(150,.74,.26,.73,.25,[(time.time()-30,.74)],[(time.time()-30,.26)],0,1000,up_depth=50,down_depth=50)
    assert x is None


def test_bot_b_depth_cap():
    s=ConvergenceStrategy(max_order=10, max_depth_participation=.25)
    x=s.decide(220,.94,.06,.93,.05,[(time.time()-30,.80)],[(time.time()-30,.10)],0,1000,up_depth=4,down_depth=4)
    assert x is not None
    assert x.notional <= 4*.94*.25 + 1e-9


def test_bot_b_resolution_cutoff():
    s=ConvergenceStrategy()
    assert s.decide(240,.99,.01,.98,.00,[],[],0,1000,up_depth=100,down_depth=100) is None


def test_bot_b_incremental_regimes():
    s=ConvergenceStrategy(max_order=10, max_depth_participation=1.0)
    weak=s.decide(120,.76,.24,.75,.23,[(time.time()-30,.70)],[(time.time()-30,.24)],0,1000,up_depth=100,down_depth=100)
    strong=s.decide(200,.86,.14,.85,.13,[(time.time()-30,.76)],[(time.time()-30,.24)],0,1000,up_depth=100,down_depth=100)
    late=s.decide(220,.95,.05,.94,.04,[(time.time()-30,.82)],[(time.time()-30,.08)],0,1000,up_depth=100,down_depth=100)
    assert weak and strong and late
    assert weak.notional <= strong.notional <= late.notional


def test_bot_b_asset_exposure_cap():
    s=ConvergenceStrategy(max_asset_exposure=5, max_depth_participation=1.0)
    x=s.decide(220,.95,.05,.94,.04,[(time.time()-30,.82)],[(time.time()-30,.08)],0,1000,up_depth=100,down_depth=100,asset_exposure=5)
    assert x is None


def test_bot_b_asset_exposure_remaining_budget():
    s=ConvergenceStrategy(max_asset_exposure=5, max_order=10, max_depth_participation=1.0)
    x=s.decide(220,.95,.05,.94,.04,[(time.time()-30,.82)],[(time.time()-30,.08)],0,1000,up_depth=100,down_depth=100,asset_exposure=3)
    assert x is not None and x.notional <= 2 + 1e-9
