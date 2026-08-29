import json,time,os
from pathlib import Path

class PaperLedger:
    def __init__(self,path,initial_cash=1000):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        self.cash=float(initial_cash); self.realized=0.; self.positions={}; self.trades=[]
        self.start_equity=float(initial_cash); self.peak_equity=float(initial_cash); self.last_equity=float(initial_cash)
        self._load()
    def _load(self):
        if not self.path.exists(): return
        try:
            d=json.loads(self.path.read_text()); self.cash=float(d['cash']); self.realized=float(d.get('realized',0)); self.positions=d.get('positions',{}); self.trades=d.get('trades',[]); self.start_equity=float(d.get('start_equity',self.cash)); self.peak_equity=float(d.get('peak_equity',self.start_equity)); self.last_equity=float(d.get('last_equity',self.start_equity))
        except Exception as e: raise RuntimeError(f'paper state corrupt/unreadable: {e}')
    def save(self):
        tmp=self.path.with_suffix('.tmp'); tmp.write_text(json.dumps({'cash':self.cash,'realized':self.realized,'positions':self.positions,'trades':self.trades[-10000:],'start_equity':self.start_equity,'peak_equity':self.peak_equity,'last_equity':self.last_equity},indent=2)); os.replace(tmp,self.path)
    def total_open_cost(self): return sum(float(p['cost']) for p in self.positions.values())
    def exposure(self,condition): return sum(float(p['cost']) for p in self.positions.values() if p.get('condition')==condition)
    def buy(self,condition,token,market,side,price,notional,ts,meta=None):
        price=float(price); notional=min(float(notional),self.cash)
        if price<=0 or price>=1: raise ValueError('invalid paper execution price')
        if notional<=0: raise ValueError('insufficient paper cash')
        shares=notional/price; key=f'{condition}:{token}'; p=self.positions.get(key,{'condition':condition,'token':token,'market':market,'side':side,'shares':0.,'cost':0.,'avg':0.})
        p['shares']+=shares; p['cost']+=notional; p['avg']=p['cost']/p['shares']; p['last_price']=price; p.update(meta or {}); self.positions[key]=p; self.cash-=notional
        t={'ts':ts,'action':'BUY','condition':condition,'token':token,'market':market,'side':side,'price':price,'shares':shares,'notional':notional,'status':'OPEN'}; t.update(meta or {}); self.trades.append(t); self.save(); return t
    def mark(self,books):
        value=unreal=0.; marked=0
        for p in self.positions.values():
            bid=books.get(p['token']); px=bid if bid is not None else p['avg']; v=p['shares']*px; value+=v; unreal+=v-p['cost']; marked+=1 if bid is not None else 0
        equity=self.cash+value; self.peak_equity=max(self.peak_equity,equity); self.last_equity=equity; self.save()
        return {'cash':self.cash,'open_cost':self.total_open_cost(),'market_value':value,'unrealized':unreal,'realized':self.realized,'equity':equity,'pnl':equity-self.start_equity,'drawdown':equity-self.peak_equity,'marked':marked}
    def settle(self,condition,winner_token):
        closed=[]
        for key,p in list(self.positions.items()):
            if p['condition']!=condition: continue
            payout=p['shares'] if p['token']==winner_token else 0.
            pnl=payout-p['cost']
            settlement_per_share=1.0 if p['token']==winner_token else 0.0
            self.cash+=payout; self.realized+=pnl
            self.trades.append({'ts':time.time(),'action':'SETTLE','condition':condition,'token':p['token'],'side':p['side'],'price':p['avg'],'shares':p['shares'],'notional':p['cost'],'payout':payout,'pnl':pnl,'settlement_per_share':settlement_per_share,'status':'WIN' if pnl>=0 else 'LOSS'})
            closed.append({'key':key,'pnl':pnl,'settlement_per_share':settlement_per_share,'shares':p['shares'],'cost':p['cost'],'payout':payout,'side':p['side']})
            del self.positions[key]
        if closed:self.save()
        return closed
