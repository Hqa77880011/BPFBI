"""EACS-FL single-global-model protocol, primary equations (3)--(10)."""
import numpy as np


class EACS:
    def __init__(self,path,clients,rounds,k,weight=1.,maximum_delay=3600.):
        if not np.isfinite(weight) or weight<0 or not np.isfinite(maximum_delay) or maximum_delay<=0:raise ValueError('Nonnegative finite V and positive finite deadline required')
        with np.load(path,allow_pickle=False) as archive:self.trace={name:archive[name] for name in archive.files}
        energy,delay,budget=(self.trace[name] for name in ('energy','delay','budget'))
        if energy.shape!=delay.shape or energy.shape!=(rounds+1,clients) or budget.shape!=(clients,):raise ValueError('Resource trace requires energy/delay[rounds+1,clients], budget[clients]')
        if not all(np.isfinite(x).all() for x in (energy,delay,budget)) or (energy<0).any() or (delay<0).any() or (budget<=0).any():raise ValueError('Nonnegative finite observed costs and positive energy budgets required')
        self.energy=energy[0].copy();self.delay=np.minimum(delay[0],maximum_delay);self.pulls=np.ones(clients)
        self.budget=budget.copy();self.spent=np.zeros(clients);self.queue=np.zeros(clients)
        self.rounds=rounds;self.k=k;self.weight=weight;self.maximum_delay=maximum_delay;self.failures=0
    def ranking(self,round_index,available):
        # Initialization observations are preflight measurements; only earlier returned costs drive this score.
        confidence=np.sqrt((self.k+1)*np.log(max(1,round_index+1))/self.pulls)
        quality=self.weight*self.delay+self.queue*self.energy-confidence
        eligible=np.asarray(available,bool)&(self.budget-self.spent>=self.energy)
        return np.array(sorted(np.flatnonzero(eligible),key=lambda i:(quality[i],i)),int)
    def observe(self,round_index,selected):
        current=round_index+1;cost=np.zeros_like(self.energy)
        for client in selected:
            cost[client]=self.trace['energy'][current,client];delay=min(self.trace['delay'][current,client],self.maximum_delay)
            self.pulls[client]+=1
            self.energy[client]+=(cost[client]-self.energy[client])/self.pulls[client]
            self.delay[client]+=(delay-self.delay[client])/self.pulls[client]
        self.spent+=cost;self.queue=np.maximum(self.queue+cost-self.budget/self.rounds,0)
        on_time=np.array([self.trace['delay'][current,i]<=self.maximum_delay for i in selected],bool)
        self.failures+=int((~on_time).sum())
        return on_time
    def report(self):
        return dict(energy_spent=self.spent.tolist(),energy_budget=self.budget.tolist(),virtual_energy_queue=self.queue.tolist(),energy_budget_violations=np.flatnonzero(self.spent>self.budget).tolist(),deadline_failures=self.failures)
