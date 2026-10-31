"""Small CPU check of chronological preprocessing, federated updates and metrics."""
import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from bpfbi import cli
from bpfbi.model import BuildingModel, objective


def invoke(*args):
    with patch.object(sys,'argv',['bpfbi',*map(str,args)]): cli.main()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='outputs/functional-check')
    root=Path(parser.parse_args().output);root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.manual_seed(11)
    hours=120;rows=[]
    for client in range(2):
        for i,date in enumerate(pd.date_range('2020-01-01',periods=hours,freq='h')):
            rows.append(dict(client=client,timestamp=date,load=10+client+np.sin(i/5),temperature=15+i/100))
    pd.DataFrame(rows).to_csv(root/'input.csv',index=False)
    data=root/'data'
    invoke('prepare','--csv',root/'input.csv','--output',data,'--window',4,'--weather','temperature','--anomaly-fraction','.15')
    with np.load(data/'client_000.npz') as archive: values={k:archive[k] for k in archive.files}
    np.testing.assert_allclose(values['mean'],np.mean([r['load'] for r in rows[:72]]),rtol=1e-6)
    dataset=cli.Windows(values,'train');x,y,a=next(iter(torch.utils.data.DataLoader(dataset,batch_size=4)))
    model=BuildingModel(x.shape[-1],4);optimizer=torch.optim.Adam(model.parameters(),lr=.01)
    before=model.encoder.input.weight.detach().clone();loss=objective(model(x),y,a)
    assert torch.isfinite(loss);loss.backward()
    assert model.encoder.input.weight.grad.abs().sum()>0 and model.head.weight.grad.abs().sum()>0
    optimizer.step();assert not torch.equal(before,model.encoder.input.weight)
    for method in ('full','fedavg','fedprox','fedper','local','no_debt','no_calibration','centralized','scaffold','ditto','acsp_fl'):
        extra=['--match-run',root/'full/history.json'] if method in ('local','centralized') else []
        if method=='scaffold':extra+=['--optimizer','sgd']
        invoke('run','--data',data,'--output',root/method,'--method',method,'--rounds',1,
            '--clients-per-round',2,'--availability-probability',1,'--local-epochs',2,'--width',4,'--batch-size',64,'--bootstrap',20,*extra)
        metrics=json.loads((root/method/'metrics.json').read_text())
        assert np.isfinite(metrics['mean_nrmse'])
        assert sum(metrics['participation'])==2
    from bpfbi.selection import acsp_candidates
    np.testing.assert_equal(acsp_candidates([1,3,2],1,.5),[1])
    a=json.loads((root/'full/metrics.json').read_text());b=json.loads((root/'local/metrics.json').read_text())
    assert a['optimizer_steps']==b['optimizer_steps']
    for method in ('centralized','scaffold','ditto'):
        c=json.loads((root/method/'metrics.json').read_text());assert c['optimizer_steps']==a['optimizer_steps']
        invoke('evaluate','--data',data,'--model',root/method/'model_000.pt','--output',root/(method+'_frozen'))
        np.testing.assert_allclose(np.load(root/method/'prediction_000.npz')['prediction'],np.load(root/(method+'_frozen')/'predictions.npz')['prediction'],rtol=1e-5,atol=1e-6)
    a0=torch.load(root/'centralized/model_000.pt',weights_only=True)['state'];a1=torch.load(root/'centralized/model_001.pt',weights_only=True)['state']
    assert all(torch.equal(a0[k],a1[k]) for k in a0)
    from bpfbi.data import interpolate_short
    np.testing.assert_allclose(interpolate_short([0,np.nan,np.nan,3]),[0,1,2,3])
    assert np.isnan(interpolate_short([0,np.nan,np.nan,np.nan,np.nan,5])[1:5]).all()
    assert values['train_anomaly'].sum()==round(.15*len(values['train_y']))
    invoke('evaluate','--data',data,'--model',root/'full/model_000.pt','--output',root/'frozen')
    a=np.load(root/'full/prediction_000.npz');b=np.load(root/'frozen/predictions.npz')
    np.testing.assert_allclose(a['prediction'],b['prediction'],rtol=1e-5,atol=1e-6)
    # Two perfect medians; intervals [0,2] and [1,3] imply coverage 1, width 2, Winkler 2.
    small=dict(test_y=np.array([1.,2.]),test_observed=np.array([1.,2.]),
        test_anomaly=np.array([0.,0.]),test_time=np.arange(2)*3_600_000_000_000,scale=np.array(1.),mean=np.array(0.))
    metrics,_=cli.evaluate(np.array([[1.,0.,2.,-10.],[2.,1.,3.,-10.]]),small,0.)
    assert metrics['nrmse']==0 and metrics['picp']==1 and metrics['mpiw']==2 and metrics['winkler']==2
    invoke('compare','--first',root/'full','--second',root/'fedavg','--output',root/'comparison.json')
    invoke('plot','--metrics',root/'full/metrics.json',root/'fedavg/metrics.json','--output',root/'figures')
    print(json.dumps(dict(status='passed',methods=11,rounds=1,checkpoint_equal=True,metric_example=True)))


if __name__=='__main__':main()
