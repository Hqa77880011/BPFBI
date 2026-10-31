"""Fit/reload classical baselines after verify.py prepares its small two-client fixture."""
import json,sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
from bpfbi.classical import main,Classical

root=Path('outputs/functional-check')
for method in ('seasonal_naive','xgboost'):
    output=root/method
    arguments=['classical','--data',str(root/'data'),'--method',method,'--output',str(output),'--period','2','--trees','3','--bootstrap','20']
    with patch.object(sys,'argv',arguments):main()
    arguments[arguments.index('--output')+1]=str(root/(method+'-frozen'))
    arguments+=['--checkpoint-dir',str(output)]
    with patch.object(sys,'argv',arguments):main()
    for client in (0,1):
        a=np.load(output/f'prediction_{client:03d}.npz');b=np.load(root/(method+'-frozen')/f'prediction_{client:03d}.npz')
        np.testing.assert_allclose(a['prediction'],b['prediction'],rtol=1e-6,atol=1e-7)
    metrics=json.loads((output/'metrics.json').read_text());assert np.isfinite(metrics['mean_nrmse'])
x=np.tile(np.arange(4,dtype=float)[None,:,None],(2,1,1))
m=Classical('seasonal_naive',period=2).fit(x,np.array([2.,2.]),np.zeros(2),horizon=1)
np.testing.assert_equal(m.predict(x,1)[:,0],[2.,2.])
print('SeasonalNaive and actual XGBoost fit/frozen reload/metrics and seasonal hand example passed')
