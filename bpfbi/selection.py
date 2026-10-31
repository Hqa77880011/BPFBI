"""ACSP-FL regression adaptation of paper equations (4)--(7).

Source: https://arxiv.org/abs/2411.17833; official Server/server.py, MIT.
"""
import math
import numpy as np


def acsp_candidates(validation_mse,round_index,decay):
    """Below-mean performance filtering, ordering and exponential count decay."""
    if not 0<=decay<1:raise ValueError('ACSP decay must be in [0,1)')
    error=np.asarray(validation_mse,float)
    ordered=np.argsort(-error,kind='stable')
    eligible=ordered if round_index==0 else ordered[error[ordered]>=error.mean()]
    return eligible[:math.ceil(len(eligible)*(1-decay)**round_index)]
