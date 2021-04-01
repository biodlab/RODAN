import torch
import numpy as np

#
# from Bonito 
# https://github.com/nanoporetech/bonito
# Oxord Nanopore Technologies, Ltd. Public License Version 1.0
# See LICENSE.txt in the bonito repository
#

def ctc_label_smoothing_loss(log_probs, targets, lengths, weights):
    T, N, C = log_probs.shape
    log_probs_lengths = torch.full(size=(N, ), fill_value=T, dtype=torch.int64)
    loss = torch.nn.functional.ctc_loss(log_probs.to(torch.float32), targets, log_probs_lengths, lengths, reduction='mean', zero_infinity=True)
    label_smoothing_loss = -((log_probs * weights.to(log_probs.device)).mean())
    return {'loss': loss + label_smoothing_loss, 'ctc_loss': loss, 'label_smooth_loss': label_smoothing_loss}

def med_mad(x, factor=1.4826):
    med = np.median(x)
    mad = np.median(np.absolute(x - med)) * factor
    return med, mad
