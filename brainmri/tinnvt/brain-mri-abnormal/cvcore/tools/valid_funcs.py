import torch
import numpy as np
from sklearn.metrics import accuracy_score

def MultiHeadAccuracyScore(predictions, targets):
    scores = []
    for p, t in zip(predictions, targets): # per batch
        batch_score = []
        for ci in range(len(p)): # per head
            pc = torch.argmax(p[ci], dim=1)
            tc = t[:,ci]
            batch_score.append(accuracy_score(tc.cpu().numpy(), pc.cpu().numpy()))
        scores.append(batch_score)
    scores = np.mean(scores, 0)

    return scores

def MultiHeadCrossEntropyLoss(predictions, targets):
    loss_func = torch.nn.CrossEntropyLoss()
    loss_val = 0
    for p, t in zip(predictions, targets): # per batch
        for ci in range(len(p)): # per head
            pc = p[ci]
            tc = t[:,ci]
            loss_val += loss_func(pc, tc) / len(predictions)

    return loss_val