import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
from cvcore.utils import save_checkpoint
<<<<<<< HEAD
from sklearn.metrics import f1_score, recall_score, precision_score
=======
from sklearn.metrics import f1_score
>>>>>>> master


def valid_model(_print, cfg, model, valid_loader,
                loss_function, score_function, epoch,
                best_metric=None, checkpoint=False):
    # switch to evaluate mode
    model.eval()

    preds = []
    targets = []
    embeddings = []
    tbar = tqdm(valid_loader)

    with torch.no_grad():
        for i, (image, lb) in enumerate(tbar):
            image = image.cuda()
            lb = lb.cuda().unsqueeze(-1)
            w_output, w_embeddings = model(image, output_embeddings=True)

            preds.append(w_output.cpu())
            embeddings.append(w_embeddings.cpu())
            targets.append(lb.cpu())

    preds, targets, embeddings = torch.cat(preds, 0).float(), torch.cat(targets, 0), torch.cat(embeddings, 0).float()

    if cfg.INFER.SAVE_NAME:
        np.save(f"{cfg.DIRS.OUTPUTS}/{cfg.INFER.SAVE_NAME}", embeddings.numpy())

    # record
    val_loss = loss_function(preds, targets)
    score = score_function(y_true=targets, y_score=torch.sigmoid(preds))

    _print(f"VAL LOSS: {val_loss:.5f}, SCORE: {score:.4f}")
    # checkpoint
    if checkpoint:
        is_best = score > best_metric
        best_metric = max(score, best_metric)
        save_dict = {"epoch": epoch + 1,
                     "arch": cfg.NAME,
                     "state_dict": model.state_dict(),
                     "best_metric": best_metric}
        save_filename = f"{cfg.NAME}.pth"
        if is_best: # only save best checkpoint, no need resume
            save_checkpoint(save_dict, is_best,
                            root=cfg.DIRS.WEIGHTS, filename=save_filename)
            print("$$$ score improved, new checkpoint saved $$$")
        return val_loss, best_metric
<<<<<<< HEAD
    else:
        best_fscore = 0.
        best_threshold = 0.
        for threshold_t in range(1, 99):
            threshold = threshold_t / 100
            preds_s = torch.sigmoid(preds)
            preds_b = preds_s > threshold
            fscore = f1_score(y_true=targets, y_pred=preds_b)
            if fscore > best_fscore:
                best_fscore = fscore
                best_threshold = threshold
                print(f"f1 score {fscore:.4f} @ threshold {threshold:.2f}")
        preds_b = preds_s > best_threshold
        rscore = recall_score(y_true=targets, y_pred=preds_b)
        pscore = precision_score(y_true=targets, y_pred=preds_b)
        print(f"recall score {rscore:.4f} @ best f1")
        print(f"precision score {pscore:.4f} @ best f1")
        preds_s = preds_s.numpy()
        np.save(os.path.join(cfg.DIRS.OUTPUTS, "probs"), preds_s)
=======
    # not checkpointing
    else:
        best_fscore = 0.
        pred_sigmoid = torch.sigmoid(preds)
        for threshold_t in range(1, 99):
            threshold = threshold_t / 100.
            pred_binary = pred_sigmoid > threshold
            fscore = f1_score(y_true=targets, y_pred=pred_binary)
            if fscore > best_fscore:
                best_fscore = fscore
                print(f"new best f1 score: {fscore:.4f} @ threshold {threshold:.2f}")

>>>>>>> master
