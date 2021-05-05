import os
import pandas as pd
import torch.nn as nn
from torch.cuda.amp import GradScaler

from sklearn.metrics import roc_auc_score

from cvcore.config import get_cfg_defaults
from cvcore.data import make_image_label_dataloader
from cvcore.model import build_model
from cvcore.solver import make_optimizer, build_scheduler
from cvcore.utils import setup_determinism, setup_logger, load_checkpoint
from cvcore.tools import parse_args, train_loop, valid_model, copy_model

scaler = GradScaler()

def main(args, cfg):    
    # Set logger
    logger = setup_logger(
        args.mode,
        cfg.DIRS.LOGS,
        0,
        filename=f"{cfg.NAME}.txt")

    # Define model
    model = build_model(cfg)
    if cfg.SOLVER.SWA.ENABLED:
        model_swa = build_model(cfg)
    else:
        model_swa = None
    optimizer = make_optimizer(cfg, model)

    # Define loss
    if cfg.LOSS.NAME == "ce":
        train_criterion = nn.BCEWithLogitsLoss()
        valid_criterion = nn.BCEWithLogitsLoss()
        

    model = model.cuda()
    model = nn.DataParallel(model)
    if cfg.SOLVER.SWA.ENABLED:
        model_swa = model_swa.cuda()
        model_swa = nn.DataParallel(model_swa)
    train_criterion = train_criterion.cuda()

    # Load checkpoint
    model, start_epoch, best_metric = load_checkpoint(args, logger.info, model)

    # Load and split data
    if args.mode in ("train", "valid"):
        valid_df = pd.read_csv(cfg.DATA.CSV.VALIDATION)
        valid_loader = make_image_label_dataloader(
            cfg, "valid", valid_df["imageUid"].values, valid_df["abnormal"].values)
        if args.mode == "train":
            train_df = pd.read_csv(cfg.DATA.CSV.TRAIN)
            train_loader = make_image_label_dataloader(
                cfg, "train", train_df["imageUid"].values, train_df["abnormal"].values)
    elif args.mode == "test":
        #TODO: Write test dataloader.
        pass

    # Build training scheduler
    if args.mode == "train":
        scheduler = build_scheduler(cfg, len(train_loader))


    # Run script
    if args.mode == "train":
        for epoch in range(start_epoch, cfg.TRAIN.EPOCHES[-1]):
            if cfg.SOLVER.SWA.ENABLED and epoch == cfg.SOLVER.SWA.START_EPOCH:
                copy_model(model_swa, model)
            train_loop(logger.info, cfg, model,
                       model_swa if epoch >= cfg.SOLVER.SWA.START_EPOCH else None,
                       train_loader, train_criterion, optimizer,
                       scheduler, epoch, scaler)
            _, best_metric = valid_model(logger.info, cfg,
                                      model_swa if cfg.SOLVER.SWA.ENABLED and \
                                        epoch >= cfg.SOLVER.SWA.START_EPOCH else model,
                                      valid_loader, valid_criterion,
                                      roc_auc_score, epoch, best_metric, True)
    elif args.mode == "valid":
        valid_model(logger.info, cfg, model,
                    valid_loader, valid_criterion,
                    roc_auc_score, start_epoch)


if __name__ == "__main__":
    args = parse_args()
    cfg = get_cfg_defaults()

    if args.config != "":
        cfg.merge_from_file(args.config)
    if args.opts != "":
        cfg.merge_from_list(args.opts)

    # make dirs
    for _dir in ["WEIGHTS", "OUTPUTS", "LOGS"]:
        if not os.path.isdir(cfg.DIRS[_dir]):
            os.mkdir(cfg.DIRS[_dir])
    # seed, run
    setup_determinism(cfg.SYSTEM.SEED)
    main(args, cfg)