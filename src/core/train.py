import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.core.datasets import loadDatasetsCSV
from src.core.model_torch import loadModel
from src.utils.file_utils import makeRunnameAndWriteYAML, save_results
from src.utils.train_utils import BiasLoss, EarlyStopperDim, eval_results, get_lr, predict_dim


def train_dim(args):
    """
    Trains multidimensional speech quality model.
    """

    # Initialize  -------------------------------------------------------------
    ds_train, ds_val = loadDatasetsCSV(args)
    model, dev, model_args = loadModel(args)
    runname = makeRunnameAndWriteYAML(args)
    results = {
        "runname": runname,
        "epoch": "0",
        "filename": runname + ".tar",
        "loss": "0",
        "ep_runtime": "0",
        **args,
    }
    for key in results:
        results[key] = str(results[key])
    results_hist = pd.DataFrame(results, index=[0])
    if args["tr_parallel"]:
        model = nn.DataParallel(model)
    model.to(dev)

    # Optimizer  -------------------------------------------------------------
    opt = optim.Adam(model.parameters(), lr=args["tr_lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, "min", verbose=True, threshold=0.003, patience=args["tr_lr_patience"])
    early_stop = EarlyStopperDim(args["tr_early_stop"])

    biasLosses = []

    for i in range(5):
        biasLosses.append(
            BiasLoss(
                ds_train.df.db,
                anchor_db=args["tr_bias_anchor_db"],
                mapping=args["tr_bias_mapping"],
                min_r=args["tr_bias_min_r"],
                do_print=(args["tr_verbose"] > 0),
            )
        )

    # Dataloader    -----------------------------------------------------------
    dl_train = DataLoader(
        ds_train,
        batch_size=args["tr_bs"],
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=args["tr_num_workers"],
    )

    # Start training loop   ---------------------------------------------------
    logger.info("--> start training")
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(args["tr_epochs"]):
        tic_epoch = time.time()
        batch_cnt = 0
        loss = 0.0
        y_mos = ds_train.df["mos"].to_numpy().reshape(-1, 1)
        y_noi = ds_train.df["noi"].to_numpy().reshape(-1, 1)
        y_dis = ds_train.df["dis"].to_numpy().reshape(-1, 1)
        y_col = ds_train.df["col"].to_numpy().reshape(-1, 1)
        y_loud = ds_train.df["loud"].to_numpy().reshape(-1, 1)
        y_train = np.concatenate((y_mos, y_noi, y_dis, y_col, y_loud), axis=1)
        y_train_hat = np.zeros((len(ds_train), 5))

        model.train()

        # Progress bar
        if args["tr_verbose"] == 2:
            pbar = tqdm(
                iterable=batch_cnt,
                total=len(dl_train),
                ascii=">—",
                bar_format="{bar} {percentage:3.0f}%, {n_fmt}/{total_fmt}, {elapsed}<{remaining}{postfix}",
            )

        directions = 2 if args["td_lstm_bidirectional"] else 1
        h0 = torch.zeros(args["td_lstm_num_layers"] * directions * torch.cuda.device_count(), args["tr_bs"], args["td_lstm_h"])
        c0 = torch.zeros(args["td_lstm_num_layers"] * directions * torch.cuda.device_count(), args["tr_bs"], args["td_lstm_h"])
        for xb_spec, yb_mos, (idx, n_wins) in dl_train:
            # Estimate batch ---------------------------------------------------
            xb_spec = xb_spec.to(dev)
            yb_mos = yb_mos.to(dev)
            n_wins = n_wins.to(dev)

            # Forward pass ----------------------------------------------------
            yb_mos_hat, _, _ = model(xb_spec, n_wins, h0, c0)
            y_train_hat[idx, :] = yb_mos_hat.detach().cpu().numpy()

            # Loss ------------------------------------------------------------
            lossb = torch.tensor(0.0).to(dev)
            for i, biasLoss in enumerate(biasLosses):
                lossb += biasLoss.get_loss(yb_mos[:, i].view(-1, 1), yb_mos_hat[:, i].view(-1, 1), idx)
            # Backprop  -------------------------------------------------------
            lossb.backward()
            opt.step()
            opt.zero_grad()

            # Update total loss -----------------------------------------------
            loss += lossb.item()
            batch_cnt += 1

            if args["tr_verbose"] == 2:
                pbar.set_postfix(loss=lossb.item())
                pbar.update()

        if args["tr_verbose"] == 2:
            pbar.close()

        loss = loss / batch_cnt

        for i, biasLoss in enumerate(biasLosses):
            biasLosses[i].update_bias(y_train[:, i].reshape(-1, 1), y_train_hat[:, i].reshape(-1, 1))

        # Evaluate   -----------------------------------------------------------
        ds_train.df["mos_pred"] = y_train_hat[:, 0].reshape(-1, 1)
        ds_train.df["noi_pred"] = y_train_hat[:, 1].reshape(-1, 1)
        ds_train.df["dis_pred"] = y_train_hat[:, 2].reshape(-1, 1)
        ds_train.df["col_pred"] = y_train_hat[:, 3].reshape(-1, 1)
        ds_train.df["loud_pred"] = y_train_hat[:, 4].reshape(-1, 1)

        if args["tr_verbose"] > 0:
            logger.info("\n<---- Training ---->")
            logger.info("--> MOS:")
        db_results_train_mos, r_train_mos = eval_results(
            ds_train.df,
            dcon=ds_train.df_con,
            target_mos="mos",
            target_ci="mos_ci",
            pred="mos_pred",
            mapping="first_order",
            do_print=(args["tr_verbose"] > 0),
        )

        if args["tr_verbose"] > 0:
            logger.info("--> NOI:")
        db_results_train_noi, r_train_noi = eval_results(
            ds_train.df,
            dcon=ds_train.df_con,
            target_mos="noi",
            target_ci="noi_ci",
            pred="noi_pred",
            mapping="first_order",
            do_print=(args["tr_verbose"] > 0),
        )

        if args["tr_verbose"] > 0:
            logger.info("--> DIS:")
        db_results_train_dis, r_train_dis = eval_results(
            ds_train.df,
            dcon=ds_train.df_con,
            target_mos="dis",
            target_ci="dis_ci",
            pred="dis_pred",
            mapping="first_order",
            do_print=(args["tr_verbose"] > 0),
        )

        if args["tr_verbose"] > 0:
            logger.info("--> COL:")
        db_results_train_col, r_train_col = eval_results(
            ds_train.df,
            dcon=ds_train.df_con,
            target_mos="col",
            target_ci="col_ci",
            pred="col_pred",
            mapping="first_order",
            do_print=(args["tr_verbose"] > 0),
        )

        if args["tr_verbose"] > 0:
            logger.info("--> LOUD:")
        db_results_train_loud, r_train_loud = eval_results(
            ds_train.df,
            dcon=ds_train.df_con,
            target_mos="loud",
            target_ci="loud_ci",
            pred="loud_pred",
            mapping="first_order",
            do_print=(args["tr_verbose"] > 0),
        )

        predict_dim(model, ds_val, args["tr_bs_val"], dev, h0, c0, num_workers=args["tr_num_workers"])

        if args["tr_verbose"] > 0:
            logger.info("<---- Validation ---->")
            logger.info("--> MOS:")
        db_results_val_mos, r_val_mos = eval_results(
            ds_val.df,
            dcon=ds_val.df_con,
            target_mos="mos",
            target_ci="mos_ci",
            pred="mos_pred",
            mapping="first_order",
            do_print=(args["tr_verbose"] > 0),
        )

        if args["tr_verbose"] > 0:
            logger.info("--> NOI:")
        db_results_val_noi, r_val_noi = eval_results(
            ds_val.df,
            dcon=ds_val.df_con,
            target_mos="noi",
            target_ci="noi_ci",
            pred="noi_pred",
            mapping="first_order",
            do_print=(args["tr_verbose"] > 0),
        )
        r_val_noi = {k + "_noi": v for k, v in r_val_noi.items()}

        if args["tr_verbose"] > 0:
            logger.info("--> DIS:")
        db_results_val_dis, r_val_dis = eval_results(
            ds_val.df,
            dcon=ds_val.df_con,
            target_mos="dis",
            target_ci="dis_ci",
            pred="dis_pred",
            mapping="first_order",
            do_print=(args["tr_verbose"] > 0),
        )
        r_val_dis = {k + "_dis": v for k, v in r_val_dis.items()}

        if args["tr_verbose"] > 0:
            logger.info("--> COL:")
        db_results_val_col, r_val_col = eval_results(
            ds_val.df,
            dcon=ds_val.df_con,
            target_mos="col",
            target_ci="col_ci",
            pred="col_pred",
            mapping="first_order",
            do_print=(args["tr_verbose"] > 0),
        )
        r_val_col = {k + "_col": v for k, v in r_val_col.items()}

        if args["tr_verbose"] > 0:
            logger.info("--> LOUD:")
        db_results_val_loud, r_val_loud = eval_results(
            ds_val.df,
            dcon=ds_val.df_con,
            target_mos="loud",
            target_ci="loud_ci",
            pred="loud_pred",
            mapping="first_order",
            do_print=(args["tr_verbose"] > 0),
        )
        r_val_loud = {k + "_loud": v for k, v in r_val_loud.items()}

        r = {
            "train_r_p_mean_file": r_train_mos["r_p_mean_file"],
            "train_rmse_map_mean_file": r_train_mos["rmse_map_mean_file"],
            "train_r_p_mean_file_noi": r_train_noi["r_p_mean_file"],
            "train_rmse_map_mean_file_noi": r_train_noi["rmse_map_mean_file"],
            "train_r_p_mean_file_dis": r_train_dis["r_p_mean_file"],
            "train_rmse_map_mean_file_dis": r_train_dis["rmse_map_mean_file"],
            "train_r_p_mean_file_col": r_train_col["r_p_mean_file"],
            "train_rmse_map_mean_file_col": r_train_col["rmse_map_mean_file"],
            "train_r_p_mean_file_loud": r_train_loud["r_p_mean_file"],
            "train_rmse_map_mean_file_loud": r_train_loud["rmse_map_mean_file"],
            **r_val_mos,
            **r_val_noi,
            **r_val_dis,
            **r_val_col,
            **r_val_loud,
        }

        db_results = {
            "db_results_val_mos": db_results_val_mos,
            "db_results_val_noi": db_results_val_noi,
            "db_results_val_dis": db_results_val_dis,
            "db_results_val_col": db_results_val_col,
            "db_results_val_loud": db_results_val_loud,
        }

        # Scheduler update    ---------------------------------------------
        scheduler.step(loss)
        earl_stp = early_stop.step(r)

        # Print    --------------------------------------------------------
        ep_runtime = time.time() - tic_epoch

        r_dim_mos_mean = (
            1
            / 5
            * (
                r["r_p_mean_file"]
                + r["r_p_mean_file_noi"]
                + r["r_p_mean_file_col"]
                + r["r_p_mean_file_dis"]
                + r["r_p_mean_file_loud"]
            )
        )
        logger.info(
            "ep {} sec {:0.0f} es {} lr {:0.0e} loss {:0.4f} // "
            "r_p_tr {:0.2f} rmse_map_tr {:0.2f} // r_dim_mos_mean {:0.2f}, r_p {:0.2f} rmse_map {:0.2f} // "
            "best_r_p {:0.2f} best_rmse_map {:0.2f},".format(
                epoch + 1,
                ep_runtime,
                early_stop.cnt,
                get_lr(opt),
                loss,
                r["train_r_p_mean_file"],
                r["train_rmse_map_mean_file"],
                r_dim_mos_mean,
                r["r_p_mean_file"],
                r["rmse_map_mean_file"],
                early_stop.best_r_p,
                early_stop.best_rmse,
            )
        )

        # Save results and model  -----------------------------------------
        save_results(
            model, args, model_args, opt, epoch, loss, ep_runtime, r, db_results, early_stop.best, runname, results_hist
        )

        # Early stopping    -----------------------------------------------
        if earl_stp:
            logger.info(
                "--> Early stopping. best_r_p {:0.2f} best_rmse {:0.2f}".format(early_stop.best_r_p, early_stop.best_rmse)
            )
            return

            # Training done --------------------------------------------------------
    logger.info("--> Training done. best_r_p {:0.2f} best_rmse {:0.2f}".format(early_stop.best_r_p, early_stop.best_rmse))
    return
