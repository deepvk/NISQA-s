import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from loguru import logger
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

from src.utils.math_helpers import calc_eval_metrics, calc_mapped, calc_mapping


class EarlyStopperDim(object):
    """
    Early stopping class for dimension model.

    Training is stopped if neither RMSE or Pearson's correlation
    is improving after "patience" epochs.
    """

    def __init__(self, patience):
        self.best_rmse = 1e10
        self.best_rmse_noi = 1e10
        self.best_rmse_col = 1e10
        self.best_rmse_dis = 1e10
        self.best_rmse_loud = 1e10

        self.best_r_p = -1e10
        self.best_r_p_noi = -1e10
        self.best_r_p_col = -1e10
        self.best_r_p_dis = -1e10
        self.best_r_p_loud = -1e10

        self.cnt = -1
        self.patience = patience
        self.best = False

    def step(self, r):
        self.best = False

        if r["r_p_mean_file"] > self.best_r_p:
            self.best_r_p = r["r_p_mean_file"]
            self.cnt = -1
        if r["r_p_mean_file_noi"] > self.best_r_p_noi:
            self.best_r_p_noi = r["r_p_mean_file_noi"]
            self.cnt = -1
        if r["r_p_mean_file_col"] > self.best_r_p_col:
            self.best_r_p_col = r["r_p_mean_file_col"]
            self.cnt = -1
        if r["r_p_mean_file_dis"] > self.best_r_p_dis:
            self.best_r_p_dis = r["r_p_mean_file_dis"]
            self.cnt = -1
        if r["r_p_mean_file_loud"] > self.best_r_p_loud:
            self.best_r_p_loud = r["r_p_mean_file_loud"]
            self.cnt = -1

        if r["rmse_map_mean_file"] < self.best_rmse:
            self.best_rmse = r["rmse_map_mean_file"]
            self.cnt = -1
            self.best = True
        if r["rmse_map_mean_file_noi"] < self.best_rmse_noi:
            self.best_rmse_noi = r["rmse_map_mean_file_noi"]
            self.cnt = -1
        if r["rmse_map_mean_file_col"] < self.best_rmse_col:
            self.best_rmse_col = r["rmse_map_mean_file_col"]
            self.cnt = -1
        if r["rmse_map_mean_file_dis"] < self.best_rmse_dis:
            self.best_rmse_dis = r["rmse_map_mean_file_dis"]
            self.cnt = -1
        if r["rmse_map_mean_file_loud"] < self.best_rmse_loud:
            self.best_rmse_loud = r["rmse_map_mean_file_loud"]
            self.cnt = -1

        self.cnt += 1

        stop_early = self.cnt >= self.patience
        return stop_early


class BiasLoss(object):
    """
    Bias loss class.

    Calculates loss while considering database bias.
    """

    def __init__(self, db, anchor_db=None, mapping="first_order", min_r=0.7, loss_weight=0.0, do_print=True):
        self.db = db
        self.mapping = mapping
        self.min_r = min_r
        self.anchor_db = anchor_db
        self.loss_weight = loss_weight
        self.verbose = do_print

        self.b = np.zeros((len(db), 4))
        self.b[:, 1] = 1
        self.update = False

        self.apply_bias_loss = True
        if (self.min_r is None) or (self.mapping is None):
            self.apply_bias_loss = False

    def get_loss(self, yb, yb_hat, idx):
        if not self.apply_bias_loss:
            return self._nan_mse(yb_hat, yb)

        b = torch.tensor(self.b, dtype=torch.float).to(yb_hat.device)
        b = b[idx, :]

        yb_hat_map = (
            b[:, 0] + b[:, 1] * yb_hat[:, 0] + b[:, 2] * yb_hat[:, 0] ** 2 + b[:, 3] * yb_hat[:, 0] ** 3
        ).view(-1, 1)

        loss_bias = self._nan_mse(yb_hat_map, yb)
        loss_normal = self._nan_mse(yb_hat, yb)

        return loss_bias + self.loss_weight * loss_normal

    def update_bias(self, y, y_hat):
        if not self.apply_bias_loss:
            return
        y_hat = y_hat.reshape(-1)
        y = y.reshape(-1)

        if not self.update:
            r = pearsonr(y[~np.isnan(y)], y_hat[~np.isnan(y)])[0]

            if self.verbose:
                logger.info("--> bias update: min_r {:0.2f}, r_p {:0.2f}".format(r, self.min_r))
            if r > self.min_r:
                self.update = True

        if self.update:
            if self.verbose:
                logger.info("--> bias updated")
            for db_name in self.db.unique():
                db_idx = (self.db == db_name).to_numpy().nonzero()
                y_hat_db = y_hat[db_idx]
                y_db = y[db_idx]

                if not np.isnan(y_db).any():
                    if self.mapping == "first_order":
                        b_db = self._calc_bias_first_order(y_hat_db, y_db)
                    else:
                        raise NotImplementedError
                    if not db_name == self.anchor_db:
                        self.b[db_idx, : len(b_db)] = b_db

    def _calc_bias_first_order(self, y_hat, y):
        A = np.vstack([np.ones(len(y_hat)), y_hat]).T
        btmp = np.linalg.lstsq(A, y, rcond=None)[0]
        b = np.zeros((4))
        b[0:2] = btmp
        return b

    def _nan_mse(self, y, y_hat):
        err = (y - y_hat).view(-1)
        idx_not_nan = ~torch.isnan(err)
        nan_err = err[idx_not_nan]
        return torch.mean(nan_err**2)


def eval_results(
    df, dcon=None, target_mos="mos", target_ci="mos_ci", pred="mos_pred", mapping=None, do_print=False, do_plot=False
):
    """
    Evaluates a trained model on given dataset.
    """
    # Loop through databases
    db_results_df = []
    df["y_hat_map"] = np.nan

    for db_name in df.db.astype("category").cat.categories:
        df_db = df.loc[df.db == db_name]
        if dcon is not None:
            dcon_db = dcon.loc[dcon.db == db_name]
        else:
            dcon_db = None

        # per file -----------------------------------------------------------
        y = df_db[target_mos].to_numpy()
        if np.isnan(y).any():
            r = {"r_p": np.nan, "r_s": np.nan, "rmse": np.nan, "r_p_map": np.nan, "r_s_map": np.nan, "rmse_map": np.nan}
        else:
            y_hat = df_db[pred].to_numpy()

            b, d = calc_mapping(df_db, mapping=mapping, target_mos=target_mos, target_ci=target_ci, pred=pred)
            y_hat_map = calc_mapped(y_hat, b)

            r = calc_eval_metrics(y, y_hat, y_hat_map=y_hat_map, d=d)
            r.pop("rmse_star_map")
        r = {f"{k}_file": v for k, v in r.items()}

        # per con ------------------------------------------------------------
        r_con = {
            "r_p": np.nan,
            "r_s": np.nan,
            "rmse": np.nan,
            "r_p_map": np.nan,
            "r_s_map": np.nan,
            "rmse_map": np.nan,
            "rmse_star_map": np.nan,
        }

        if (dcon_db is not None) and ("con" in df_db):
            y_con = dcon_db[target_mos].to_numpy()
            y_con_hat = df_db.groupby("con").mean().get(pred).to_numpy()

            if not np.isnan(y_con).any():
                if target_ci in dcon_db:
                    ci_con = dcon_db[target_ci].to_numpy()
                else:
                    ci_con = None

                b_con, d = calc_mapping(
                    df_db, dcon_db=dcon_db, mapping=mapping, target_mos=target_mos, target_ci=target_ci, pred=pred
                )

                df_db["y_hat_map"] = calc_mapped(y_hat, b_con)
                df["y_hat_map"].loc[df.db == db_name] = df_db["y_hat_map"]

                y_con_hat_map = df_db.groupby("con").mean().get("y_hat_map").to_numpy()
                r_con = calc_eval_metrics(y_con, y_con_hat, y_hat_map=y_con_hat_map, d=d, ci=ci_con)

        r_con = {f"{k}_con": v for k, v in r_con.items()}
        r = {**r, **r_con}

        # ---------------------------------------------------------------------
        db_results_df.append({"db": db_name, **r})
        # Plot  ------------------------------------------------------------------
        if do_plot and (not np.isnan(y).any()):
            xx = np.arange(0, 6, 0.01)
            yy = calc_mapped(xx, b)

            plt.figure(figsize=(3.0, 3.0), dpi=300)
            plt.clf()
            plt.plot(y_hat, y, "o", label="Original data", markersize=2)
            plt.plot([0, 5], [0, 5], "gray")
            plt.plot(xx, yy, "r", label="Fitted line")
            plt.axis([1, 5, 1, 5])
            plt.gca().set_aspect("equal", adjustable="box")
            plt.grid(True)
            plt.xticks(np.arange(1, 6))
            plt.yticks(np.arange(1, 6))
            plt.title(db_name + " per file")
            plt.ylabel("Subjective " + target_mos.upper())
            plt.xlabel("Predicted " + target_mos.upper())
            # plt.savefig('corr_diagram_fr_' + db_name + '.pdf', dpi=300, bbox_inches="tight")
            plt.show()

            if (dcon_db is not None) and ("con" in df_db):
                xx = np.arange(0, 6, 0.01)
                yy = calc_mapped(xx, b_con)

                plt.figure(figsize=(3.0, 3.0), dpi=300)
                plt.clf()
                plt.plot(y_con_hat, y_con, "o", label="Original data", markersize=3)
                plt.plot([0, 5], [0, 5], "gray")
                plt.plot(xx, yy, "r", label="Fitted line")
                plt.axis([1, 5, 1, 5])
                plt.gca().set_aspect("equal", adjustable="box")
                plt.grid(True)
                plt.xticks(np.arange(1, 6))
                plt.yticks(np.arange(1, 6))
                plt.title(db_name + " per con")
                plt.ylabel("Sub " + target_mos.upper())
                plt.xlabel("Pred " + target_mos.upper())
                # plt.savefig(db_name + '.pdf', dpi=300, bbox_inches="tight")
                plt.show()

                # Print ------------------------------------------------------------------
        if do_print and (not np.isnan(y).any()):
            if (dcon_db is not None) and ("con" in df_db):
                logger.info(
                    "%-30s r_p_file: %0.2f, rmse_map_file: %0.2f, r_p_con: %0.2f, rmse_map_con: %0.2f, rmse_star_map_con: %0.2f"
                    % (
                        db_name + ":",
                        r["r_p_file"],
                        r["rmse_map_file"],
                        r["r_p_con"],
                        r["rmse_map_con"],
                        r["rmse_star_map_con"],
                    )
                )
            else:
                logger.info(
                    "%-30s r_p_file: %0.2f, rmse_map_file: %0.2f" % (db_name + ":", r["r_p_file"], r["rmse_map_file"])
                )

    # Save individual database results in DataFrame
    db_results_df = pd.DataFrame(db_results_df)

    r_average = {}
    r_average["r_p_mean_file"] = db_results_df.r_p_file.mean()
    r_average["rmse_mean_file"] = db_results_df.rmse_file.mean()
    r_average["rmse_map_mean_file"] = db_results_df.rmse_map_file.mean()

    if dcon_db is not None:
        r_average["r_p_mean_con"] = db_results_df.r_p_con.mean()
        r_average["rmse_mean_con"] = db_results_df.rmse_con.mean()
        r_average["rmse_map_mean_con"] = db_results_df.rmse_map_con.mean()
        r_average["rmse_star_map_mean_con"] = db_results_df.rmse_star_map_con.mean()
    else:
        r_average["r_p_mean_con"] = np.nan
        r_average["rmse_mean_con"] = np.nan
        r_average["rmse_map_mean_con"] = np.nan
        r_average["rmse_star_map_mean_con"] = np.nan

    # Get overall per file results
    y = df[target_mos].to_numpy()
    y_hat = df[pred].to_numpy()

    r_total_file = calc_eval_metrics(y, y_hat)
    r_total_file = {"r_p_all": r_total_file["r_p"], "rmse_all": r_total_file["rmse"]}

    overall_results = {**r_total_file, **r_average}

    return db_results_df, overall_results


def get_lr(optimizer):
    """
    Get current learning rate from Pytorch optimizer.
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]


@torch.no_grad()
def predict_dim(model, ds, bs, dev, h0, c0, num_workers=0):
    """
    predict_dim: predicts MOS and dimensions of the given dataset with given
    model.
    """
    dl = DataLoader(ds, batch_size=bs, shuffle=False, drop_last=False, pin_memory=False, num_workers=num_workers)
    model.to(dev)
    model.eval()
    y_hat_tensors = []
    for xb, yb, (idx, n_wins) in dl:
        y_hat, _, _ = model(xb.to(dev), n_wins.to(dev), h0, c0)
        y_hat_tensors.append(torch.stack([y_hat, yb.to(dev)]))

    yy = torch.cat(y_hat_tensors, dim=1)

    y_hat = yy[0, :, :].cpu().numpy()
    y = yy[1, :, :].cpu().numpy()

    ds.df["mos_pred"] = y_hat[:, 0].reshape(-1, 1)
    ds.df["noi_pred"] = y_hat[:, 1].reshape(-1, 1)
    ds.df["dis_pred"] = y_hat[:, 2].reshape(-1, 1)
    ds.df["col_pred"] = y_hat[:, 3].reshape(-1, 1)
    ds.df["loud_pred"] = y_hat[:, 4].reshape(-1, 1)

    return y_hat, y


def yamlparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", default="config/nisqa_s.yaml", type=str, help="YAML file with config")
    args = parser.parse_args()
    args = vars(args)
    return args
