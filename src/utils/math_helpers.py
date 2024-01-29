import numpy as np
from loguru import logger
from scipy.optimize import minimize
from scipy.stats import pearsonr


def fit_first_order(y_con, y_con_hat):
    A = np.vstack([np.ones(len(y_con_hat)), y_con_hat]).T
    b = np.linalg.lstsq(A, y_con, rcond=None)[0]
    return b


def fit_second_order(y_con, y_con_hat):
    A = np.vstack([np.ones(len(y_con_hat)), y_con_hat, y_con_hat**2]).T
    b = np.linalg.lstsq(A, y_con, rcond=None)[0]
    return b


def fit_third_order(y_con, y_con_hat):
    A = np.vstack([np.ones(len(y_con_hat)), y_con_hat, y_con_hat**2, y_con_hat**3]).T
    b = np.linalg.lstsq(A, y_con, rcond=None)[0]

    p = np.poly1d(np.flipud(b))
    p2 = np.polyder(p)
    rr = np.roots(p2)
    r = rr[np.imag(rr) == 0]
    monotonic = all(np.logical_or(r > max(y_con_hat), r < min(y_con_hat)))
    if not monotonic:
        logger.error("Not monotonic!")
    return b


def fit_monotonic_third_order(dfile_db, dcon_db=None, pred=None, target_mos=None, target_ci=None, mapping=None):
    """
    Fits third-order function with the constrained to be monotonically.
    increasing. This function may not return an optimal fitting.
    """
    y = dfile_db[target_mos].to_numpy()

    y_hat = dfile_db[pred].to_numpy()

    if dcon_db is None:
        if target_ci in dfile_db:
            ci = dfile_db[target_ci].to_numpy()
        else:
            ci = 0
    else:
        y_con = dcon_db[target_mos].to_numpy()

        if target_ci in dcon_db:
            ci = dcon_db[target_ci].to_numpy()
        else:
            ci = 0

    x = y_hat
    y_hat_min = min(y_hat) - 0.01
    y_hat_max = max(y_hat) + 0.01

    def polynomial(p, x):
        return p[0] + p[1] * x + p[2] * x**2 + p[3] * x**3

    def constraint_2nd_der(p):
        return 2 * p[2] + 6 * p[3] * x

    def constraint_1st_der(p):
        x = np.arange(y_hat_min, y_hat_max, 0.1)
        return p[1] + 2 * p[2] * x + 3 * p[3] * x**2

    def objective_con(p):
        x_map = polynomial(p, x)
        dfile_db["x_map"] = x_map
        x_map_con = dfile_db.groupby("con").mean().x_map.to_numpy()
        err = x_map_con - y_con
        if mapping == "pError":
            p_err = (abs(err) - ci).clip(min=0)
            return (p_err**2).sum()
        elif mapping == "error":
            return (err**2).sum()
        else:
            raise NotImplementedError

    def objective_file(p):
        x_map = polynomial(p, x)
        err = x_map - y
        if mapping == "pError":
            p_err = (abs(err) - ci).clip(min=0)
            return (p_err**2).sum()
        elif mapping == "error":
            return (err**2).sum()
        else:
            raise NotImplementedError

    cons = dict(type="ineq", fun=constraint_1st_der)

    if dcon_db is None:
        res = minimize(
            objective_file,
            x0=np.array([0.0, 1.0, 0.0, 0.0]),
            method="SLSQP",
            constraints=cons,
        )
    else:
        res = minimize(
            objective_con,
            x0=np.array([0.0, 1.0, 0.0, 0.0]),
            method="SLSQP",
            constraints=cons,
        )
    b = res.x
    return b


def calc_mapped(x, b):
    N = x.shape[0]
    order = b.shape[0] - 1
    A = np.zeros([N, order + 1])
    for i in range(order + 1):
        A[:, i] = x ** (i)
    return A @ b


def calc_mapping(
    dfile_db,
    mapping=None,
    dcon_db=None,
    target_mos=None,
    target_ci=None,
    pred=None,
):
    """
    Computes mapping between subjective and predicted MOS.
    """
    if dcon_db is not None:
        y = dcon_db[target_mos].to_numpy()
        y_hat = dfile_db.groupby("con").mean().get(pred).to_numpy()
    else:
        y = dfile_db[target_mos].to_numpy()
        y_hat = dfile_db[pred].to_numpy()

    if mapping is None:
        b = np.array([0, 1, 0, 0])
        d_map = 0
    elif mapping == "first_order":
        b = fit_first_order(y, y_hat)
        d_map = 1
    elif mapping == "second_order":
        b = fit_second_order(y, y_hat)
        d_map = 3
    elif mapping == "third_order_not_monotonic":
        b = fit_third_order(y, y_hat)
        d_map = 4
    elif mapping == "third_order":
        b = fit_monotonic_third_order(
            dfile_db,
            dcon_db=dcon_db,
            pred=pred,
            target_mos=target_mos,
            target_ci=target_ci,
            mapping="error",
        )
        d_map = 4
    else:
        raise NotImplementedError

    return b, d_map


def calc_eval_metrics(y, y_hat, y_hat_map=None, d=None, ci=None):
    """
    Calculate RMSE, mapped RMSE, mapped RMSE* and Pearson's correlation.
    See ITU-T P.1401 for details on RMSE*.
    """
    r = {
        "r_p": np.nan,
        "rmse": np.nan,
        "rmse_map": np.nan,
        "rmse_star_map": np.nan,
    }
    if is_const(y_hat) or any(np.isnan(y)):
        r["r_p"] = np.nan
    else:
        r["r_p"] = pearsonr(y, y_hat)[0]
    r["rmse"] = calc_rmse(y, y_hat)
    if y_hat_map is not None:
        r["rmse_map"] = calc_rmse(y, y_hat_map, d=d)
        if ci is not None:
            r["rmse_star_map"] = calc_rmse_star(y, y_hat_map, ci, d)[0]
    return r


def calc_rmse_star(mos_sub, mos_obj, ci, d):
    N = mos_sub.shape[0]
    error = mos_sub - mos_obj

    if np.isnan(ci).any():
        p_error = np.nan
        rmse_star = np.nan
    else:
        p_error = (abs(error) - ci).clip(min=0)  # Eq (7-27) P.1401
        if (N - d) < 1:
            rmse_star = np.nan
        else:
            rmse_star = np.sqrt(1 / (N - d) * sum(p_error**2))  # Eq (7-29) P.1401

    return rmse_star, p_error, error


def calc_rmse(y_true, y_pred, d=0):
    if d == 0:
        return np.sqrt(np.mean(np.square(y_true - y_pred)))
    N = y_true.shape[0]
    if (N - d) < 1:
        return np.nan
    return np.sqrt(1 / (N - d) * np.sum(np.square(y_true - y_pred)))  # Eq (7-29) P.1401


def is_const(x):
    if np.linalg.norm(x - np.mean(x)) < 1e-13 * np.abs(np.mean(x)):
        return True
    else:
        return np.all(x == x[0])
