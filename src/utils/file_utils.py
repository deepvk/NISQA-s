import datetime
import os
from pathlib import Path

import torch
import yaml


def save_results(model, args, model_args, opt, epoch, loss, ep_runtime, r, db_results, best, runname, results_hist):
    """
    Save model/results in dictionary and write results csv.
    """
    if args["tr_checkpoint"] == "best_only":
        filename = runname + ".tar"
    else:
        filename = runname + "__" + ("ep_{:03d}".format(epoch + 1)) + ".tar"
    run_output_dir = os.path.join(args["output_dir"], runname)
    model_path = os.path.join(run_output_dir, filename)
    results_path = os.path.join(run_output_dir, runname + "__results.csv")
    Path(run_output_dir).mkdir(parents=True, exist_ok=True)

    results = {
        "runname": runname,
        "epoch": "{:05d}".format(epoch + 1),
        "filename": filename,
        "loss": loss,
        "ep_runtime": "{:0.2f}".format(ep_runtime),
        **r,
        **args,
    }

    for key in results:
        results[key] = str(results[key])

    results_hist.loc[epoch] = results
    results_hist.to_csv(results_path, index=False)

    if (args["tr_checkpoint"] == "every_epoch") or (args["tr_checkpoint"] == "best_only" and best):
        if hasattr(model, "module"):
            state_dict = model.module.state_dict()
            model_name = model.module.name
        else:
            state_dict = model.state_dict()
            model_name = model.name

        torch_dict = {
            "runname": runname,
            "epoch": epoch + 1,
            "model_args": model_args,
            "args": args,
            "model_state_dict": state_dict,
            "optimizer_state_dict": opt.state_dict(),
            "db_results": db_results,
            "results": results,
            "model_name": model_name,
        }

        torch.save(torch_dict, model_path)

    elif (args["tr_checkpoint"] != "every_epoch") and (args["tr_checkpoint"] != "best_only"):
        raise ValueError("selected tr_checkpoint option not available")


def makeRunnameAndWriteYAML(args):
    """
    Creates individual run name.
    """
    runname = args["name"] + "_" + datetime.datetime.today().strftime("%y%m%d_%H%M%S%f")
    print("runname: " + runname)
    run_output_dir = os.path.join(args["output_dir"], runname)
    Path(run_output_dir).mkdir(parents=True, exist_ok=True)
    yaml_path = os.path.join(run_output_dir, runname + ".yaml")
    with open(yaml_path, "w") as file:
        yaml.dump(args, file, default_flow_style=None, sort_keys=False)

    return runname
