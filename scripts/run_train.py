import yaml

from src.core.train import train_dim
from src.utils.train_utils import yamlparser

if __name__ == "__main__":
    args = yamlparser()
    with open(args["yaml"], "r") as ymlfile:
        args_yaml = yaml.load(ymlfile, Loader=yaml.FullLoader)
    args = {**args_yaml, **args}

    train_dim(args)
