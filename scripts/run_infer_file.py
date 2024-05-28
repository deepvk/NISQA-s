import numpy as np
import soundfile as sf
import torch
import yaml

from src.core.model_torch import model_init
from src.utils.process_utils import process
from src.utils.train_utils import yamlparser

if __name__ == "__main__":
    args = yamlparser()

    with open(args["yaml"], "r") as ymlfile:
        args_yaml = yaml.load(ymlfile, Loader=yaml.FullLoader)
    args = {**args_yaml, **args}

    # init model
    model, h0, c0 = model_init(args)

    # split input file into frames
    audio, sr = sf.read(args["sample"])
    framesize = sr * args["frame"]
    audio = torch.as_tensor(audio)

    # if length of audio is not divisible by framesize, then pad
    if audio.shape[0] % framesize != 0:
        audio = torch.cat((audio, torch.zeros(framesize - audio.shape[0] % framesize)))

    audio_spl = torch.split(audio, framesize, dim=0)

    # if warmup is needed
    if args["warmup"]:
        _, _, _ = process(torch.zeros((1, framesize)), sr, model, h0, c0, args)

    out_all = []
    print("NOI    COL   DISC  LOUD  MOS")
    np.set_printoptions(precision=3)
    for audio in audio_spl:
        print(audio.shape)
        out, h0, c0 = process(audio, sr, model, h0, c0, args)
        out_all.append(out[0].numpy())

    avg_out = np.mean(out_all, axis=0)
    print("Average over file:")
    print(avg_out)
