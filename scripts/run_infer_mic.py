import os.path
import queue

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import yaml

from src.core.model_torch import model_init
from src.utils.process_utils import process
from src.utils.train_utils import yamlparser

q = queue.Queue()


def callback(indata, frames, time, status):
    global buffer, N, args, sr, model, h0, c0
    buffer = np.concatenate((buffer, indata[:, 0]))

    if len(buffer) >= N:
        q.put(buffer[:N].copy())
        out, h0, c0 = process(buffer[:N], sr, model, h0, c0, args)
        buffer = buffer[N:]


if __name__ == "__main__":
    args = yamlparser()

    with open(args["yaml"], "r") as ymlfile:
        args_yaml = yaml.load(ymlfile, Loader=yaml.FullLoader)
    args = {**args_yaml, **args}

    model, h0, c0 = model_init(args)

    print(sd.query_devices())

    device = None
    sr = int(sd.query_devices(sd.default.device[0])["default_samplerate"])
    if args["sd_device"]:
        device = args["sd_device"]
        sr = int(sd.query_devices(device)["default_samplerate"])

    print("Samplerate of input device - {} Hz".format(sr))

    buffer = np.zeros(0)
    N = int(sr * args["frame"])

    if args["warmup"]:
        _, _, _ = process(torch.zeros((1, N)), sr, model, h0, c0, args)

    if args["sd_dump"]:
        if os.path.isfile(args["sd_dump"]):
            os.remove(args["sd_dump"])

        try:
            with sf.SoundFile(args["sd_dump"], mode="x", samplerate=sr, channels=1, subtype="PCM_16") as file:
                with sd.InputStream(
                    samplerate=sr,
                    channels=1,
                    dtype="float32",
                    blocksize=0,
                    device=device,
                    latency=None,
                    extra_settings=None,
                    callback=callback,
                ):
                    print("Listening... Press CTRL+C to stop.")
                    print("NOI    COL   DISC  LOUD  MOS")
                    while True:
                        file.write(q.get())
        except KeyboardInterrupt:
            print("\nRecording stopped.")
    else:
        try:
            with sd.InputStream(
                samplerate=sr,
                channels=1,
                dtype="float32",
                blocksize=0,
                device=device,
                latency=None,
                extra_settings=None,
                callback=callback,
            ):
                print("Listening... Press CTRL+C to stop.")
                print("NOI    COL   DISC  LOUD  MOS")
                while True:
                    q.get()
        except KeyboardInterrupt:
            print("\nRecording stopped.")
