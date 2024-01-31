import numpy as np
import torch
import torchaudio as ta


def get_ta_melspec(
    y,
    sr=48e3,
    n_fft=1024,
    hop_length=80,
    win_length=170,
    n_mels=32,
    fmax=16e3,  # for the sake of consistency with original librosa implementation
):
    """
    Calculate mel-spectrograms with torchaudio (librosa-like).
    """
    if isinstance(y, str):
        try:
            y, sr = ta.load(y)
            y = y[0]
        except:
            raise ValueError("Could not load file {}".format(y))

    melSpec = ta.transforms.MelSpectrogram(
        sample_rate=int(sr),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window_fn=torch.hann_window,
        center=True,
        pad_mode="reflect",
        power=1.0,
        n_mels=n_mels,
        norm=None,
        mel_scale="htk",
    )

    S = melSpec(y)

    spec = ta.functional.amplitude_to_DB(S, amin=1e-4, top_db=80.0, multiplier=20.0, db_multiplier=0.0)
    return spec


def segment_specs(x, seg_length=15, seg_hop=3, max_length=None):
    """
    Segment a spectrogram into "seg_length" wide spectrogram segments.
    Instead of using only the frequency bin of the current time step,
    the neighboring bins are included as input to the CNN. For example
    for a seg_length of 7, the previous 3 and the follwing 3 frequency
    bins are included.

    A spectrogram with input size [H x W] will be segmented to:
    [W-(seg_length-1) x C x H x seg_length], where W is the width of the
    original mel-spec (corresponding to the length of the speech signal),
    H is the height of the mel-spec (corresponding to the number of mel bands),
    C is the number of CNN input Channels (always one in our case).
    """
    n_wins = x.shape[1] - (seg_length - 1)
    transposed_x = x.transpose(1, 0)

    unfolded_x = transposed_x.unfold(0, seg_length, 1)
    expanded_x = unfolded_x.unsqueeze(1)
    x = expanded_x[::seg_hop, :]
    n_wins = int(np.ceil(n_wins / seg_hop))
    if max_length is not None:
        if max_length < x.shape[0]:
            raise ValueError(
                "n_wins {} > max_length {}. Increase max window length ms_max_segments!".format(x.shape[0], max_length)
            )
        x_padded = torch.zeros((max_length, x.shape[1], x.shape[2], x.shape[3]))
        x_padded[:n_wins, :] = x
        x = x_padded

    return x, np.array(n_wins)


def process(audio, sr, model, h0, c0, args):
    audio = get_ta_melspec(
        torch.as_tensor(audio).float(),
        sr,
        args["ms_n_fft"],
        args["ms_hop_length"],
        args["ms_win_length"],
        args["ms_n_mels"],
        args["ms_fmax"],
    )
    audio, n_wins = segment_specs(audio, args["ms_seg_length"], args["ms_seg_hop_length"])
    np.set_printoptions(precision=3)
    if args["updates"]:
        n_wins = args["updates"]
        audio = torch.split(audio, args["updates"], dim=0)
        for seg in audio:
            if seg.shape[0] < n_wins:
                to_pad = torch.zeros(audio[0].shape)
                to_pad[: seg.shape[0], :] = seg
                seg = to_pad
            with torch.no_grad():
                out, h0, c0 = model(seg.unsqueeze(0).float(), torch.as_tensor(n_wins).unsqueeze(0), h0, c0)
            print(out[0].numpy())

    else:
        with torch.no_grad():
            out, h0, c0 = model(audio.unsqueeze(0).float(), torch.as_tensor(n_wins).unsqueeze(0), h0, c0)
        print(out[0].numpy())
    return out, h0, c0
