import multiprocessing
import os

import numpy as np
import pandas as pd
from loguru import logger
from torch.utils.data import Dataset
from tqdm import tqdm

from src.utils.process_utils import get_ta_melspec, segment_specs


class SpeechQualityDataset(Dataset):
    """
    Dataset for Speech Quality Model.
    """

    def __init__(
        self,
        df,
        df_con=None,
        data_dir="",
        folder_column="",
        filename_column="filename",
        mos_column="MOS",
        seg_length=15,
        to_memory=False,
        to_memory_workers=0,
        transform=None,
        seg_hop_length=1,
        ms_n_fft=1024,
        ms_hop_length=80,
        ms_win_length=170,
        ms_n_mels=32,
        ms_sr=48e3,
        ms_fmax=16e3,
        ms_channel=None,
        ms_max_length=None,
    ):
        self.df = df
        self.df_con = df_con
        self.data_dir = data_dir
        self.folder_column = folder_column
        self.filename_column = filename_column
        self.mos_column = mos_column
        self.seg_length = seg_length
        self.seg_hop_length = seg_hop_length
        self.transform = transform
        self.to_memory_workers = to_memory_workers
        self.ms_n_fft = ms_n_fft
        self.ms_hop_length = ms_hop_length
        self.ms_win_length = ms_win_length
        self.ms_n_mels = ms_n_mels
        self.ms_sr = ms_sr
        self.ms_fmax = ms_fmax
        self.ms_channel = ms_channel
        self.max_length = ms_max_length

        # if True load all specs to memory
        self.to_memory = False
        if to_memory:
            self._to_memory()

    def _to_memory_multi_helper(self, idx):
        return [self._load_spec(i) for i in idx]

    def _to_memory(self):
        if self.to_memory_workers == 0:
            self.mem_list = [self._load_spec(idx) for idx in tqdm(range(len(self)))]
        else:
            buffer_size = 128
            idx = np.arange(len(self))
            n_bufs = int(len(idx) / buffer_size)
            idx = (
                idx[: buffer_size * n_bufs].reshape(-1, buffer_size).tolist()
                + idx[buffer_size * n_bufs :].reshape(1, -1).tolist()
            )
            with multiprocessing.Pool(self.to_memory_workers) as pool:
                mem_list = []
                for out in tqdm(pool.imap(self._to_memory_multi_helper, idx), total=len(idx)):
                    mem_list = mem_list + out
                self.mem_list = mem_list
        self.to_memory = True

    def _load_spec(self, index):
        # Load spec
        file_path = os.path.join(self.data_dir, self.df[self.filename_column].iloc[index])
        spec = get_ta_melspec(
            file_path,
            sr=self.ms_sr,
            n_fft=self.ms_n_fft,
            hop_length=self.ms_hop_length,
            win_length=self.ms_win_length,
            n_mels=self.ms_n_mels,
            fmax=self.ms_fmax,
        )

        return spec

    def __getitem__(self, index):
        assert isinstance(index, int), "index must be integer (no slice)"

        if self.to_memory:
            spec = self.mem_list[index]
        else:
            spec = self._load_spec(index)

        # Apply transformation if given
        if self.transform:
            spec = self.transform(spec)

        x_spec_seg, n_wins = segment_specs(spec, self.seg_length, self.seg_hop_length, self.max_length)

        # Get MOS (apply NaN in case of prediction only mode)
        if self.mos_column == "predict_only":
            y = np.full((5, 1), np.nan).reshape(-1).astype("float32")
        else:
            y_mos = self.df["mos"].iloc[index].reshape(-1).astype("float32")
            y_noi = self.df["noi"].iloc[index].reshape(-1).astype("float32")
            y_dis = self.df["dis"].iloc[index].reshape(-1).astype("float32")
            y_col = self.df["col"].iloc[index].reshape(-1).astype("float32")
            y_loud = self.df["loud"].iloc[index].reshape(-1).astype("float32")
            y = np.concatenate((y_mos, y_noi, y_dis, y_col, y_loud), axis=0)

        return x_spec_seg, y, (index, n_wins)

    def __len__(self):
        return len(self.df)


def loadDatasetsCSV(args):
    """
    Loads training and validation dataset for training.
    """
    csv_file_path = os.path.join(args["data_dir"], args["csv_file"])
    dfile = pd.read_csv(csv_file_path)

    if not set(args["csv_db_train"] + args["csv_db_val"]).issubset(dfile.db.unique().tolist()):
        missing_datasets = set(args["csv_db_train"] + args["csv_db_val"]).difference(dfile.db.unique().tolist())
        raise ValueError("Not all dbs found in csv:", missing_datasets)

    df_train = dfile[dfile.db.isin(args["csv_db_train"])].reset_index()
    df_val = dfile[dfile.db.isin(args["csv_db_val"])].reset_index()

    if args["csv_con"] is not None:
        csv_con_path = os.path.join(args["data_dir"], args["csv_con"])
        dcon = pd.read_csv(csv_con_path)
        dcon_train = dcon[dcon.db.isin(args["csv_db_train"])].reset_index()
        dcon_val = dcon[dcon.db.isin(args["csv_db_val"])].reset_index()
    else:
        dcon = None
        dcon_train = None
        dcon_val = None

    logger.info("Training size: {}, Validation size: {}".format(len(df_train), len(df_val)))

    # creating Datasets ---------------------------------------------------
    ds_train = SpeechQualityDataset(
        df_train,
        df_con=dcon_train,
        data_dir=args["data_dir"],
        filename_column=args["csv_deg"],
        mos_column=args["csv_mos_train"],
        seg_length=args["ms_seg_length"],
        to_memory=args["tr_ds_to_memory"],
        to_memory_workers=args["tr_ds_to_memory_workers"],
        seg_hop_length=args["ms_seg_hop_length"],
        transform=None,
        ms_n_fft=args["ms_n_fft"],
        ms_hop_length=args["ms_hop_length"],
        ms_win_length=args["ms_win_length"],
        ms_n_mels=args["ms_n_mels"],
        ms_sr=args["ms_sr"],
        ms_fmax=args["ms_fmax"],
        ms_channel=args["ms_channel"],
        ms_max_length=args["ms_max_length"],
    )

    ds_val = SpeechQualityDataset(
        df_val,
        df_con=dcon_val,
        data_dir=args["data_dir"],
        filename_column=args["csv_deg"],
        mos_column=args["csv_mos_val"],
        seg_length=args["ms_seg_length"],
        to_memory=args["tr_ds_to_memory"],
        to_memory_workers=args["tr_ds_to_memory_workers"],
        seg_hop_length=args["ms_seg_hop_length"],
        transform=None,
        ms_n_fft=args["ms_n_fft"],
        ms_hop_length=args["ms_hop_length"],
        ms_win_length=args["ms_win_length"],
        ms_n_mels=args["ms_n_mels"],
        ms_sr=args["ms_sr"],
        ms_fmax=args["ms_fmax"],
        ms_channel=args["ms_channel"],
        ms_max_length=args["ms_max_length"],
    )

    return ds_train, ds_val
