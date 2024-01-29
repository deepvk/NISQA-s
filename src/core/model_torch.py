import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class PoolAvg(torch.nn.Module):
    """
    PoolAvg: Average pooling that consideres masked time-steps.
    """

    def __init__(self, d_input, output_size):
        super().__init__()

        self.linear = nn.Linear(d_input, output_size)

    def forward(self, x, n_wins):
        mask = torch.arange(x.shape[1])[None, :] < n_wins[:, None].to("cpu").to(torch.long)
        mask = ~mask.unsqueeze(2).to(x.device)
        x.masked_fill_(mask, 0.0)

        x = torch.div(x.sum(1), n_wins.unsqueeze(1))

        x = self.linear(x)

        return x


class AdaptCNN(nn.Module):
    """
    AdaptCNN: CNN with adaptive maxpooling that can be used as framewise model.
    Overall, it has six convolutional layers. This CNN module is more flexible
    than the StandardCNN that requires a fixed input dimension of 48x15.
    """

    def __init__(
            self,
            input_channels,
            c_out_1,
            c_out_2,
            c_out_3,
            kernel_size,
            dropout,
            pool_1,
            pool_2,
            pool_3,
            fc_out_h=20,
    ):
        super().__init__()
        self.name = "CNN_adapt"

        self.input_channels = input_channels
        self.c_out_1 = c_out_1
        self.c_out_2 = c_out_2
        self.c_out_3 = c_out_3
        self.kernel_size = kernel_size
        self.pool_1 = pool_1
        self.pool_2 = pool_2
        self.pool_3 = pool_3
        self.dropout_rate = dropout
        self.fc_out_h = fc_out_h

        self.dropout = nn.Dropout2d(p=self.dropout_rate)

        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size, self.kernel_size)

        self.kernel_size_last = (self.kernel_size[0], self.pool_3[1])

        if self.kernel_size[1] == 1:
            self.cnn_pad = (1, 0)
        else:
            self.cnn_pad = (1, 1)

        self.conv1 = nn.Conv2d(self.input_channels, self.c_out_1, self.kernel_size, padding=self.cnn_pad)

        self.bn1 = nn.BatchNorm2d(self.conv1.out_channels)

        self.conv2 = nn.Conv2d(self.conv1.out_channels, self.c_out_2, self.kernel_size, padding=self.cnn_pad)

        self.bn2 = nn.BatchNorm2d(self.conv2.out_channels)

        self.conv3 = nn.Conv2d(self.conv2.out_channels, self.c_out_3, self.kernel_size, padding=self.cnn_pad)

        self.bn3 = nn.BatchNorm2d(self.conv3.out_channels)

        self.conv4 = nn.Conv2d(self.conv3.out_channels, self.c_out_3, self.kernel_size, padding=self.cnn_pad)

        self.bn4 = nn.BatchNorm2d(self.conv4.out_channels)

        self.conv5 = nn.Conv2d(self.conv4.out_channels, self.c_out_3, self.kernel_size, padding=self.cnn_pad)

        self.bn5 = nn.BatchNorm2d(self.conv5.out_channels)

        self.conv6 = nn.Conv2d(self.conv5.out_channels, self.c_out_3, self.kernel_size_last, padding=(1, 0))

        self.bn6 = nn.BatchNorm2d(self.conv6.out_channels)

        if self.fc_out_h:
            self.fc = nn.Linear(self.conv6.out_channels * self.pool_3[0], self.fc_out_h)
            self.fan_out = self.fc_out_h
        else:
            self.fan_out = self.conv6.out_channels * self.pool_3[0]

    def forward(self, x, n_wins):
        x_packed = pack_padded_sequence(x, n_wins.cpu(), batch_first=True, enforce_sorted=False)

        x = F.relu(self.bn1(self.conv1(x_packed.data)))
        x = F.adaptive_max_pool2d(x, output_size=self.pool_1)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.adaptive_max_pool2d(x, output_size=self.pool_2)
        x = self.dropout(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = F.adaptive_max_pool2d(x, output_size=self.pool_3)
        x = self.dropout(x)

        x = F.relu(self.bn5(self.conv5(x)))
        x = self.dropout(x)

        x = F.relu(self.bn6(self.conv6(x)))

        x = x.view(-1, self.conv6.out_channels * self.pool_3[0])

        if self.fc_out_h:
            x = self.fc(x)

        x = x_packed._replace(data=x)

        x, _ = pad_packed_sequence(x, batch_first=True, padding_value=0.0, total_length=n_wins.max())
        return x


class LSTM(nn.Module):
    def __init__(self, input_size, lstm_h=128, num_layers=2, dropout=0.1, bidirectional=True):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_h,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )
        if bidirectional:
            num_directions = 2
        else:
            num_directions = 1
        self.fan_out = num_directions * lstm_h

    def forward(self, x, n_wins, h0=None, c0=None):
        x = pack_padded_sequence(x, n_wins.cpu(), batch_first=True, enforce_sorted=False)
        self.lstm.flatten_parameters()
        x, (h, c) = self.lstm(x, (h0, c0))
        x, _ = pad_packed_sequence(x, batch_first=True, padding_value=0.0, total_length=n_wins.max())
        return x, h, c


class NISQA_DIM(nn.Module):
    """
    NISQA_DIM: The main speech quality model with speech quality dimension
    estimation (MOS, Noisiness, Coloration, Discontinuity, and Loudness).
    """

    def __init__(
            self,
            cnn_c_out_1=16,
            cnn_c_out_2=32,
            cnn_c_out_3=64,
            cnn_kernel_size=3,
            cnn_dropout=0.2,
            cnn_pool_1=[24, 7],
            cnn_pool_2=[12, 5],
            cnn_pool_3=[6, 3],
            cnn_fc_out_h=None,
            td_lstm_h=128,
            td_lstm_num_layers=1,
            td_lstm_dropout=0,
            td_lstm_bidirectional=True,
    ):
        super().__init__()

        self.name = "NISQA_DIM"

        self.cnn = AdaptCNN(
            input_channels=1,
            c_out_1=cnn_c_out_1,
            c_out_2=cnn_c_out_2,
            c_out_3=cnn_c_out_3,
            kernel_size=cnn_kernel_size,
            dropout=cnn_dropout,
            pool_1=cnn_pool_1,
            pool_2=cnn_pool_2,
            pool_3=cnn_pool_3,
            fc_out_h=cnn_fc_out_h,
        )

        self.time_dependency = LSTM(
            input_size=self.cnn.fan_out,
            lstm_h=td_lstm_h,
            num_layers=td_lstm_num_layers,
            dropout=td_lstm_dropout,
            bidirectional=td_lstm_bidirectional,
        )

        self.pool_layers = nn.ModuleList(
            [
                PoolAvg(
                    self.time_dependency.fan_out,
                    output_size=1,
                )
                for _ in range(5)
            ]
        )

    def forward(self, x, n_wins, h0, c0):
        x = self.cnn(x, n_wins)
        x, h, c = self.time_dependency(x, n_wins, h0, c0)
        out = [mod(x, n_wins) for mod in self.pool_layers]
        out = torch.cat(out, dim=1)

        return out, h, c


def loadModel(args):
    """
    Loads the Pytorch models with given input arguments.
    """
    # if True overwrite input arguments from pretrained model
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    if "tr_device" in args:
        if args["tr_device"] == "cpu":
            dev = torch.device("cpu")
        elif args["tr_device"] == "cuda":
            dev = torch.device("cuda")
    print("Device: {}".format(dev))

    if "tr_parallel" in args:
        if (dev == torch.device("cpu")) and args["tr_parallel"]:
            args["tr_parallel"] == False
            print("Using CPU -> tr_parallel set to False")
    if args["pretrained_model"]:
        if os.path.isabs(args["pretrained_model"]):
            model_path = os.path.join(args["pretrained_model"])
        else:
            model_path = os.path.join(os.getcwd(), args["pretrained_model"])
        checkpoint = torch.load(model_path, map_location=dev)

        # update checkpoint arguments with new arguments
        checkpoint["args"].update(args)
        args = checkpoint["args"]

    args["dim"] = True
    args["csv_mos_train"] = None  # column names hardcoded for dim models
    args["csv_mos_val"] = None

    args["double_ended"] = False
    args["csv_ref"] = None

    # Load Model
    model_args = {
        "cnn_c_out_1": args["cnn_c_out_1"],
        "cnn_c_out_2": args["cnn_c_out_2"],
        "cnn_c_out_3": args["cnn_c_out_3"],
        "cnn_kernel_size": args["cnn_kernel_size"],
        "cnn_dropout": args["cnn_dropout"],
        "cnn_pool_1": args["cnn_pool_1"],
        "cnn_pool_2": args["cnn_pool_2"],
        "cnn_pool_3": args["cnn_pool_3"],
        "cnn_fc_out_h": args["cnn_fc_out_h"],
        "td_lstm_h": args["td_lstm_h"],
        "td_lstm_num_layers": args["td_lstm_num_layers"],
        "td_lstm_dropout": args["td_lstm_dropout"],
        "td_lstm_bidirectional": args["td_lstm_bidirectional"],
    }

    model = NISQA_DIM(**model_args)

    # Load weights if pretrained model is used ------------------------------------
    if args["pretrained_model"]:
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        print("Loaded pretrained model from " + args["pretrained_model"])
        if missing_keys:
            print("missing_keys:")
            print(missing_keys)
        if unexpected_keys:
            print("unexpected_keys:")
            print(unexpected_keys)
    return model, dev, model_args


def model_init(args):
    model = NISQA_DIM(
        cnn_c_out_1=args["cnn_c_out_1"],
        cnn_c_out_2=args["cnn_c_out_2"],
        cnn_c_out_3=args["cnn_c_out_3"],
        cnn_kernel_size=args["cnn_kernel_size"],
        cnn_dropout=args["cnn_dropout"],
        cnn_pool_1=args["cnn_pool_1"],
        cnn_pool_2=args["cnn_pool_2"],
        cnn_pool_3=args["cnn_pool_3"],
        cnn_fc_out_h=args["cnn_fc_out_h"],
        td_lstm_h=args["td_lstm_h"],
        td_lstm_num_layers=args["td_lstm_num_layers"],
        td_lstm_dropout=args["td_lstm_num_layers"],
        td_lstm_bidirectional=args["td_lstm_bidirectional"],
    )

    ckp = torch.load(args["ckp"], map_location="cpu")
    model.load_state_dict(ckp["model_state_dict"], strict=True)
    model = model.to(torch.device(args["inf_device"]))
    model.eval()

    # init lstm states
    directions = 2 if args["td_lstm_bidirectional"] else 1
    h0 = torch.zeros(args["td_lstm_num_layers"] * directions, 1, args["td_lstm_h"])
    c0 = torch.zeros(args["td_lstm_num_layers"] * directions, 1, args["td_lstm_h"])

    return model, h0, c0
