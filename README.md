# NISQA-s: Speech Quality and Naturalness Assessment for Online Inference

NISQA-s is highly stripped and optimized version of original [NISQA](https://github.com/gabrielmittag/NISQA) metric. 
It is aiming to create universal metrics set for both offline and online evaluation of audio quality.

This version supports only CNN+LSTM version of original model (since other modifications don't support streaming or perform too slow). 
It uses the same architecture with some tweaks for streaming purposes.
Also there's no MOS-only model, since main model supports MOS prediction (for simplicity of the code and repo).

## Installation

_(Optional)_ Create new venv or conda env

Then just `pip install -r requirements.txt`

Note that there may be some problems with `torch` installation. If so, follow official [PyTorch instructions](https://pytorch.org/get-started/locally/)

## Quick start

If you want to just run this repo with provided config and samples - 
```
python -m scripts.run_infer_file
```

If you want to test online inference from your mic - 
```
python -m scripts.run_infer_mic
```
This will log inference results to terminal, so pay attention to it.

## Config options

Default config is `config/nisqa_s.yaml`. All configurations for everything related to training and inference are happening here. 
There are detailed comments about each parameter, so we'll cover only the most important ones for inference:

* `ckp`: path to trained checkpoint (`weights/nisqa_s.tar` by default)

* `sample`: path to evaluated file 

If you plan to run online inference, you should pay close attention to last 4 arguments in this config:

* `frame` lets you choose length of buffer to feed into the model;

* `updates` will make the model spit metrics more often (check argument description)

* `sd_device`'s ID should be provided if you want to run this on different input devices (e.g. sound-card mic).
First run of `run_infer_mic.py` will show you those IDs.

* `sd_dump` lets you save mic input to check the results in offline later.

And finally, you can run custom config for your experiments - just add `--yaml` argument to `python -m scripts.run_infer_file`/`python -m scripts.run_infer_mic` and provide path to yor own config:
```
python -m scripts.run_infer_file --yaml path/to/custom/config.yaml
```

## Training

We provide simple interface for training your own version of NISQA-s. 

Firstly, you will need the dataset. You can obtain it from [official NISQA repo](https://github.com/gabrielmittag/NISQA/wiki/NISQA-Corpus).
This is probably the only (but definitely the best) way to train this, since the data needs to be very specifically labeled for this to work.

To train the same version as provided - 
```
python -m scripts.run_train
```

Remember to check `name` of the experiment in `nisqa_s.yaml` 
and path to NISQA Corpus in `data_dir`, as well as path to save the model (`output_dir`)

### Training and model parameters in config

* Since you're most probably using NISQA Corpus, there is no need to change anything in `Dataset options`. 
If you use some hand-made dataset - you need to refer to [this guide](https://github.com/gabrielmittag/NISQA#finetuning--transfer-learning).

* `Training options` contains all parameters connected to training setup (like learning rates, batch size etc.).

* You can also experiment with bias loss by enabling `Bias loss options`

* Change `Mel-Specs options` if you want to experiment on different samplerates, Fourier lengths or sample length for training 
(although it is highly not recommended to lower value of `ms_max_length` because of NISQA Corpus labeling)

* `CNN parameters` and `LSTM parameters` - change those to experiment on different parameters of convolutional and recurrent layers.

Note that provided checkpoint is trained with provided config.

# Citations
```
@article{Mittag_Naderi_Chehadi_Möller_2021, 
  title={Nisqa: A deep CNN-self-attention model for multidimensional speech quality prediction with crowdsourced datasets}, 
  DOI={10.21437/interspeech.2021-299}, 
  journal={Interspeech 2021}, 
  author={Mittag, Gabriel and Naderi, Babak and Chehadi, Assmaa and Möller, Sebastian}, 
  year={2021}
} 
```
```
@misc{deepvk2024nisqa,
  author = {Ivan, Beskrovnyi},
  title = {nisqa-s},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {https://github.com/deepvk/nisqa-s}
}
```








