import argparse
import sys

import torch

sys.path.append("./options")

from Conv_TasNet_wavlm_dwconvFuse import ConvTasNet as Net_dwAtt1
from Conv_TasNet_wavlm_dwconvFuse_att2 import ConvTasNet as Net_dwAtt2
from Conv_TasNet_wavlm_dwconvFuse_film import ConvTasNet as Net_dwAtt_film
from Conv_TasNet_wavlm_dwconvFuse_nogate import ConvTasNet as Net_dwAtt1_nogate
from Conv_TasNet_wavlm_dwconvFuse_wogate import ConvTasNet as Net_dwAtt1_wogate
from Conv_TasNet_wavlm_dwconvFuse_woshare import ConvTasNet as Net_dwAtt1_woshare
from Conv_TasNet_wavlm_film import ConvTasNet as Net_film
from Conv_TasNet_wavlm_repeat_gate import ConvTasNet as Net_gate
from Conv_TasNet_wavlm_up import ConvTasNet as Net_up
from DataLoaders import make_dataloader
from option import parse
from trainer_attn import Trainer
from utils import get_logger


def build_model(model_net, net_conf):
    if model_net == "Net_dwAtt1":
        return Net_dwAtt1(**net_conf)
    if model_net == "Net_dwAtt2":
        return Net_dwAtt2(**net_conf)
    if model_net == "Net_dwAtt_film":
        return Net_dwAtt_film(**net_conf)
    if model_net == "Net_film":
        return Net_film(**net_conf)
    if model_net == "Net_gate":
        return Net_gate(**net_conf)
    if model_net == "Net_up":
        return Net_up(**net_conf)
    if model_net == "Net_dwAtt1_wogate":
        return Net_dwAtt1_wogate(**net_conf)
    if model_net == "Net_dwAtt1_nogate":
        return Net_dwAtt1_nogate(**net_conf)
    if model_net == "Net_dwAtt1_woshare":
        return Net_dwAtt1_woshare(**net_conf)
    raise ValueError(f"Unknown model_net: {model_net}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", type=str, help="Path to option YAML file.")
    parser.add_argument("--model_net", type=str, help="Model Net.")
    args = parser.parse_args()

    opt = parse(args.opt, is_tain=True)
    logger = get_logger(__name__)

    logger.info("Building the model of Conv-TasNet")
    net = build_model(args.model_net, opt["net_conf"])

    logger.info("Building the trainer of Conv-TasNet")
    gpuid = tuple(opt["gpu_ids"])
    trainer = Trainer(
        net,
        **opt["train"],
        resume=opt["resume"],
        gpuid=gpuid,
        optimizer_kwargs=opt["optimizer_kwargs"],
    )

    logger.info("Making the train and test data loader")
    train_loader = make_dataloader(
        is_train=True,
        data_kwargs=opt["datasets"]["train"],
        num_workers=opt["datasets"]["num_workers"],
        batch_size=opt["datasets"]["batch_size"],
    )
    val_loader = make_dataloader(
        is_train=False,
        data_kwargs=opt["datasets"]["val"],
        num_workers=opt["datasets"]["num_workers"],
        batch_size=opt["datasets"]["batch_size"],
    )
    logger.info(
        "Train data loader: {}, Test data loader: {}".format(
            len(train_loader), len(val_loader)
        )
    )
    trainer.run(train_loader, val_loader)


if __name__ == "__main__":
    main()
