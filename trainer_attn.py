import os
import time

import matplotlib.pyplot as plt
import torch
from torch.nn.parallel import data_parallel
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau

from Conv_TasNet import check_parameters
from SI_SNR import si_snr_loss
from utils import get_logger


def to_device(dicts, device):
    """
    Load batch dict to target CUDA device.
    """

    def to_cuda(datas):
        if isinstance(datas, torch.Tensor):
            return datas.to(device)
        if isinstance(datas, list):
            return [data.to(device) for data in datas]
        raise RuntimeError("datas is not torch.Tensor and list type")

    if isinstance(dicts, dict):
        return {key: to_cuda(dicts[key]) for key in dicts}
    raise RuntimeError("input egs's type is not dict")


def init_new_modules(net):
    import torch.nn as nn

    if hasattr(net, "gate"):
        if isinstance(net.gate, nn.Linear):
            nn.init.constant_(net.gate.weight, 0.5)
            nn.init.zeros_(net.gate.bias)
        elif isinstance(net.gate, nn.Parameter):
            with torch.no_grad():
                net.gate.fill_(0.5)

    if hasattr(net, "cross_attention"):
        for name in ["query_proj", "key_proj", "value_proj", "output_proj"]:
            layer = getattr(net.cross_attention, name)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)


def mark_only_added_modules_trainable(net, logger):
    """
    Freeze WavLM and the original ConvTasNet backbone, then unfreeze everything
    else that was added for semantic injection / conditioning.

    Original ConvTasNet backbone considered frozen:
    - encoder / bottleneck / normalization
    - separation repeat blocks' vanilla ConvTasNet layers
    - mask head / decoder
    """
    for param in net.parameters():
        param.requires_grad = False

    original_top_level_prefixes = (
        "encoder.",
        "layern.",
        "LayerN_S.",
        "bottleneck.",
        "BottleN_S.",
        "gen_masks.",
        "decoder.",
    )
    original_block_keywords = (
        ".conv1x1.",
        ".Conv1D_1.",
        ".prelu1.",
        ".PReLU_1.",
        ".norm1.",
        ".norm_1.",
        ".dwconv.",
        ".prelu2.",
        ".PReLU_2.",
        ".norm2.",
        ".norm_2.",
        ".sc_conv.",
        ".Sc_conv.",
    )

    trainable = []
    frozen_wavlm = 0
    frozen_backbone = 0

    for name, param in net.named_parameters():
        is_wavlm = name.startswith("wavlm_encoder.")
        is_original_top_level = name.startswith(original_top_level_prefixes)
        is_original_block = (
            name.startswith("separation.repeats.")
            and any(keyword in name for keyword in original_block_keywords)
        )

        if is_wavlm:
            frozen_wavlm += 1
            logger.info(f"Freeze wavlm parameter: {name}")
            continue

        if is_original_top_level or is_original_block:
            frozen_backbone += 1
            logger.info(f"Freeze backbone parameter: {name}")
            continue

        param.requires_grad = True
        trainable.append((name, param))
        logger.info(f"Train added-module parameter: {name}")

    if not trainable:
        raise RuntimeError(
            "No added-module parameters were found after freezing WavLM and "
            "the original ConvTasNet backbone."
        )

    total_params_m = sum(param.numel() for _, param in trainable) / 1e6
    logger.info(
        "Added-module trainable params: %.2f M | frozen wavlm tensors: %d | "
        "frozen backbone tensors: %d"
        % (total_params_m, frozen_wavlm, frozen_backbone)
    )
    print(f"Added-module trainable params: {total_params_m:.2f} M")

    return [param for _, param in trainable]


def log_trainable_parameters(net, logger):
    trainable = [(name, param) for name, param in net.named_parameters() if param.requires_grad]
    total_numel = sum(param.numel() for _, param in trainable)

    logger.info("=" * 80)
    logger.info("Trainable parameter list")
    logger.info("Trainable tensors: %d", len(trainable))
    logger.info("Trainable elements total: %d", total_numel)

    for name, param in trainable:
        logger.info(
            "[TRAIN] %s | shape=%s | numel=%d | dtype=%s",
            name,
            tuple(param.shape),
            param.numel(),
            param.dtype,
        )

    logger.info("=" * 80)


class Trainer:
    """
    Trainer for Conv-TasNet variants where only cross-attention is trainable.
    """

    def __init__(
        self,
        net,
        checkpoint="checkpoint",
        optimizer="adam",
        gpuid=0,
        optimizer_kwargs=None,
        clip_norm=None,
        min_lr=0,
        patience=0,
        factor=0.5,
        logging_period=100,
        resume=None,
        stop=10,
        num_epochs=100,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device unavailable...exist")
        if not isinstance(gpuid, tuple):
            gpuid = (gpuid,)
        self.device = torch.device(f"cuda:{gpuid[0]}")
        self.gpuid = gpuid

        if checkpoint and not os.path.exists(checkpoint):
            os.makedirs(checkpoint)
        self.checkpoint = checkpoint

        self.logger = get_logger(os.path.join(checkpoint, "trainer.log"), file=True)
        self.clip_norm = clip_norm
        self.logging_period = logging_period
        self.cur_epoch = 0
        self.stop = stop

        if resume["resume_state"]:
            best_ckpt_path = os.path.join(resume["path"], self.checkpoint, "best.pt")
            if os.path.exists(best_ckpt_path):
                cpt = torch.load(best_ckpt_path, map_location="cpu")
                self.cur_epoch = cpt["epoch"]
                self.logger.info(
                    "Resume from checkpoint {}: epoch {:d}".format(
                        resume["path"], self.cur_epoch
                    )
                )
                net.load_state_dict(cpt["model_state_dict"])
                self.net = net.to(self.device)
                self.optimizer = self.create_optimizer(
                    optimizer, optimizer_kwargs, state=cpt["optim_state_dict"]
                )
            else:
                cpt = torch.load(
                    "/home/student/zt/ConvTasNet_master/baseline_checkpoint/best.pt",
                    map_location="cpu",
                )
                net.load_state_dict(cpt["model_state_dict"], strict=False)
                self.logger.info("Loaded baseline checkpoint")
                
                from safetensors.torch import load_file
                state_dict = load_file("/home/student/zt/ConvTasNet_master/baseline_checkpoint/encoder_1b_clean.safetensors")
                self.logger.info("Loaded WavLM finetuned checkpoint")
                # state_dict = torch.load(
                    # "wavlm_large_pretrained/pytorch_model.bin", map_location="cpu"
                # )
                # state_dict = {"encoder." + key: value for key, value in state_dict.items()}
                # self.logger.info("Loaded WavLM pretrained checkpoint")
                net.wavlm_encoder.load_state_dict(state_dict, strict=False)

                init_new_modules(net)
                self.net = net.to(self.device)
                self.optimizer = self.create_optimizer(optimizer, optimizer_kwargs)
        else:
            self.net = net.to(self.device)
            self.optimizer = self.create_optimizer(optimizer, optimizer_kwargs)

        self.param = check_parameters(self.net)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=factor, patience=patience, min_lr=min_lr
        )

        self.logger.info("Starting preparing model ............")
        self.logger.info(
            "Loading model to GPUs:{}, #param: {:.2f}M".format(self.gpuid, self.param)
        )
        if clip_norm:
            self.logger.info("Gradient clipping by {}, default L2".format(clip_norm))

        self.num_epochs = num_epochs
        self.mse = torch.nn.MSELoss()

    def create_optimizer(self, optimizer, kwargs, state=None):
        supported_optimizer = {
            "sgd": torch.optim.SGD,
            "rmsprop": torch.optim.RMSprop,
            "adam": torch.optim.Adam,
            "adadelta": torch.optim.Adadelta,
            "adagrad": torch.optim.Adagrad,
            "adamax": torch.optim.Adamax,
        }
        if optimizer not in supported_optimizer:
            raise ValueError("Now only support optimizer {}".format(optimizer))

        trainable_params = mark_only_added_modules_trainable(self.net, self.logger)
        log_trainable_parameters(self.net, self.logger)
        opt = supported_optimizer[optimizer](trainable_params, **kwargs)

        self.logger.info("Create optimizer {0}: {1}".format(optimizer, kwargs))
        if state is not None:
            opt.load_state_dict(state)
            self.logger.info("Load optimizer state dict from checkpoint")
        return opt

    def save_checkpoint(self, best=True):
        torch.save(
            {
                "epoch": self.cur_epoch,
                "model_state_dict": self.net.state_dict(),
                "optim_state_dict": self.optimizer.state_dict(),
            },
            os.path.join(self.checkpoint, "{0}.pt".format("best" if best else "last")),
        )

    def train(self, train_dataloader):
        self.logger.info("Training model ......")
        losses = []
        start = time.time()
        current_step = 0
        for egs in train_dataloader:
            current_step += 1
            egs = to_device(egs, self.device)
            self.optimizer.zero_grad()
            ests = data_parallel(self.net, egs["mix"], device_ids=self.gpuid)
            loss = si_snr_loss(ests, egs)
            loss.backward()
            if self.clip_norm:
                clip_grad_norm_(self.net.parameters(), self.clip_norm)
            self.optimizer.step()
            losses.append(loss.item())
            if len(losses) % self.logging_period == 0:
                avg_loss = sum(losses[-self.logging_period:]) / self.logging_period
                self.logger.info(
                    "<epoch:{:3d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}, batch:{:d} utterances> ".format(
                        self.cur_epoch,
                        current_step,
                        self.optimizer.param_groups[0]["lr"],
                        avg_loss,
                        len(losses),
                    )
                )
                print(
                    "<epoch:{:3d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}, batch:{:d} utterances> ".format(
                        self.cur_epoch,
                        current_step,
                        self.optimizer.param_groups[0]["lr"],
                        avg_loss,
                        len(losses),
                    )
                )
        end = time.time()
        total_loss_avg = sum(losses) / len(losses)
        self.logger.info(
            "<epoch:{:3d}, lr:{:.3e}, loss:{:.3f}, Total time:{:.3f} min> ".format(
                self.cur_epoch,
                self.optimizer.param_groups[0]["lr"],
                total_loss_avg,
                (end - start) / 60,
            )
        )
        return total_loss_avg

    def val(self, val_dataloader):
        self.logger.info("Validation model ......")
        self.net.eval()
        losses = []
        current_step = 0
        start = time.time()
        with torch.no_grad():
            for egs in val_dataloader:
                current_step += 1
                egs = to_device(egs, self.device)
                ests = data_parallel(self.net, egs["mix"], device_ids=self.gpuid)
                loss = si_snr_loss(ests, egs)
                losses.append(loss.item())
                if len(losses) % self.logging_period == 0:
                    avg_loss = sum(losses[-self.logging_period:]) / self.logging_period
                    self.logger.info(
                        "<epoch:{:3d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}, batch:{:d} utterances> ".format(
                            self.cur_epoch,
                            current_step,
                            self.optimizer.param_groups[0]["lr"],
                            avg_loss,
                            len(losses),
                        )
                    )
                    print(
                        "<epoch:{:3d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}, batch:{:d} utterances> ".format(
                            self.cur_epoch,
                            current_step,
                            self.optimizer.param_groups[0]["lr"],
                            avg_loss,
                            len(losses),
                        )
                    )
        end = time.time()
        total_loss_avg = sum(losses) / len(losses)
        self.logger.info(
            "<epoch:{:3d}, lr:{:.3e}, loss:{:.3f}, Total time:{:.3f} min> ".format(
                self.cur_epoch,
                self.optimizer.param_groups[0]["lr"],
                total_loss_avg,
                (end - start) / 60,
            )
        )
        return total_loss_avg

    def run(self, train_loader, val_loader):
        train_loss, val_loss = [], []
        with torch.cuda.device(self.gpuid[0]):
            self.save_checkpoint(best=False)
            v_loss = self.val(val_loader)
            best_loss = v_loss
            self.logger.info("Starting epoch from {:d}, loss = {:.4f}".format(self.cur_epoch, v_loss))

            no_improve = 0
            while self.cur_epoch < self.num_epochs:
                self.net.train()
                self.cur_epoch += 1
                t_loss = self.train(train_loader)
                v_loss = self.val(val_loader)

                train_loss.append(t_loss)
                val_loss.append(v_loss)
                self.scheduler.step(v_loss)

                if v_loss >= best_loss:
                    no_improve += 1
                    self.logger.info(
                        "No improvement, Best Loss: {:.4f}, Now Loss: {:.4f}, Counter: {:d}".format(
                            best_loss, v_loss, no_improve
                        )
                    )
                else:
                    best_loss = v_loss
                    no_improve = 0
                    self.save_checkpoint(best=True)
                    self.logger.info("Epoch: {:d}, Now Best Loss Change: {:.4f}".format(self.cur_epoch, best_loss))

                self.save_checkpoint(best=False)
                plt.figure()
                plt.plot(range(1, self.cur_epoch + 1), train_loss, color="blue")
                plt.plot(range(1, self.cur_epoch + 1), val_loss, color="red")
                plt.xlabel("epoch")
                plt.ylabel("loss")
                plt.savefig(os.path.join(self.checkpoint, "loss.png"))
                plt.close()

                if no_improve == self.stop:
                    self.logger.info(
                        "Stop training cause no impr for {:d} epochs".format(no_improve)
                    )
                    break
