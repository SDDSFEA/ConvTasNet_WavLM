import torch
import time
import os
import sys
from utils import get_logger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.parallel import data_parallel
from torch.nn.utils import clip_grad_norm_
from SI_SNR import si_snr_loss
import matplotlib.pyplot as plt
from Conv_TasNet import check_parameters
import torch.nn as nn

def to_device(dicts, device):
    '''
       load dict data to cuda
    '''
    def to_cuda(datas):
        if isinstance(datas, torch.Tensor):
            return datas.to(device)
        elif isinstance(datas,list):
            return [data.to(device) for data in datas]
        else:
            raise RuntimeError('datas is not torch.Tensor and list type')

    if isinstance(dicts, dict):
        return {key: to_cuda(dicts[key]) for key in dicts}
    else:
        raise RuntimeError('input egs\'s type is not dict')

def init_new_modules(net):
    # 初始化 gate
    if hasattr(net, 'gate'):
        if isinstance(net.gate, nn.Linear):
            nn.init.constant_(net.gate.weight, 0.5)
            nn.init.zeros_(net.gate.bias)
        elif isinstance(net.gate, nn.Parameter):
            with torch.no_grad():
                net.gate.fill_(0.5)
    # 初始化 cross attention Linear
    if hasattr(net, 'cross_attention'):
        for name in ['query_proj', 'key_proj', 'value_proj', 'output_proj']:
            layer = getattr(net.cross_attention, name)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

import torch
import torch.nn as nn

def freeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False


def _set_requires_grad(obj, flag: bool):
    # obj can be nn.Module / nn.Parameter / Tensor
    if obj is None:
        return
    if isinstance(obj, nn.Module):
        for p in obj.parameters():
            p.requires_grad = flag
    elif isinstance(obj, (nn.Parameter, torch.Tensor)):
        obj.requires_grad = flag

def unfreeze_only_attn_gate_and_sep_attn(model: nn.Module, logger=None):
    """
    目标：
      1) Conv1D_Block 里 self.attn_gate（以及可选 self.gate）解冻
      2) SeparationModule repeats[*]['attn'] (名字匹配 .repeats. + .attn.) 解冻
      其余全部 freeze
    """
    freeze_all(model)

    def _log(msg):
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)

    # A) 解冻 Conv1D_Block 的 gate 类属性（包括 attn_gate）
    gate_attr_names = ["attn_gate", "gate"]  # 你还想解冻别的 gate 就往里加
    gate_hits = 0

    for module_name, m in model.named_modules():
        for attr in gate_attr_names:
            if hasattr(m, attr):
                g = getattr(m, attr)
                _set_requires_grad(g, True)
                gate_hits += 1
                _log(f"[UNFREEZE] {module_name}.{attr}  type={type(g)}")

    # B) 解冻 SeparationModule repeats[*]['attn']：按参数名最稳
    attn_param_hits = 0
    for name, p in model.named_parameters():
        if (".repeats." in name) and (".attn." in name):
            p.requires_grad = True
            attn_param_hits += 1

    _log(f"gate_hits = {gate_hits}, attn_param_hits = {attn_param_hits}")

    # C) 打印所有可训练参数（完整）
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    _log(f"✅ Trainable tensors = {len(trainable)}")
    for n, p in trainable:
        _log(f"[T] {n} | shape={tuple(p.shape)} | numel={p.numel():,}")

    if len(trainable) == 0:
        raise RuntimeError("No trainable params after unfreeze. Check gate/attn naming.")

    return trainable


def print_only_trainable(model: nn.Module, logger=None):
    def _emit(s):
        if logger is not None:
            logger.info(s)
        else:
            print(s)

    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    _emit("=" * 80)
    _emit(f"✅ Trainable tensors: {len(trainable)}")
    total_numel = sum(p.numel() for _, p in trainable)
    _emit(f"✅ Trainable elements total: {total_numel:,}")

    for name, p in trainable:
        _emit(f"[T] {name} | shape={tuple(p.shape)} | numel={p.numel():,} | dtype={p.dtype} | device={p.device}")
    _emit("=" * 80)

    if len(trainable) == 0:
        _emit("⚠️ No trainable parameters! (requires_grad=True is empty)")
        # 额外给你一点 debug 信息：有哪些模块有 gate
        _emit("---- Debug: modules that have attribute `gate` ----")
        for module_name, m in model.named_modules():
            if hasattr(m, "gate"):
                g = getattr(m, "gate")
                _emit(f"[gate] module={module_name} gate_type={type(g)}")
        _emit("---- Debug: any parameter names containing 'attn' ----")
        for n, p in model.named_parameters():
            if "attn" in n:
                _emit(f"[attn?] {n} | requires_grad={p.requires_grad}")

    return trainable


# ✅ 建议你在 create_optimizer 里这样用：
def create_optimizer(self, optimizer, kwargs):
    # 先执行你的 “全冻 + 解冻 gate/attn” 逻辑
    # unfreeze_gate_and_sep_attn(self.net, logger=self.logger)

    # 然后打印所有可训练参数（完整打印）
    trainable = print_only_trainable(self.net, logger=self.logger)

    trainable_params = [p for _, p in trainable]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable params -> optimizer would be empty. Check logs above.")

    opt = supported_optimizer[optimizer](trainable_params, **kwargs)
    return opt



class Trainer():
    '''
       Trainer of Conv-Tasnet
       input:
             net: load the Conv-Tasnet model
             checkpoint: save model path
             optimizer: name of opetimizer
             gpu_ids: (int/tuple) id of gpus
             optimizer_kwargs: the kwargs of optimizer
             clip_norm: maximum of clip norm, default: None
             min_lr: minimun of learning rate
             patience: Number of epochs with no improvement after which learning rate will be reduced
             factor: Factor by which the learning rate will be reduced. new_lr = lr * factor
             logging_period: How long to print
             resume: the kwargs of resume, including path of model, Whether to restart
             stop: Stop training cause no improvement
    '''

    def __init__(self,
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
                 num_epochs=100):
        # if the cuda is available and if the gpus' type is tuple
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device unavailable...exist")
        if not isinstance(gpuid, tuple):
            gpuid = (gpuid, )
        self.device = torch.device("cuda:{}".format(gpuid[0]))
        self.gpuid = gpuid

        # mkdir the file of Experiment path
        if checkpoint and not os.path.exists(checkpoint):
            os.makedirs(checkpoint)
        self.checkpoint = checkpoint

        # build the logger object
        self.logger = get_logger(
            os.path.join(checkpoint, "trainer.log"), file=True)
        self.clip_norm = clip_norm
        self.logging_period = logging_period
        self.cur_epoch = 0  # current epoch
        self.stop = stop

        # Whether to resume the model
        if resume['resume_state']:
            best_ckpt_path = os.path.join(resume['path'], self.checkpoint, 'best.pt')
            if os.path.exists(best_ckpt_path):
                # ===== 使用 resume 逻辑 =====
                cpt = torch.load(best_ckpt_path, map_location='cpu')
                self.cur_epoch = cpt['epoch']
                self.logger.info("Resume from checkpoint {}: epoch {:d}".format(
                    resume['path'], self.cur_epoch))
                net.load_state_dict(cpt['model_state_dict'])
                self.net = net.to(self.device)
                self.optimizer = self.create_optimizer(optimizer, optimizer_kwargs, state=cpt['optim_state_dict'])
            else:
                # 从wavLM encoder和baseline初始权重开始从头训练
                cpt = torch.load("/lustre/users/shi/datasets/librimix/ckpt_convtasnet/best.pt", map_location='cpu')
                missing, unexpected = net.load_state_dict(cpt['model_state_dict'], strict=False)
                self.logger.info("从baseline预训练权重中加载")
                from safetensors.torch import load_file
                state_dict = load_file("/lustre/users/shi/datasets/librimix/ckpt_convtasnet/encoder_1b_clean.safetensors")
                self.logger.info("从Wavlm预训练权重中加载")
                missing, unexpected = net.wavlm_encoder.load_state_dict(state_dict, strict=False)
                init_new_modules(net)
                print("Missing keys:", missing)
                print("Unexpected keys:", unexpected)
                self.net = net.to(self.device)
                self.optimizer = self.create_optimizer(optimizer, optimizer_kwargs)
        else:
            self.net = net.to(self.device)
            self.optimizer = self.create_optimizer(optimizer, optimizer_kwargs)
        # check model parameters
        self.param = check_parameters(self.net)

        # Reduce lr
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=patience, min_lr=min_lr)
            # self.optimizer, mode='min', factor=0.5, patience=patience, verbose=True, min_lr=min_lr)

        # logging
        self.logger.info("Starting preparing model ............")
        self.logger.info("Loading model to GPUs:{}, #param: {:.2f}M".format(
            self.gpuid, self.param))
        self.clip_norm = clip_norm
        # clip norm
        if clip_norm:
            self.logger.info(
                "Gradient clipping by {}, default L2".format(clip_norm))

        # number of epoch
        self.num_epochs = num_epochs
        self.mse = torch.nn.MSELoss()

    def create_optimizer(self, optimizer, kwargs, state=None):
        '''
           create optimizer
           optimizer: (str) name of optimizer
           kwargs: the kwargs of optimizer
           state: the load model optimizer state
        '''
        supported_optimizer = {
            "sgd": torch.optim.SGD,  # momentum, weight_decay, lr
            "rmsprop": torch.optim.RMSprop,  # momentum, weight_decay, lr
            "adam": torch.optim.Adam,  # weight_decay, lr
            "adadelta": torch.optim.Adadelta,  # weight_decay, lr
            "adagrad": torch.optim.Adagrad,  # lr, lr_decay, weight_decay
            "adamax": torch.optim.Adamax  # lr, weight_decay
        }
        if optimizer not in supported_optimizer:
            raise ValueError("Now only support optimizer {}".format(optimizer))
        ###################################################################
        # # 1. 冻结所有参数
        # for p in self.net.parameters():
        #     p.requires_grad = False

        # # 2. 解冻语义注入模块
        # semantic_params = self.net.get_semantic_parameters()
        # semantic_ids = set(id(p) for p in semantic_params)
        # for name, param in self.net.named_parameters():
        #     if id(param) in semantic_ids:
        #         param.requires_grad = True
        #         self.logger.info(f"Stage 1 可训练参数: {name}")

        # # 3. 构造优化器（只传 semantic_params）
        # opt = supported_optimizer[optimizer](semantic_params, **kwargs)

        # print(f"Stage 1 可训练参数量: {sum(p.numel() for p in semantic_params)/1e6:.2f} M")

        #######################################################################
        """
        # 1️⃣ 解冻所有参数
        for p in self.net.parameters():
            p.requires_grad = True

        # 2️⃣ 冻结不想训练的部分
        freeze_params = self.net.get_wavlmencoder_parameters()  # 这里举例，冻结纯音频路径
        freeze_ids = set(id(p) for p in freeze_params)

        for name, param in self.net.named_parameters():
            if id(param) in freeze_ids:
                param.requires_grad = False
                self.logger.info(f"冻结参数: {name}")

        # 3️⃣ 构造优化器（只传 requires_grad=True 的参数）
        trainable_params = [p for p in self.net.parameters() if p.requires_grad]
        opt = supported_optimizer[optimizer](trainable_params, **kwargs)

        print(f"可训练参数量: {sum(p.numel() for p in trainable_params)/1e6:.2f} M")
        """
        trainable_names = unfreeze_only_attn_gate_and_sep_attn(self.net, logger=self.logger)
        trainable_params = [p for p in self.net.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            raise RuntimeError("No trainable params after unfreeze. Check logs above.")
        print_only_trainable(self.net)

        opt = supported_optimizer[optimizer](trainable_params, **kwargs)
        #################### 全量训练###########################################
        # opt = supported_optimizer[optimizer](self.net.parameters(), **kwargs)
        #######################################################################
        
        # opt = supported_optimizer[optimizer](self.net.parameters(), **kwargs)
        self.logger.info("Create optimizer {0}: {1}".format(optimizer, kwargs))
        if state is not None:
            opt.load_state_dict(state)
            self.logger.info("Load optimizer state dict from checkpoint")
        return opt

    def save_checkpoint(self, best=True):
        '''
            save model
            best: the best model
        '''
        torch.save(
            {
                "epoch": self.cur_epoch,
                "model_state_dict": self.net.state_dict(),
                "optim_state_dict": self.optimizer.state_dict()
            },
            os.path.join(self.checkpoint,
                         "{0}.pt".format("best" if best else "last")))

    def train(self, train_dataloader):
        '''
           training model
        '''
        self.logger.info('Training model ......')
        losses = []
        start = time.time()
        current_step = 0
        for egs in train_dataloader:
            current_step += 1
            egs = to_device(egs, self.device)
            self.optimizer.zero_grad()
            ests = data_parallel(self.net, egs['mix'], device_ids=self.gpuid)
            loss = si_snr_loss(ests, egs)
            loss.backward()
            if self.clip_norm:
                clip_grad_norm_(self.net.parameters(), self.clip_norm)
            self.optimizer.step()
            losses.append(loss.item())
            if len(losses)%self.logging_period == 0:
                avg_loss = sum(
                    losses[-self.logging_period:])/self.logging_period
                self.logger.info('<epoch:{:3d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}, batch:{:d} utterances> '.format(
                    self.cur_epoch, current_step, self.optimizer.param_groups[0]['lr'], avg_loss, len(losses)))
                print('<epoch:{:3d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}, batch:{:d} utterances> '.format(
                    self.cur_epoch, current_step, self.optimizer.param_groups[0]['lr'], avg_loss, len(losses)))
        end = time.time()
        total_loss_avg = sum(losses)/len(losses)
        self.logger.info('<epoch:{:3d}, lr:{:.3e}, loss:{:.3f}, Total time:{:.3f} min> '.format(
            self.cur_epoch, self.optimizer.param_groups[0]['lr'], total_loss_avg, (end-start)/60))
        return total_loss_avg

    def val(self, val_dataloader):
        '''
           validation model
        '''
        self.logger.info('Validation model ......')
        self.net.eval()
        losses = []
        current_step = 0
        start = time.time()
        with torch.no_grad():
            for egs in val_dataloader:
                current_step += 1
                egs = to_device(egs, self.device)
                ests = data_parallel(self.net, egs['mix'], device_ids=self.gpuid)
                loss = si_snr_loss(ests, egs)
                losses.append(loss.item())
                if len(losses)%self.logging_period == 0:
                    avg_loss = sum(
                        losses[-self.logging_period:])/self.logging_period
                    self.logger.info('<epoch:{:3d}, iter:{:d}, lr:{:.3e}, loss:{:.3f}, batch:{:d} utterances> '.format(
                        self.cur_epoch, current_step, self.optimizer.param_groups[0]['lr'], avg_loss, len(losses)))
        end = time.time()
        total_loss_avg = sum(losses)/len(losses)
        self.logger.info('<epoch:{:3d}, lr:{:.3e}, loss:{:.3f}, Total time:{:.3f} min> '.format(
            self.cur_epoch, self.optimizer.param_groups[0]['lr'], total_loss_avg, (end-start)/60))
        return total_loss_avg

    def run(self, train_dataloader, val_dataloader):
        train_losses = []
        val_losses = []
        
        with torch.cuda.device(self.gpuid[0]):
            self.save_checkpoint(best=False)
            val_loss = self.val(val_dataloader)
            best_loss = val_loss
            self.logger.info("Starting epoch from {:d}, loss = {:.4f}".format(
                self.cur_epoch, best_loss))
            no_impr = 0

            self.scheduler.best = best_loss
            while self.cur_epoch < self.num_epochs:
                self.cur_epoch += 1
                train_loss = self.train(train_dataloader)
                val_loss = self.val(val_dataloader)

                train_losses.append(train_loss)
                val_losses.append(val_loss)

                if val_loss > best_loss:
                    no_impr += 1
                    self.logger.info('no improvement, best loss: {:.4f}'.format(self.scheduler.best))
                else:
                    best_loss = val_loss
                    no_impr = 0
                    self.save_checkpoint(best=True)
                    self.logger.info('Epoch: {:d}, now best loss change: {:.4f}'.format(self.cur_epoch,best_loss))
                # schedule here
                self.scheduler.step(val_loss)
                # save last checkpoint
                self.save_checkpoint(best=False)
                if no_impr == self.stop:
                    self.logger.info(
                        "Stop training cause no impr for {:d} epochs".format(
                            no_impr))
                    break
            self.logger.info("Training for {:d}/{:d} epoches done!".format(
                self.cur_epoch, self.num_epochs))
            
         # loss image
        plt.title("Loss of train and test")
        x = [i for i in range(self.cur_epoch)]
        plt.plot(x, train_losses, 'b-', label=u'train_loss',linewidth=0.8)
        plt.plot(x, val_losses, 'c-', label=u'val_loss',linewidth=0.8)
        plt.legend()
        #plt.xticks(l, lx)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.savefig('conv_tasnet_LRS.png')
