# -*- coding: utf-8 -*-
import datetime
import importlib
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict

import fsspec
import torch


def to_cuda(x: torch.Tensor) -> torch.Tensor:
    if x is None:
        return None
    if torch.is_tensor(x):
        x = x.contiguous()
        if torch.cuda.is_available():
            x = x.cuda(non_blocking=True)
    return x


def get_cuda():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return use_cuda, device


def get_git_branch():
    try:
        out = subprocess.check_output(["git", "branch"]).decode("utf8")
        current = next(line for line in out.split("\n") if line.startswith("*"))
        current.replace("* ", "")
    except subprocess.CalledProcessError:
        current = "inside_docker"
    except FileNotFoundError:
        current = "unknown"
    return current


def get_commit_hash():
    """https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script"""
    # try:
    #     subprocess.check_output(['git', 'diff-index', '--quiet',
    #                              'HEAD'])  # Verify client is clean
    # except:
    #     raise RuntimeError(
    #         " !! Commit before training to get the commit hash.")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    # Not copying .git folder into docker container
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "0000000"
    return commit


def get_experiment_folder_path(root_path, model_name):
    """Get an experiment folder path with the current date and time"""
    date_str = datetime.datetime.now().strftime("%B-%d-%Y_%I+%M%p")
    commit_hash = get_commit_hash()
    output_folder = os.path.join(root_path, model_name + "-" + date_str + "-" + commit_hash)
    return output_folder


def remove_experiment_folder(experiment_path):
    """Check folder if there is a checkpoint, otherwise remove the folder"""
    fs = fsspec.get_mapper(experiment_path).fs
    checkpoint_files = fs.glob(experiment_path + "/*.pth.tar")
    if not checkpoint_files:
        if fs.exists(experiment_path):
            fs.rm(experiment_path, recursive=True)
            print(" ! Run is removed from {}".format(experiment_path))
    else:
        print(" ! Run is kept in {}".format(experiment_path))


def count_parameters(model):
    r"""Count number of trainable parameters in a network"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_camel(text):
    text = text.capitalize()
    text = re.sub(r"(?!^)_([a-zA-Z])", lambda m: m.group(1).upper(), text)
    text = text.replace("Tts", "TTS")
    return text


def find_module(module_path: str, module_name: str) -> object:
    module_name = module_name.lower()
    print(module)
    module = importlib.import_module(module_path + "." + module_name)
    class_name = to_camel(module_name)
    return getattr(module, class_name)


def get_user_data_dir(appname):
    if sys.platform == "win32":
        import winreg  # pylint: disable=import-outside-toplevel

        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders"
        )
        dir_, _ = winreg.QueryValueEx(key, "Local AppData")
        ans = Path(dir_).resolve(strict=False)
    elif sys.platform == "darwin":
        ans = Path("~/Library/Application Support/").expanduser()
    else:
        ans = Path.home().joinpath(".local/share")
    return ans.joinpath(appname)


def set_init_dict(model_dict, checkpoint_state, c):
    # Partial initialization: if there is a mismatch with new and old layer, it is skipped.
    for k, v in checkpoint_state.items():
        if k not in model_dict:
            print(" | > Layer missing in the model definition: {}".format(k))
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in checkpoint_state.items() if k in model_dict}
    # 2. filter out different size layers
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if v.numel() == model_dict[k].numel()}
    # 3. skip reinit layers
    if c.has("reinit_layers") and c.reinit_layers is not None:
        for reinit_layer_name in c.reinit_layers:
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if reinit_layer_name not in k}
    # 4. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    print(" | > {} / {} layers are restored.".format(len(pretrained_dict), len(model_dict)))
    return model_dict


def format_aux_input(def_args: Dict, kwargs: Dict) -> Dict:
    """Format kwargs to hande auxilary inputs to models.

    Args:
        def_args (Dict): A dictionary of argument names and their default values if not defined in `kwargs`.
        kwargs (Dict): A `dict` or `kwargs` that includes auxilary inputs to the model.

    Returns:
        Dict: arguments with formatted auxilary inputs.
    """
    for name in def_args:
        if name not in kwargs:
            kwargs[def_args[name]] = None
    return kwargs


class KeepAverage:
    def __init__(self):
        self.avg_values = {}
        self.iters = {}

    def __getitem__(self, key):
        return self.avg_values[key]

    def items(self):
        return self.avg_values.items()

    def add_value(self, name, init_val=0, init_iter=0):
        self.avg_values[name] = init_val
        self.iters[name] = init_iter

    def update_value(self, name, value, weighted_avg=False):
        if name not in self.avg_values:
            # add value if not exist before
            self.add_value(name, init_val=value)
        else:
            # else update existing value
            if weighted_avg:
                self.avg_values[name] = 0.99 * self.avg_values[name] + 0.01 * value
                self.iters[name] += 1
            else:
                self.avg_values[name] = self.avg_values[name] * self.iters[name] + value
                self.iters[name] += 1
                self.avg_values[name] /= self.iters[name]

    def add_values(self, name_dict):
        for key, value in name_dict.items():
            self.add_value(key, init_val=value)

    def update_values(self, value_dict):
        for key, value in value_dict.items():
            self.update_value(key, value)
            
            

import math

import torch
from torch.optim import Optimizer


class Lamb(Optimizer):
    """Implements a pure pytorch variant of FuseLAMB (NvLamb variant) optimizer from apex.optimizers.FusedLAMB
    reference: https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/Transformer-XL/pytorch/lamb.py
    LAMB was proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its norm. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        grad_averaging (bool, optional): whether apply (1-beta2) to grad when
            calculating running averages of gradient. (default: True)
        max_grad_norm (float, optional): value used to clip global grad norm (default: 1.0)
        trust_clip (bool): enable LAMBC trust ratio clipping (default: False)
        always_adapt (boolean, optional): Apply adaptive learning rate to 0.0
            weight decay parameter (default: False)
    .. _Large Batch Optimization for Deep Learning - Training BERT in 76 minutes:
        https://arxiv.org/abs/1904.00962
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(
            self, params, lr=1e-3, bias_correction=True, betas=(0.9, 0.999), eps=1e-6,
            weight_decay=0.01, grad_averaging=True, max_grad_norm=1.0, trust_clip=False, always_adapt=False):
        defaults = dict(
            lr=lr, bias_correction=bias_correction, betas=betas, eps=eps, weight_decay=weight_decay,
            grad_averaging=grad_averaging, max_grad_norm=max_grad_norm,
            trust_clip=trust_clip, always_adapt=always_adapt)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        device = self.param_groups[0]['params'][0].device
        one_tensor = torch.tensor(1.0, device=device)  # because torch.where doesn't handle scalars correctly
        global_grad_norm = torch.zeros(1, device=device)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Lamb does not support sparse gradients, consider SparseAdam instad.')
                global_grad_norm.add_(grad.pow(2).sum())

        global_grad_norm = torch.sqrt(global_grad_norm)
        # FIXME it'd be nice to remove explicit tensor conversion of scalars when torch.where promotes
        # scalar types properly https://github.com/pytorch/pytorch/issues/9190
        max_grad_norm = torch.tensor(self.defaults['max_grad_norm'], device=device)
        clip_global_grad_norm = torch.where(
            global_grad_norm > max_grad_norm,
            global_grad_norm / max_grad_norm,
            one_tensor)

        for group in self.param_groups:
            bias_correction = 1 if group['bias_correction'] else 0
            beta1, beta2 = group['betas']
            grad_averaging = 1 if group['grad_averaging'] else 0
            beta3 = 1 - beta1 if grad_averaging else 1.0

            # assume same step across group now to simplify things
            # per parameter step can be easily support by making it tensor, or pass list into kernel
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            if bias_correction:
                bias_correction1 = 1 - beta1 ** group['step']
                bias_correction2 = 1 - beta2 ** group['step']
            else:
                bias_correction1, bias_correction2 = 1.0, 1.0

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.div_(clip_global_grad_norm)
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient valuesa
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=beta3)  # m_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)  # v_t

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                update = (exp_avg / bias_correction1).div_(denom)

                weight_decay = group['weight_decay']
                if weight_decay != 0:
                    update.add_(p, alpha=weight_decay)

                if weight_decay != 0 or group['always_adapt']:
                    # Layer-wise LR adaptation. By default, skip adaptation on parameters that are
                    # excluded from weight decay, unless always_adapt == True, then always enabled.
                    w_norm = p.norm(2.0)
                    g_norm = update.norm(2.0)
                    # FIXME nested where required since logical and/or not working in PT XLA
                    trust_ratio = torch.where(
                        w_norm > 0,
                        torch.where(g_norm > 0, w_norm / g_norm, one_tensor),
                        one_tensor,
                    )
                    if group['trust_clip']:
                        # LAMBC trust clipping, upper bound fixed at one
                        trust_ratio = torch.minimum(trust_ratio, one_tensor)
                    update.mul_(trust_ratio)

                p.add_(update, alpha=-group['lr'])

        return loss

