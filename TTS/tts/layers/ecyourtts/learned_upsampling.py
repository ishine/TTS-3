import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from TTS.tts.utils.helpers import sequence_mask

from TTS.tts.layers.generic.normalization import LayerNorm

class RangePredictor(nn.Module):
    """
    Range Predictor module as in https://arxiv.org/pdf/2010.04301.pdf

    Model::
        x -> 2 x BiLSTM -> Linear -> o
    """

    def __init__(self, in_channels, hidden_channels, kernel_size, dropout_p, cond_channels=None, language_emb_dim=None):
        super().__init__()

        # add language embedding dim in the input
        if language_emb_dim:
            in_channels += language_emb_dim

        # class arguments
        self.in_channels = in_channels
        self.filter_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dropout_p = dropout_p
        # layers
        self.drop = nn.Dropout(dropout_p)
        self.conv_1 = nn.Conv1d(in_channels + 1, hidden_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = LayerNorm(hidden_channels)
        self.conv_2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = LayerNorm(hidden_channels)
        # output layer
        self.proj = nn.Conv1d(hidden_channels, 1, 1)
        if cond_channels is not None and cond_channels != 0:
            self.cond = nn.Conv1d(cond_channels, in_channels, 1)

        if language_emb_dim != 0 and language_emb_dim is not None:
            self.cond_lang = nn.Conv1d(language_emb_dim, in_channels, 1)

    def forward(self, x, x_mask, dr, g=None, lang_emb=None):
        """
        Shapes:
            - x: :math:`[B, C, T]`
            - x_mask: :math:`[B, 1, T]`
            - g: :math:`[B, C, 1]`
        """
        if g is not None:
            x = x + self.cond(g)

        if lang_emb is not None:
            x = x + self.cond_lang(lang_emb)

        x = torch.cat([x, dr.unsqueeze(1)], dim=1)

        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x)
        o = F.softplus(x.squeeze(1))
        return o  # [B, T]


class GaussianUpsampling(nn.Module):
    """
    Gaussian Upsampling for duration regulation as in Non-attention Tacotron https://arxiv.org/abs/2010.04301
    This code is from https://github.com/BridgetteSong/ExpressiveTacotron/blob/master/model_duration.py
    """

    def __init__(self):
        super(GaussianUpsampling, self).__init__()
        self.mask_score = -1e15

    def forward(self, x, durations, vars, x_lengths=None):
        """
        Gaussian upsampling
        Shapes:
            - x:  [B, C, T]
            - durations: [B, T]
            - vars : [B, T]
            - x_lengths : [B]
            - encoder_upsampling_outputs: [B, T, C]
        """
        x = x.transpose(1, 2)
        B = x.size(0)
        N = x.size(1)
        T = int(torch.sum(durations, dim=1).max().item())
        c = torch.cumsum(durations, dim=1).float() - 0.5 * durations
        c = c.unsqueeze(2)  # [B, N, 1]
        t = torch.arange(T, device=x.device).expand(B, N, T).float()  # [B, N, T]
        vars = vars.view(B, -1, 1)  # [B, N, 1]

        w_t = -0.5 * (np.log(2.0 * np.pi) + torch.log(vars) + torch.pow(t - c, 2) / vars)  # [B, N, T]

        if x_lengths is not None:
            input_masks = ~sequence_mask(x_lengths, None)  # [B, T, 1]
            masks = input_masks.unsqueeze(2)
            w_t.data.masked_fill_(masks, self.mask_score)
        w_t = F.softmax(w_t, dim=1)

        encoder_upsampling_outputs = torch.bmm(w_t.transpose(1, 2), x)  # [B, T, C]
        return encoder_upsampling_outputs.transpose(1, 2)
