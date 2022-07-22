import math

import torch
from torch import nn

from TTS.tts.layers.glow_tts.glow import WN
from TTS.tts.layers.glow_tts.transformer import RelativePositionTransformer
from TTS.tts.utils.helpers import sequence_mask

LRELU_SLOPE = 0.1


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class TextEncoder(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        out_channels: int,
        hidden_channels: int,
        hidden_channels_ffn: int,
        num_heads: int,
        num_layers: int,
        kernel_size: int,
        dropout_p: float,
        language_emb_dim: int = None,
        emo_emb_dim: int = None,
    ):
        """Text Encoder for VITS model.

        Args:
            n_vocab (int): Number of characters for the embedding layer.
            out_channels (int): Number of channels for the output.
            hidden_channels (int): Number of channels for the hidden layers.
            hidden_channels_ffn (int): Number of channels for the convolutional layers.
            num_heads (int): Number of attention heads for the Transformer layers.
            num_layers (int): Number of Transformer layers.
            kernel_size (int): Kernel size for the FFN layers in Transformer network.
            dropout_p (float): Dropout rate for the Transformer layers.
        """
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        self.emb = nn.Embedding(n_vocab, hidden_channels)

        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        if language_emb_dim:
            hidden_channels += language_emb_dim

        if emo_emb_dim:
            hidden_channels += emo_emb_dim

        self.encoder = RelativePositionTransformer(
            in_channels=hidden_channels,
            out_channels=self.out_channels,
            hidden_channels=hidden_channels,
            hidden_channels_ffn=hidden_channels_ffn,
            num_heads=num_heads,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout_p=dropout_p,
            layer_norm_type="2",
            rel_attn_window_size=4,
        )
        self.proj = nn.Conv1d(self.out_channels, self.out_channels * 2, 1)

    def forward(self, x, x_lengths, lang_emb=None, emo_emb=None):
        """
        Shapes:
            - x: :math:`[B, T]`
            - x_length: :math:`[B]`
        """
        assert x.shape[0] == x_lengths.shape[0]
        x_emb = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]
        x_emb = torch.transpose(x_emb, 1, 2)  # [b, h, t]

        # concat the lang emb in embedding chars
        x_enc = x_emb
        if lang_emb is not None:
            x_enc = torch.cat((x_enc, lang_emb.expand(x_enc.size(0), -1, x_enc.size(2))), dim=1)

        if emo_emb is not None:
            x_enc = torch.cat((x_enc, emo_emb.expand(x_enc.size(0), -1, x_enc.size(2))), dim=1)

        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x_enc.size(2)), 1).to(x_emb.dtype)  # [b, 1, t]

        o_en = self.encoder(x_enc * x_mask, x_mask)
        stats = self.proj(o_en) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x_emb, o_en, m, logs, x_mask

class ContextEncoder(nn.Module):
    def __init__(self, in_channels, cond_channels=0, spk_emb_channels=0, emo_emb_channels=0, num_lstm_layers=1, lstm_norm="spectral"):
        super().__init__()

        in_lstm_channels = spk_emb_channels + in_channels
        hidden_lstm_channels = int((spk_emb_channels + in_channels) / 2)

        if cond_channels > 0:
            in_lstm_channels = cond_channels + in_channels + spk_emb_channels
            hidden_lstm_channels = in_lstm_channels // 2

        if emo_emb_channels > 0:
            in_lstm_channels += emo_emb_channels

        self.hidden_lstm_channels = hidden_lstm_channels

        self.lstm = torch.nn.LSTM(
            input_size=in_lstm_channels,
            hidden_size=hidden_lstm_channels,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
        )

        if lstm_norm is not None:
            if "spectral" in lstm_norm:
                lstm_norm_fn = torch.nn.utils.spectral_norm
            elif "weight" in lstm_norm:
                lstm_norm_fn = torch.nn.utils.weight_norm

        self.lstm = lstm_norm_fn(self.lstm, "weight_hh_l0")
        self.lstm = lstm_norm_fn(self.lstm, "weight_hh_l0_reverse")

    def forward(self, x, x_len, spk_emb=None, emo_emb=None, cond=None):
        spk_emb = spk_emb.expand(-1, -1, x.shape[2])
        context_w_spk_emb = torch.cat((x, spk_emb), 1)
        if emo_emb is not None:
            emo_emb = emo_emb.expand(-1, -1, x.shape[2])
            context_w_spk_emb = torch.cat((context_w_spk_emb, emo_emb), 1)
        if cond is not None:
            context_w_spk_emb = torch.cat((context_w_spk_emb, cond), 1)
        unfolded_out_lens_packed = nn.utils.rnn.pack_padded_sequence(
            context_w_spk_emb.transpose(1, 2), x_len.to('cpu'), batch_first=True, enforce_sorted=False
        )
        self.lstm.flatten_parameters()
        context, _ = self.lstm(unfolded_out_lens_packed)
        context, _ = nn.utils.rnn.pad_packed_sequence(context, batch_first=True)
        context = context.transpose(1, 2)
        return context

class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        num_layers,
        dropout_p=0,
        cond_channels=0,
        mean_only=False,
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.half_channels = channels // 2
        self.mean_only = mean_only
        # input layer
        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        # coupling layers
        self.enc = WN(
            hidden_channels,
            hidden_channels,
            kernel_size,
            dilation_rate,
            num_layers,
            dropout_p=dropout_p,
            c_in_channels=cond_channels,
        )
        # output layer
        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        """
        Note:
            Set `reverse` to True for inference.

        Shapes:
            - x: :math:`[B, C, T]`
            - x_mask: :math:`[B, 1, T]`
            - g: :math:`[B, C, 1]`
        """
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, log_scale = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            log_scale = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(log_scale) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(log_scale, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-log_scale) * x_mask
            x = torch.cat([x0, x1], 1)
            return x


class ResidualCouplingBlocks(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        num_layers: int,
        num_flows=4,
        cond_channels=0,
    ):
        """Redisual Coupling blocks for VITS flow layers.

        Args:
            channels (int): Number of input and output tensor channels.
            hidden_channels (int): Number of hidden network channels.
            kernel_size (int): Kernel size of the WaveNet layers.
            dilation_rate (int): Dilation rate of the WaveNet layers.
            num_layers (int): Number of the WaveNet layers.
            num_flows (int, optional): Number of Residual Coupling blocks. Defaults to 4.
            cond_channels (int, optional): Number of channels of the conditioning tensor. Defaults to 0.
        """
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.num_layers = num_layers
        self.num_flows = num_flows
        self.cond_channels = cond_channels

        self.flows = nn.ModuleList()
        for _ in range(num_flows):
            self.flows.append(
                ResidualCouplingBlock(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    num_layers,
                    cond_channels=cond_channels,
                    mean_only=True,
                )
            )

    def forward(self, x, x_mask, g=None, reverse=False):
        """
        Note:
            Set `reverse` to True for inference.

        Shapes:
            - x: :math:`[B, C, T]`
            - x_mask: :math:`[B, 1, T]`
            - g: :math:`[B, C, 1]`
        """
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
                x = torch.flip(x, [1])
        else:
            for flow in reversed(self.flows):
                x = torch.flip(x, [1])
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        num_layers: int,
        cond_channels=0,
    ):
        """Posterior Encoder of VITS model.

        ::
            x -> conv1x1() -> WaveNet() (non-causal) -> conv1x1() -> split() -> [m, s] -> sample(m, s) -> z

        Args:
            in_channels (int): Number of input tensor channels.
            out_channels (int): Number of output tensor channels.
            hidden_channels (int): Number of hidden channels.
            kernel_size (int): Kernel size of the WaveNet convolution layers.
            dilation_rate (int): Dilation rate of the WaveNet layers.
            num_layers (int): Number of the WaveNet layers.
            cond_channels (int, optional): Number of conditioning tensor channels. Defaults to 0.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.num_layers = num_layers
        self.cond_channels = cond_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels, hidden_channels, kernel_size, dilation_rate, num_layers, c_in_channels=cond_channels
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        """
        Shapes:
            - x: :math:`[B, C, T]`
            - x_lengths: :math:`[B, 1]`
            - g: :math:`[B, C, 1]`
        """
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        mean, log_scale = torch.split(stats, self.out_channels, dim=1)
        z = (mean + torch.randn_like(mean) * torch.exp(log_scale)) * x_mask
        return z, mean, log_scale, x_mask
