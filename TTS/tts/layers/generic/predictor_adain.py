import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import remove_spectral_norm


def get_num_adain_params(model):
    """
    input:
    - model: nn.module
    output:
    - num_adain_params: int
    """
    # return the number of AdaIN parameters needed by the model
    num_adain_params = 0
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm1d":
            num_adain_params += 2 * m.num_features
    return num_adain_params


def assign_adain_params(adain_params, model):

    """
    inputs:
    - adain_params: b x parameter_size
    - model: nn.module
    function:
    assign_adain_params
    """
    # assign the adain_params to the AdaIN layers in model
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm1d":
            mean = adain_params[:, : m.num_features]
            std = adain_params[:, m.num_features : 2 * m.num_features]
            m.bias = mean.contiguous().view(-1)
            m.weight = std.contiguous().view(-1)
            if adain_params.size(1) > 2 * m.num_features:
                adain_params = adain_params[:, 2 * m.num_features :]


class AdaptiveInstanceNorm1d(nn.Module):
    """Reference: https://github.com/microsoft/SpareNet/blob/876f37b99f77bb8d479d65acf32aebb4f4d6b1a8/models/sparenet_generator.py
    input:
    - inp: (b, c, m)
    output:
    - out: (b, c, m')
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ):
        super(AdaptiveInstanceNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped,
            running_mean,
            running_var,
            self.weight,
            self.bias,
            True,
            self.momentum,
            self.eps,
        )

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self.num_features) + ")"


class ZeroTemporalPad(nn.Module):
    """Pad sequences to equal lentgh in the temporal dimension"""

    def __init__(self, kernel_size, dilation):
        super().__init__()
        total_pad = dilation * (kernel_size - 1)
        begin = total_pad // 2
        end = total_pad - begin
        self.pad_layer = nn.ZeroPad2d((0, 0, begin, end))

    def forward(self, x):
        return self.pad_layer(x)


class Conv1dAdaIN(nn.Module):
    """1d convolutional with batch norm.
    conv1d -> relu -> AdaIN blocks.

    Note:
        Batch normalization is applied after ReLU regarding the original implementation.

    Args:
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
        kernel_size (int): kernel size for convolutional filters.
        dilation (int): dilation for convolution layers.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        padding = dilation * (kernel_size - 1)
        pad_s = padding // 2
        pad_e = padding - pad_s
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.pad = nn.ZeroPad2d((pad_s, pad_e, 0, 0))  # uneven left and right padding
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        o = self.conv1d(x)
        o = self.pad(o)
        o = nn.functional.relu(o)
        o = self.norm(o)
        return o

    def reset_parameters(self):
        self.conv1d.reset_parameters()


class Conv1dAdaINBlock(nn.Module):
    """1d convolutional block with batch norm. It is a set of conv1d -> relu -> AdaIN blocks.

    Args:
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
        hidden_channels (int): number of inner convolution channels.
        kernel_size (int): kernel size for convolutional filters.
        dilation (int): dilation for convolution layers.
        num_conv_blocks (int, optional): number of convolutional blocks. Defaults to 2.
    """

    def __init__(self, in_channels, out_channels, hidden_channels, kernel_size, dilation, num_conv_blocks=2):
        super().__init__()
        self.conv_bn_blocks = []
        for idx in range(num_conv_blocks):
            layer = Conv1dAdaIN(
                in_channels if idx == 0 else hidden_channels,
                out_channels if idx == (num_conv_blocks - 1) else hidden_channels,
                kernel_size,
                dilation,
            )
            self.conv_bn_blocks.append(layer)
        self.conv_bn_blocks = nn.Sequential(*self.conv_bn_blocks)

    def forward(self, x):
        """
        Shapes:
            x: (B, D, T)
        """
        return self.conv_bn_blocks(x)

    def reset_parameters(self):
        for layer in self.conv_bn_blocks.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()


class ResidualConv1dAdaINBlock(nn.Module):
    """Residual Convolutional Blocks with AdaIN
    Each block has 'num_conv_block' conv layers and 'num_res_blocks' such blocks are connected
    with residual connections.

    conv_block = (conv1d -> relu -> AdaIn) x 'num_conv_blocks'
    residuak_conv_block =  (x -> conv_block ->  + ->) x 'num_res_blocks'
                            ' - - - - - - - - - ^
    Args:
        in_channels (int): number of input channels.
        out_channels (int): number of output channels.
        hidden_channels (int): number of inner convolution channels.
        kernel_size (int): kernel size for convolutional filters.
        dilations (list): dilations for each convolution layer.
        style_dim (int, optional): number of style channels. Defaults to 64.
        num_res_blocks (int, optional): number of residual blocks. Defaults to 3.
        num_conv_blocks (int, optional): number of convolutional blocks in each residual block. Defaults to 2.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilations,
        style_dim=64,
        num_res_blocks=3,
        num_conv_blocks=2,
    ):

        super().__init__()
        assert len(dilations) == num_res_blocks
        self.style_dim = style_dim
        self.res_blocks = nn.ModuleList()
        for idx, dilation in enumerate(dilations):
            block = Conv1dAdaINBlock(
                in_channels if idx == 0 else hidden_channels,
                out_channels if (idx + 1) == len(dilations) else hidden_channels,
                hidden_channels,
                kernel_size,
                dilation,
                num_conv_blocks,
            )
            self.res_blocks.append(block)

        # MLP to generate AdaIN parameters
        self.mlp = nn.Sequential(
            nn.Linear(self.style_dim, self.style_dim),
            nn.ReLU(),
            nn.Linear(self.style_dim, get_num_adain_params(self.res_blocks)),
        )

    def forward(self, x, style, x_mask=None):
        # compute and apply AdaIn parameters
        adain_params = self.mlp(style)
        assign_adain_params(adain_params, self.res_blocks)

        if x_mask is None:
            x_mask = 1.0
        o = x * x_mask
        for block in self.res_blocks:
            res = o
            o = block(o)
            o = o + res
            if x_mask is not None:
                o = o * x_mask
        return o

    def reset_parameters(self):
        for layer in self.res_blocks.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        for layer in self.mlp.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()


class PredictorAdaIN(nn.Module):
    # AdaIn predictor inspired on StyleTTS paper: https://arxiv.org/pdf/2205.15439.pdf
    def __init__(
        self,
        in_dim,
        spk_emb_channels=0,
        emo_emb_channels=0,
        out_dim=1,
        hidden_channels=256,
        lstm_hidden_channels=512,
        num_conv_blocks=2,
        num_res_blocks=3,
        num_lstm_layers=3,
    ):
        super(PredictorAdaIN, self).__init__()
        in_channels = in_dim + in_dim
        self.num_lstm_layers = num_lstm_layers
        self.bilstm = BiLSTM(in_channels, lstm_hidden_channels, num_lstm_layers=num_lstm_layers)
        n_style_dim = 0
        if spk_emb_channels and emo_emb_channels:
            self.cond_spk_emb_proj = nn.Conv1d(spk_emb_channels, spk_emb_channels, 1)
            self.cond_emo_emb_proj = nn.Conv1d(emo_emb_channels, spk_emb_channels, 1)
            n_style_dim = spk_emb_channels + spk_emb_channels
        elif spk_emb_channels and not emo_emb_channels:
            n_style_dim = spk_emb_channels
        elif not spk_emb_channels and emo_emb_channels:
            n_style_dim = emo_emb_channels

        self.adain_blocks = ResidualConv1dAdaINBlock(
            hidden_channels,
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            num_res_blocks=num_res_blocks,
            num_conv_blocks=num_conv_blocks,
            dilations=[1] * num_res_blocks,
            style_dim=n_style_dim,
        )

        # projection to match the shape if neeed
        if hidden_channels != lstm_hidden_channels:
            self.projection = nn.Conv1d(
                lstm_hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=int((3 - 1) / 2),
            )
        else:
            self.projection = None

        self.cond_inp_proj = nn.Conv1d(n_style_dim, in_dim, 1)

        self.dense = nn.Linear(hidden_channels, out_dim)

    def forward(self, context, lens, spk_emb=None, emo_emb=None):
        cond = None
        # guarantee that emotion and speaker embedding have the same shape
        if spk_emb is not None and emo_emb is not None:
            spk_emb = self.cond_spk_emb_proj(spk_emb)
            emo_emb = self.cond_emo_emb_proj(emo_emb)
            cond = torch.cat((spk_emb, emo_emb), 1)
        elif spk_emb is not None and emo_emb is None:
            cond = spk_emb
        elif spk_emb is None and emo_emb is not None:
            cond = emo_emb

        if cond is not None:
            cond_emb = self.cond_inp_proj(cond)
            cond_emb_expanded = cond_emb.expand(-1, -1, context.shape[2])
            context = torch.cat((context, cond_emb_expanded), 1)

        # bilstm
        x = self.bilstm(context, lens)

        # projection to match the shape
        if self.projection:
            x = self.projection(x)

        # adain residual
        x = self.adain_blocks(x, cond.squeeze(-1))

        out = self.dense(x.transpose(1, 2)).transpose(1, 2)
        return out

    def reset_parameters(self):
        self.bilstm.reset_parameters()
        self.adain_blocks.reset_parameters()
        self.cond_proj.reset_parameters()
        self.dense.reset_parameters()
        if self.projection:
            self.projection.reset_parameters()


class BiLSTM(nn.Module):
    def __init__(self, in_dim, n_channels=512, num_lstm_layers=1):
        super(BiLSTM, self).__init__()
        self.num_lstm_layers = num_lstm_layers
        lstm_channels = int(n_channels // 2)
        if self.num_lstm_layers == 1:
            self.bilstm = nn.LSTM(in_dim, lstm_channels, 1, batch_first=True, bidirectional=True)
            lstm_norm_fn_pntr = nn.utils.spectral_norm

            self.bilstm = lstm_norm_fn_pntr(self.bilstm, "weight_hh_l0")
            self.bilstm = lstm_norm_fn_pntr(self.bilstm, "weight_hh_l0_reverse")
        else:
            self.bilstm = []
            num_channels = (lstm_channels * 2) // (2**self.num_lstm_layers)
            for i in range(self.num_lstm_layers):
                if i == 0:
                    layer = nn.LSTM(in_dim, num_channels, 1, batch_first=True, bidirectional=True)
                else:
                    layer = nn.LSTM(num_channels, num_channels, 1, batch_first=True, bidirectional=True)

                num_channels = num_channels * 2
                lstm_norm_fn_pntr = nn.utils.spectral_norm
                layer = lstm_norm_fn_pntr(layer, "weight_hh_l0")
                layer = lstm_norm_fn_pntr(layer, "weight_hh_l0_reverse")
                self.bilstm.append(layer)
            self.bilstm = nn.Sequential(*self.bilstm)

    def run_unsorted_inputs(self, fn, context, lens):
        lens_sorted, ids_sorted = torch.sort(lens, descending=True)
        unsort_ids = [0] * lens.size(0)
        for i in range(len(ids_sorted)):
            unsort_ids[ids_sorted[i]] = i
        lens_sorted = lens_sorted.long().cpu()

        context = context[ids_sorted]
        context = nn.utils.rnn.pack_padded_sequence(context, lens_sorted, batch_first=True)
        context = fn(context)[0]
        context = nn.utils.rnn.pad_packed_sequence(context, batch_first=True)[0]

        # map back to original indices
        context = context[unsort_ids]
        return context

    def forward(self, context, lens):
        context = context.transpose(1, 2)
        if self.num_lstm_layers == 1:
            self.bilstm.flatten_parameters()
            if lens is not None:
                context = self.run_unsorted_inputs(self.bilstm, context, lens)
            else:
                context = self.bilstm(context)[0]
        else:
            for layer in self.bilstm:
                layer.flatten_parameters()
                if lens is not None:
                    context = self.run_unsorted_inputs(layer, context, lens)
                else:
                    context = layer(context)[0]

        return context.transpose(1, 2)

    def reset_parameters(self):
        if self.num_lstm_layers == 1:
            self.bilstm.reset_parameters()
        else:
            for layer in self.bilstm:
                layer.reset_parameters()

    def remove_spectral_norm(self):
        if self.num_lstm_layers == 1:
            remove_spectral_norm(self.bilstm, name="weight_hh_l0")
            remove_spectral_norm(self.bilstm, name="weight_hh_l0_reverse")
            self.bilstm.flatten_parameters()
        else:
            for layer in self.bilstm:
                remove_spectral_norm(layer, name="weight_hh_l0")
                remove_spectral_norm(layer, name="weight_hh_l0_reverse")
                layer.flatten_parameters()
