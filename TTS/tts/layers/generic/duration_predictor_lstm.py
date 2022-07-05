import torch
from torch import nn
from torch.nn import functional as F


class BottleneckLayerLayer(nn.Module):
    def __init__(self, in_dim, reduction_factor, norm='weightnorm',
                 non_linearity='relu', kernel_size=3, use_partial_padding=False):
        super(BottleneckLayerLayer, self).__init__()

        self.reduction_factor = reduction_factor
        reduced_dim = int(in_dim / reduction_factor)
        self.out_dim = reduced_dim
        if self.reduction_factor > 1:
            fn = ConvNorm(in_dim, reduced_dim, kernel_size=kernel_size,
                          use_weight_norm=(norm == 'weightnorm'))
            if norm == 'instancenorm':
                fn = nn.Sequential(
                    fn, nn.InstanceNorm1d(reduced_dim, affine=True))

            self.projection_fn = fn
            self.non_linearity = nn.ReLU()
            if non_linearity == 'leakyrelu':
                self.non_linearity= nn.LeakyReLU()

    def forward(self, x):
        if self.reduction_factor > 1:
            x = self.projection_fn(x)
            x = self.non_linearity(x)
        return x


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear',
                 use_weight_norm=False):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.use_weight_norm = use_weight_norm
        conv_fn = torch.nn.Conv1d
        self.conv = conv_fn(in_channels, out_channels,
                            kernel_size=kernel_size, stride=stride,
                            padding=padding, dilation=dilation,
                            bias=bias)
        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))
        if self.use_weight_norm:
            self.conv = nn.utils.weight_norm(self.conv)

    def forward(self, signal, mask=None):
        conv_signal = self.conv(signal)
        if mask is not None:
            # always re-zero output if mask is
            # available to match zero-padding
            conv_signal = conv_signal * mask
        return conv_signal



class DurationPredictorLSTM(nn.Module):
    def __init__(self, in_dim, n_speaker_dim,  out_dim=1, reduction_factor=4):
        super(DurationPredictorLSTM, self).__init__()
        self.bottleneck_layer = BottleneckLayerLayer(in_dim, reduction_factor, norm='weightnorm', non_linearity='relu', kernel_size=3, use_partial_padding=False)
        self.feat_pred_fn = ConvLSTMLinear(self.bottleneck_layer.out_dim + in_dim, out_dim, n_layers=2, n_channels=256, kernel_size=3, p_dropout=0.1, lstm_type='bilstm', use_linear=True)
        self.spk_proj = nn.Conv1d(n_speaker_dim, in_dim, 1)

    def forward(self, txt_enc, spk_emb, lens):
        txt_enc = self.bottleneck_layer(txt_enc)
        spk_emb = self.spk_proj(spk_emb)
        spk_emb_expanded = spk_emb.expand(-1, -1, txt_enc.shape[2])
        context = torch.cat((txt_enc, spk_emb_expanded), 1)
        x_hat = self.feat_pred_fn(context, lens)
        return x_hat


class ConvLSTMLinear(nn.Module):
    def __init__(self, in_dim, out_dim, n_layers=2, n_channels=256,
                 kernel_size=3, p_dropout=0.1, lstm_type='bilstm',
                 use_linear=True):
        super(ConvLSTMLinear, self).__init__()
        self.out_dim = out_dim
        self.lstm_type = lstm_type
        self.use_linear = use_linear
        self.dropout = nn.Dropout(p=p_dropout)

        convolutions = []
        for i in range(n_layers):
            conv_layer = ConvNorm(
                in_dim if i == 0 else n_channels, n_channels,
                kernel_size=kernel_size, stride=1,
                padding=int((kernel_size - 1) / 2), dilation=1,
                w_init_gain='relu')
            conv_layer = torch.nn.utils.weight_norm(
                conv_layer.conv, name='weight')
            convolutions.append(conv_layer)

        self.convolutions = nn.ModuleList(convolutions)

        if not self.use_linear:
            n_channels = out_dim

        if self.lstm_type != '':
            use_bilstm = False
            lstm_channels = n_channels
            if self.lstm_type == 'bilstm':
                use_bilstm = True
                lstm_channels = int(n_channels // 2)

            self.bilstm = nn.LSTM(n_channels, lstm_channels, 1,
                                  batch_first=True, bidirectional=use_bilstm)
            lstm_norm_fn_pntr = nn.utils.spectral_norm
            self.bilstm = lstm_norm_fn_pntr(self.bilstm, 'weight_hh_l0')
            if self.lstm_type == 'bilstm':
                self.bilstm = lstm_norm_fn_pntr(self.bilstm, 'weight_hh_l0_reverse')

        if self.use_linear:
            self.dense = nn.Linear(n_channels, out_dim)

    def run_padded_sequence(self, context, lens):
        context_embedded = []
        for b_ind in range(context.size()[0]):  # TODO: speed up
            curr_context = context[b_ind:b_ind+1, :, :lens[b_ind]].clone()
            for conv in self.convolutions:
                curr_context = self.dropout(F.relu(conv(curr_context)))
            context_embedded.append(curr_context[0].transpose(0, 1))
        context = torch.nn.utils.rnn.pad_sequence(
            context_embedded, batch_first=True)
        return context

    def run_unsorted_inputs(self, fn, context, lens):
        lens_sorted, ids_sorted = torch.sort(lens, descending=True)
        unsort_ids = [0] * lens.size(0)
        for i in range(len(ids_sorted)):
            unsort_ids[ids_sorted[i]] = i
        lens_sorted = lens_sorted.long().cpu()

        context = context[ids_sorted]
        context = nn.utils.rnn.pack_padded_sequence(
            context, lens_sorted, batch_first=True)
        context = fn(context)[0]
        context = nn.utils.rnn.pad_packed_sequence(
            context, batch_first=True)[0]

        # map back to original indices
        context = context[unsort_ids]
        return context

    def forward(self, context, lens):
        if context.size()[0] > 1:
            context = self.run_padded_sequence(context, lens)
            # to B, D, T
            context = context.transpose(1, 2)
        else:
            for conv in self.convolutions:
                context = self.dropout(F.relu(conv(context)))

        if self.lstm_type != '':
            context = context.transpose(1, 2)
            self.bilstm.flatten_parameters()
            if lens is not None:
                context = self.run_unsorted_inputs(self.bilstm, context, lens)
            else:
                context = self.bilstm(context)[0]
            context = context.transpose(1, 2)

        x_hat = context
        if self.use_linear:
            x_hat = self.dense(context.transpose(1, 2)).transpose(1, 2)

        return x_hat
