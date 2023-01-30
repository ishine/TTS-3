import torch
import torch.nn.functional as F
from torch import nn
import math
import torch.nn.modules.conv as conv

class FineGrainedReferenceEncoder(nn.Module):
    '''
    Based on https://github.com/keonlee9420/Robust_Fine_Grained_Prosody_Control

    embedded_text --- [N, seq_len, encoder_embedding_dim]
    mels --- [N, n_mels*r, Ty/r], r=1
    style_embed --- [N, seq_len, prosody_embedding_dim]
    alignments --- [N, seq_len, ref_len], Ty/r = ref_len
    '''
    def __init__(self, num_mel, embedding_dim, prosody_embedding_dim, encoder_embedding_dim, fg_attention_dropout, fg_attention_dim, use_nonlinear_proj=False):
        super(FineGrainedReferenceEncoder, self).__init__()
        self.prosody_embedding_dim = embedding_dim
        self.encoder = ModifiedReferenceEncoder(num_mel, embedding_dim, use_nonlinear_proj)
        self.ref_attn = ScaledDotProductAttention(prosody_embedding_dim, encoder_embedding_dim, fg_attention_dropout, fg_attention_dim)
        self.encoder_bottleneck = nn.Linear(embedding_dim, prosody_embedding_dim * 2)

    def forward(self, embedded_text, text_lengths, mels, mels_lengths):
        embedded_prosody, _ = self.encoder(mels)

        print(f'entry shapes: {embedded_text.shape}, {text_lengths.shape}, {mels.shape}, {mels_lengths.shape}')

        # Bottleneck
        embedded_prosody = self.encoder_bottleneck(embedded_prosody)
        print(embedded_prosody.shape)
        # Obtain k and v from prosody embedding
        key, value = torch.split(embedded_prosody, self.prosody_embedding_dim, dim=-1) # [N, Ty, prosody_embedding_dim] * 2

        # Get attention mask
        text_mask = get_mask_from_lengths(text_lengths).float().unsqueeze(-1) # [B, seq_len, 1]
        mels_mask = get_mask_from_lengths(mels_lengths).float().unsqueeze(-1) # [B, req_len, 1]
        attn_mask = torch.bmm(text_mask, mels_mask.transpose(-2, -1)) # [N, seq_len, ref_len]
        # print(text_mask.shape, mels_mask.shape, attn_mask.shape, embedded_text.shape, key.shape, value.shape)
        # Attention
        style_embed, alignments = self.ref_attn(embedded_text, key, value, attn_mask)

        # Apply ReLU as the activation function to force the values of the prosody embedding to lie in [0, ∞].
        style_embed = F.relu(style_embed)

        # print(len(style_embed))
        # for i in range(len(style_embed)):
        #     print(style_embed[i].shape)

        return style_embed, alignments

class ModifiedReferenceEncoder(nn.Module):
    """NN module creating a fixed size prosody embedding from a spectrogram.

    inputs: mel spectrograms [batch_size, num_spec_frames, num_mel]
    outputs: [batch_size, embedding_dim]
    """

    def __init__(self, num_mel, embedding_dim, use_nonlinear_proj = False):

        super().__init__()
        self.num_mel = num_mel
        filters = [1] + [32, 32, 64, 64, 128, 128]
        num_layers = len(filters) - 1

        convs = [
            CoordConv2d(in_channels=filters[0], out_channels=filters[0+1], kernel_size=(3,3), stride=(1,2),padding=(1,1), with_r=True)
        ]

        convs2 = [
            nn.Conv2d(
                in_channels=filters[i], out_channels=filters[i + 1], kernel_size=(3, 3), stride=(1, 2), padding=(1, 1)
            )
            for i in range(1,num_layers)
        ]

        convs.extend(convs2)

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_features=filter_size) for filter_size in filters[1:]])

        post_conv_height = self.calculate_post_conv_height(num_mel, 3, 2, 1, num_layers)
        self.recurrence = nn.GRU(
            input_size=filters[-1] * post_conv_height, hidden_size=embedding_dim, batch_first=True
        )

        self.dropout = nn.Dropout(p=0.5)
        
        self.use_nonlinear_proj = use_nonlinear_proj

        if(self.use_nonlinear_proj):
            self.proj = nn.Linear(embedding_dim, embedding_dim)
            nn.init.xavier_normal_(self.proj.weight) # Good init for projection
            # self.proj.bias.data.zero_() # Not random bias to "move" z

    def forward(self, inputs):
        batch_size = inputs.size(0)
        x = inputs.view(batch_size, 1, -1, self.num_mel)
        # x: 4D tensor [batch_size, num_channels==1, num_frames, num_mel]
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)

        # print(x.shape)
        x = x.transpose(1, 2)
        # x: 4D tensor [batch_size, post_conv_width,
        #               num_channels==128, post_conv_height]
        post_conv_width = x.size(1)

        x = x.contiguous().view(batch_size, post_conv_width, -1)
        # x: 3D tensor [batch_size, post_conv_width,
        #               num_channels*post_conv_height]
        self.recurrence.flatten_parameters()
        # print(x.shape)
        memory , out = self.recurrence(x)
        # out: 3D tensor [seq_len==1, batch_size, encoding_size=128]

        if(self.use_nonlinear_proj):
            out = torch.tanh(self.proj(out))
            out = self.dropout(out)
            
        return memory, out.squeeze(0)

    @staticmethod
    def calculate_post_conv_height(height, kernel_size, stride, pad, n_convs):
        """Height of spec after n convolutions with fixed kernel/stride/pad."""
        for _ in range(n_convs):
            height = (height - kernel_size + 2 * pad) // stride + 1
        return height

class ScaledDotProductAttention(nn.Module): # https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Modules.py
    ''' Scaled Dot-Product Attention '''

    def __init__(self, prosody_embedding_dim, encoder_embedding_dim, fg_attention_dropout, fg_attention_dim):
        super().__init__()
        self.dropout = nn.Dropout(fg_attention_dropout)
        self.d_q = encoder_embedding_dim
        self.d_k = prosody_embedding_dim
        self.linears = nn.ModuleList([
            LinearNorm(in_dim, fg_attention_dim, bias=False, w_init_gain='tanh') \
                for in_dim in (self.d_q, self.d_k)
        ])
        self.score_mask_value = -1e9

    def forward(self, q, k, v, mask=None):
        q, k = [linear(vector) for linear, vector in zip(self.linears, (q, k))]

        alignment = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k) # [N, seq_len, ref_len]

        if mask is not None:
            alignment.data.masked_fill_(mask == 0, self.score_mask_value)

        attention_weights = self.dropout(F.softmax(alignment, dim=-1))
        attention_context = torch.bmm(attention_weights, v) # [N, seq_len, prosody_embedding_dim]

        return attention_context, attention_weights

class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)

def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool() # (B, max_len)
    return mask


class CoordConv2d(conv.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, with_r=False):
        super(CoordConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias)
        self.rank = 2
        self.addcoords = AddCoords(self.rank, with_r)
        self.conv = nn.Conv2d(in_channels + self.rank + int(with_r), out_channels,
                              kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_tensor):
        """
        input_tensor_shape: (N, C_in,H,W)
        output_tensor_shape: N,C_out,H_out,W_out）
        :return: CoordConv2d Result
        """
        out = self.addcoords(input_tensor)
        out = self.conv(out)

        return out

class AddCoords(nn.Module):
    def __init__(self, rank, with_r=False):
        super(AddCoords, self).__init__()
        self.rank = rank
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        :param input_tensor: shape (N, C_in, H, W)
        :return:
        """
        if self.rank == 1:
            batch_size_shape, channel_in_shape, dim_x = input_tensor.shape
            xx_range = torch.arange(dim_x, dtype=torch.int32)
            xx_channel = xx_range[None, None, :]

            xx_channel = xx_channel.float() / (dim_x - 1)
            xx_channel = xx_channel * 2 - 1
            xx_channel = xx_channel.repeat(batch_size_shape, 1, 1)

            if torch.cuda.is_available:
                input_tensor = input_tensor.cuda()
                xx_channel = xx_channel.cuda()
            out = torch.cat([input_tensor, xx_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)

        elif self.rank == 2:
            batch_size_shape, channel_in_shape, dim_y, dim_x = input_tensor.shape
            xx_ones = torch.ones([1, 1, 1, dim_x], dtype=torch.int32)
            yy_ones = torch.ones([1, 1, 1, dim_y], dtype=torch.int32)

            xx_range = torch.arange(dim_y, dtype=torch.int32)
            yy_range = torch.arange(dim_x, dtype=torch.int32)
            xx_range = xx_range[None, None, :, None]
            yy_range = yy_range[None, None, :, None]

            xx_channel = torch.matmul(xx_range, xx_ones)
            yy_channel = torch.matmul(yy_range, yy_ones)

            # transpose y
            yy_channel = yy_channel.permute(0, 1, 3, 2)

            xx_channel = xx_channel.float() / (dim_y - 1)
            yy_channel = yy_channel.float() / (dim_x - 1)

            xx_channel = xx_channel * 2 - 1
            yy_channel = yy_channel * 2 - 1

            xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1)
            yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1)

            if torch.cuda.is_available:
                input_tensor = input_tensor.cuda()
                xx_channel = xx_channel.cuda()
                yy_channel = yy_channel.cuda()

            out = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)

        elif self.rank == 3:
            batch_size_shape, channel_in_shape, dim_z, dim_y, dim_x = input_tensor.shape
            xx_ones = torch.ones([1, 1, 1, 1, dim_x], dtype=torch.int32)
            yy_ones = torch.ones([1, 1, 1, 1, dim_y], dtype=torch.int32)
            zz_ones = torch.ones([1, 1, 1, 1, dim_z], dtype=torch.int32)

            xy_range = torch.arange(dim_y, dtype=torch.int32)
            xy_range = xy_range[None, None, None, :, None]

            yz_range = torch.arange(dim_z, dtype=torch.int32)
            yz_range = yz_range[None, None, None, :, None]

            zx_range = torch.arange(dim_x, dtype=torch.int32)
            zx_range = zx_range[None, None, None, :, None]

            xy_channel = torch.matmul(xy_range, xx_ones)
            xx_channel = torch.cat([xy_channel + i for i in range(dim_z)], dim=2)

            yz_channel = torch.matmul(yz_range, yy_ones)
            yz_channel = yz_channel.permute(0, 1, 3, 4, 2)
            yy_channel = torch.cat([yz_channel + i for i in range(dim_x)], dim=4)

            zx_channel = torch.matmul(zx_range, zz_ones)
            zx_channel = zx_channel.permute(0, 1, 4, 2, 3)
            zz_channel = torch.cat([zx_channel + i for i in range(dim_y)], dim=3)

            if torch.cuda.is_available:
                input_tensor = input_tensor.cuda()
                xx_channel = xx_channel.cuda()
                yy_channel = yy_channel.cuda()
                zz_channel = zz_channel.cuda()
            out = torch.cat([input_tensor, xx_channel, yy_channel, zz_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) +
                                torch.pow(yy_channel - 0.5, 2) +
                                torch.pow(zz_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)
        else:
            raise NotImplementedError

        return out