import numpy as np
import torch
import torch.nn as nn

# Abstract Base Flow Class
class Flow(nn.Module):
    """
    Generic class for Flow Functions
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, z):
        """
        :param z: input variable, first dimension is batch dim
        :return: transformed z and log of absolute determinant
        """
        raise NotImplementedError('Forward pass has not been implemented.')

    def inverse(self, z):
        raise NotImplementedError('This flow has no algebraic inverse.')

## Flow Types

class Planar(Flow):
    """
    Planar flow as introduced in https://arxiv.org/abs/1505.05770:
        f(z) = z + u * h(w.T * z + b)
    """

    def __init__(self, shape, act="tanh", u=None, w=None, b=None):
        """
        Constructor of the planar flow
        :param shape: shape of the latent variable z
        :param h: nonlinear function h of the planar flow (see definition of f above)
        :param u,w,b: optional initialization for parameters
        """
        super().__init__()
        # Define Limits for Random Init of w and u
        lim_w = np.sqrt(2. / np.prod(shape))
        lim_u = np.sqrt(2)

        # Initialize u,w,b
        if u is not None:
            self.u = nn.Parameter(u)
        else:
            self.u = nn.Parameter(torch.empty(shape)[None])
            nn.init.uniform_(self.u, -lim_u, lim_u)
        if w is not None:
            self.w = nn.Parameter(w)
        else:
            self.w = nn.Parameter(torch.empty(shape)[None])
            nn.init.uniform_(self.w, -lim_w, lim_w)
        if b is not None:
            self.b = nn.Parameter(b)
        else:
            self.b = nn.Parameter(torch.zeros(1))

        # Initialize h
        self.act = act
        if act == "tanh":
            self.h = torch.tanh
        elif act == "leaky_relu":
            self.h = torch.nn.LeakyReLU(negative_slope=0.2)
        else:
            raise NotImplementedError('Nonlinearity is not implemented.')

    def forward(self, z):
        lin = torch.sum(self.w * z, list(range(1, self.w.dim()))) + self.b
        if self.act == "tanh":
            # Ensure constrain for invertibility w.T * u > -1
            inner = torch.sum(self.w * self.u)
            u = self.u + (torch.log(1 + torch.exp(inner)) - 1 - inner) * self.w / torch.sum(self.w ** 2) 
            # Derivative of h
            h_ = lambda x: 1 / torch.cosh(x) ** 2
        elif self.act == "leaky_relu":
            # Ensure constrain for invertibility
            inner = torch.sum(self.w * self.u)
            u = self.u + (torch.log(1 + torch.exp(inner)) - 1 - inner) * self.w / torch.sum(self.w ** 2)  # constraint w.T * u neq -1, use >
            # Derivative of h
            h_ = lambda x: (x < 0) * (self.h.negative_slope - 1.0) + 1.0

        z_ = z + u * self.h(lin.unsqueeze(1))
        log_det = torch.log(torch.abs(1 + torch.sum(self.w * u) * h_(lin)))
        return z_, log_det

    def inverse(self, z):
        if self.act != "leaky_relu":
            raise NotImplementedError('This flow has no algebraic inverse.')
        lin = torch.sum(self.w * z, list(range(2, self.w.dim())), keepdim=True) + self.b
        inner = torch.sum(self.w * self.u)
        a = ((lin + self.b) / (1 + inner) < 0) * (self.h.negative_slope - 1.0) + 1.0  # absorb leakyReLU slope into u
        u = a * (self.u + (torch.log(1 + torch.exp(inner)) - 1 - inner) * self.w / torch.sum(self.w ** 2))
        z_ = z - 1 / (1 + inner) * (lin + u * self.b)
        log_det = -torch.log(torch.abs(1 + torch.sum(self.w * u)))
        if log_det.dim() == 0:
            log_det = log_det.unsqueeze(0)
        if log_det.dim() == 1:
            log_det = log_det.unsqueeze(1)
        return z_, log_det

class GlowBlock(Flow):
    """
    Glow: Generative Flow with Invertible 1x1 Convolutions:
    https://arxiv.org/pdf/1807.03039
    One Block of the Glow model, comprised of
    MaskedAffineFlow (affine coupling layer)
    Invertible1x1Conv (dropped if there is only one channel)
    ActNorm (first batch used for initialization)
    """
    def __init__(self, channels, hidden_channels, scale=True, scale_map='sigmoid',
                 split_mode='channel', leaky=0.0, init_zeros=True, use_lu=True,
                 net_actnorm=False):
        """
        Constructor
        :param channels: Number of channels of the data
        :param hidden_channels: number of channels in the hidden layer of the ConvNet
        :param scale: Flag, whether to include scale in affine coupling layer
        :param scale_map: Map to be applied to the scale parameter, can be 'exp' as in
        RealNVP or 'sigmoid' as in Glow
        :param split_mode: Splitting mode, for possible values see Split class
        :param leaky: Leaky parameter of LeakyReLUs of ConvNet2d
        :param init_zeros: Flag whether to initialize last conv layer with zeros
        :param use_lu: Flag whether to parametrize weights through the LU decomposition
        in invertible 1x1 convolution layers
        :param logscale_factor: Factor which can be used to control the scale of
        the log scale factor, see https://github.com/openai/glow
        """
        super().__init__()
        self.flows = nn.ModuleList([])
        # Coupling layer
        kernel_size = (3, 1, 3)
        num_param = 2 if scale else 1
        if 'channel' == split_mode:
            channels_ = (channels // 2,) + 2 * (hidden_channels,)
            channels_ += (num_param * ((channels + 1) // 2),)
        elif 'channel_inv' == split_mode:
            channels_ = ((channels + 1) // 2,) + 2 * (hidden_channels,)
            channels_ += (num_param * (channels // 2),)
        elif 'checkerboard' in split_mode:
            channels_ = (channels,) + 2 * (hidden_channels,)
            channels_ += (num_param * channels,)
        else:
            raise NotImplementedError('Mode ' + split_mode + ' is not implemented.')
        param_map = ConvNet2d(channels_, kernel_size, leaky, init_zeros, actnorm=net_actnorm)
        self.flows += [AffineCouplingBlock(param_map, scale, scale_map, split_mode)]
        # Invertible 1x1 convolution
        if channels > 1:
            self.flows += [Invertible1x1Conv(channels, use_lu)]
        # Activation normalization
        self.flows += [ActNorm((channels,) + (1, 1))]

    def forward(self, z):
        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for flow in self.flows:
            z, log_det = flow(z)
            log_det_tot += log_det
        return z, log_det_tot

    def inverse(self, z):
        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_det_tot += log_det
        return z, log_det_tot

class ConvNet2d(nn.Module):
    """
    Convolutional Neural Network with leaky ReLU nonlinearities
    """

    def __init__(self, channels, kernel_size, leaky=0.0, init_zeros=True,
                 actnorm=False, weight_std=None):
        """
        Constructor
        :param channels: List of channels of conv layers, first entry is in_channels
        :param kernel_size: List of kernel sizes, same for height and width
        :param leaky: Leaky part of ReLU
        :param init_zeros: Flag whether last layer shall be initialized with zeros
        :param scale_output: Flag whether to scale output with a log scale parameter
        :param logscale_factor: Constant factor to be multiplied to log scaling
        :param actnorm: Flag whether activation normalization shall be done after
        each conv layer except output
        :param weight_std: Fixed std used to initialize every layer
        """
        super().__init__()
        # Build network
        net = nn.ModuleList([])
        for i in range(len(kernel_size) - 1):
            conv = nn.Conv2d(channels[i], channels[i + 1], kernel_size[i],
                             padding=kernel_size[i] // 2, bias=(not actnorm))
            if weight_std is not None:
                conv.weight.data.normal_(mean=0.0, std=weight_std)
            net.append(conv)
            if actnorm:
                net.append(SingleActNorm((channels[i + 1],) + (1, 1)))
            net.append(nn.LeakyReLU(leaky))
        i = len(kernel_size)
        net.append(nn.Conv2d(channels[i - 1], channels[i], kernel_size[i - 1],
                             padding=kernel_size[i - 1] // 2))
        if init_zeros:
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)

class SingleActNorm(nn.Module):
    """
    ActNorm layer with just one forward pass
    """
    def __init__(self, shape, logscale_factor=None):
        """
        Constructor
        :param shape: Same as shape in flows.ActNorm
        :param logscale_factor: Same as shape in flows.ActNorm
        """
        super().__init__()
        self.actNorm = ActNorm(shape, logscale_factor=logscale_factor)

    def forward(self, input):
        out, _ = self.actNorm(input)
        return out

class AffineCouplingBlock(Flow):
    """
    Affine Coupling layer including split and merge operation
    """
    def __init__(self, param_map, scale=True, scale_map='exp', split_mode='channel'):
        """
        Constructor
        :param param_map: Maps features to shift and scale parameter (if applicable)
        :param scale: Flag whether scale shall be applied
        :param scale_map: Map to be applied to the scale parameter, can be 'exp' as in
        RealNVP or 'sigmoid' as in Glow
        :param split_mode: Splitting mode, for possible values see Split class
        """
        super().__init__()
        self.flows = nn.ModuleList([])
        # Split layer
        self.flows += [Split(split_mode)]
        # Affine coupling layer
        self.flows += [AffineCoupling(param_map, scale, scale_map)]
        # Merge layer
        self.flows += [Merge(split_mode)]

    def forward(self, z):
        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for flow in self.flows:
            z, log_det = flow(z)
            log_det_tot += log_det
        return z, log_det_tot

    def inverse(self, z):
        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_det_tot += log_det
        return z, log_det_tot

class Split(Flow):
    """
    Split features into two sets
    """
    def __init__(self, mode='channel'):
        """
        Constructor
        :param mode: Splitting mode, can be
            channel: Splits first feature dimension, usually channels, into two halfs
            channel_inv: Same as channel, but with z1 and z2 flipped
            checkerboard: Splits features using a checkerboard pattern (last feature dimension must be even)
            checkerboard_inv: Same as checkerboard, but with inverted coloring
        """
        super().__init__()
        self.mode = mode

    def forward(self, z):
        if self.mode == 'channel':
            z1, z2 = z.chunk(2, dim=1)
        elif self.mode == 'channel_inv':
            z2, z1 = z.chunk(2, dim=1)
        elif 'checkerboard' in self.mode:
            n_dims = z.dim()
            cb0 = 0
            cb1 = 1
            for i in range(1, n_dims):
                cb0_ = cb0
                cb1_ = cb1
                cb0 = [cb0_ if j % 2 == 0 else cb1_ for j in range(z.size(n_dims - i))]
                cb1 = [cb1_ if j % 2 == 0 else cb0_ for j in range(z.size(n_dims - i))]
            cb = cb1 if 'inv' in self.mode else cb0
            cb = torch.tensor(cb)[None].repeat(len(z), *((n_dims - 1) * [1]))
            cb = cb.to(z.device)
            z_size = z.size()
            z1 = z.reshape(-1)[torch.nonzero(cb.view(-1), as_tuple=False)].view(*z_size[:-1], -1)
            z2 = z.reshape(-1)[torch.nonzero((1 - cb).view(-1), as_tuple=False)].view(*z_size[:-1], -1)
        else:
            raise NotImplementedError('Mode ' + self.mode + ' is not implemented.')
        log_det = 0
        return [z1, z2], log_det

    def inverse(self, z):
        z1, z2 = z
        if self.mode == 'channel':
            z = torch.cat([z1, z2], 1)
        elif self.mode == 'channel_inv':
            z = torch.cat([z2, z1], 1)
        elif 'checkerboard' in self.mode:
            n_dims = z1.dim()
            z_size = list(z1.size())
            z_size[-1] *= 2
            cb0 = 0
            cb1 = 1
            for i in range(1, n_dims):
                cb0_ = cb0
                cb1_ = cb1
                cb0 = [cb0_ if j % 2 == 0 else cb1_ for j in range(z_size[n_dims - i])]
                cb1 = [cb1_ if j % 2 == 0 else cb0_ for j in range(z_size[n_dims - i])]
            cb = cb1 if 'inv' in self.mode else cb0
            cb = torch.tensor(cb)[None].repeat(z_size[0], *((n_dims - 1) * [1]))
            cb = cb.to(z1.device)
            z1 = z1[..., None].repeat(*(n_dims * [1]), 2).view(*z_size[:-1], -1)
            z2 = z2[..., None].repeat(*(n_dims * [1]), 2).view(*z_size[:-1], -1)
            z = cb * z1 + (1 - cb) * z2
        else:
            raise NotImplementedError('Mode ' + self.mode + ' is not implemented.')
        log_det = 0
        return z, log_det

class AffineCoupling(Flow):
    """
    Affine Coupling Layer as introduced in the RealNVP paper:
    https://arxiv.org/abs/1605.08803
    """

    def __init__(self, param_map, scale=True, scale_map='exp') -> None:
        """
        Constructor
        :param param_map: Maps features to shift and scale parameter (if applicable)
        :param scale: Flag whether scale shall be applied
        :param scale_map: Map to be applied to the scale parameter, can be 'exp' as in
        RealNVP or 'sigmoid' as in Glow, 'sigmoid_inv' used multiplicative sigmoid scale
        when sampling from the model
        """
        super().__init__()
        self.add_module('param_map', param_map)
        self.scale = scale
        self.scale_map = scale_map

    def forward(self, z):
        """
        z is a list of z1 and z2: z = [z1,z2]
        z1 is left constant and affine map is applied to z2 with parameters depending on z1
        """
        z1, z2 = z
        param = self.param_map(z1)
        if self.scale:
            shift = param[:, 0::2, ...]
            scale_ = param[:, 1::2, ...]
            if self.scale_map == 'exp':
                z2 = z2 * torch.exp(scale_) + shift
                log_det = torch.sum(scale_, dim = list(range(1, shift.dim())))
            elif self.scale_map == 'sigmoid':
                scale = torch.sigmoid(scale_ + 2)
                z2 = z2/scale + shift
                log_det = -torch.sum(torch.log(scale),
                                    dim = list(range(1, shift.dim())))
            elif self.scale_map == 'sigmoid_inv':
                scale = torch.sigmoid(scale_ + 2)
                z2 = z2 + scale + shift
                log_det = torch.sum(torch.log(scale),
                                    dim = list(range(1, shift.dim())))
            else:
                raise NotImplementedError('This scale map is not implemented.')
        else:
            z2+=param
            log_det = 0
        return [z1,z2], log_det

    def inverse(self, z):
        z1, z2 = z
        param = self.param_map(z1)
        if self.scale:
            shift = param[:, 0::2, ...]
            scale_ = param[:, 1::2, ...]
            if self.scale_map == 'exp':
                z2 = (z2 - shift) * torch.exp(-scale_)
                log_det = -torch.sum(scale_, dim=list(range(1, shift.dim())))
            elif self.scale_map == 'sigmoid':
                scale = torch.sigmoid(scale_ + 2)
                z2 = (z2 - shift) * scale
                log_det = torch.sum(torch.log(scale),
                                    dim=list(range(1, shift.dim())))
            elif self.scale_map == 'sigmoid_inv':
                scale = torch.sigmoid(scale_ + 2)
                z2 = (z2 - shift) / scale
                log_det = -torch.sum(torch.log(scale),
                                     dim=list(range(1, shift.dim())))
            else:
                raise NotImplementedError('This scale map is not implemented.')
        else:
            z2 -= param
            log_det = 0
        return [z1, z2], log_det

class Merge(Split):
    """
    Same as Split but with forward and backward pass interchanged
    """
    def __init__(self, mode='channel'):
        super().__init__(mode)

    def forward(self, z):
        return super().inverse(z)

    def inverse(self, z):
        return super().forward(z)

class AffineConstFlow(Flow):
    """
    scales and shifts with learned constants per dimension. In the NICE paper
    there is a scaling layer which is a special case of this where t is None.
    """
    
    def __init__(self, shape, scale=True, shift=True) -> None:
        """
        Constructor
        :param shape: Shape of the coupling layer
        :param scale: Flag whether to apply scaling
        :param shift: Flag whether to apply shift
        :param logscale_factor: Optional factor which can be used to control
        the scale of the log scale factor.
        """
        super().__init__()
        if scale:
            self.s = nn.Parameter(torch.zeros(shape)[None])
        else:
            self.register_buffer('s', torch.zeros(shape)[None])
        if shift:
            self.t = nn.Parameter(torch.zeros(shape)[None])
        else:
            self.register_buffer('t', torch.zeros(shape)[None])
        self.n_dim = self.s.dim()
        self.batch_dims = torch.nonzero(torch.tensor(self.s.shape) == 1, as_tuple=False)[:,0].tolist()

    def forward(self, z):
        z_ = z * torch.exp(self.s) + self.t
        if len(self.batch_dims) > 1:
            prod_batch_dims = np.prod([z.size(i) for i in self.batch_dims[1:]])
        else:
            prod_batch_dims = 1
        log_det = -prod_batch_dims * torch.sum(self.s)
        return z_, log_det

    def inverse(self, z):
        z_ = (z - self.t) * torch.exp(-self.s)
        if len(self.batch_dims) > 1:
            prod_batch_dims = np.prod([z.size(i) for i in self.batch_dims[1:]])
        else:
            prod_batch_dims = 1
        log_det = -prod_batch_dims * torch.sum(self.s)
        return z_, log_det

class Invertible1x1Conv(Flow):
    """
    Invertible 1x1 convolution introduced in the Glow paper
    Assumes 4d input/output tensors of the form [N,C,H,W]
    """

    def __init__(self, num_channels, use_lu=False):
        """
        Constructor
        :param num_channels: Number of channels of the data
        :param use_lu: Flag whether to parametrize weights through the LU decomposition
        """
        super().__init__()
        self.num_channels = num_channels
        self.use_lu = use_lu
        Q = torch.linalg.qr(torch.randn(self.num_channels, self.num_channels))[0]
        if use_lu:
            P, L, U = torch.lu_unpack(*Q.lu())
            self.register_buffer('P', P)  # remains fixed during optimization
            self.L = nn.Parameter(L)  # lower triangular portion
            S = U.diag()  # "crop out" the diagonal to its own parameter
            self.register_buffer("sign_S", torch.sign(S))
            self.log_S = nn.Parameter(torch.log(torch.abs(S)))
            self.U = nn.Parameter(torch.triu(U, diagonal=1))  # "crop out" diagonal, stored in S
            self.register_buffer("eye", torch.diag(torch.ones(self.num_channels)))
        else:
            self.W = nn.Parameter(Q)

    def _assemble_W(self, inverse=False):
        # assemble W from its components (P, L, U, S)
        L = torch.tril(self.L, diagonal=-1) + self.eye
        U = torch.triu(self.U, diagonal=1) + torch.diag(self.sign_S * torch.exp(self.log_S))
        if inverse:
            if self.log_S.dtype == torch.float64:
                L_inv = torch.inverse(L)
                U_inv = torch.inverse(U)
            else:
                L_inv = torch.inverse(L.double()).type(self.log_S.dtype)
                U_inv = torch.inverse(U.double()).type(self.log_S.dtype)
            W = U_inv @ L_inv @ self.P.t()
        else:
            W = self.P @ L @ U
        return W

    def forward(self, z):
        if self.use_lu:
            W = self._assemble_W(inverse=True)
            log_det = -torch.sum(self.log_S)
        else:
            W_dtype = self.W.dtype
            if W_dtype == torch.float64:
                W = torch.inverse(self.W)
            else:
                W = torch.inverse(self.W.double()).type(W_dtype)
            W = W.view(*W.size(), 1, 1)
            log_det = -torch.slogdet(self.W)[1]
        W = W.view(self.num_channels, self.num_channels, 1, 1)

        z_ = torch.nn.functional.conv2d(z, W)
        log_det = log_det * z.size(2) * z.size(3)
        return z_, log_det

    def inverse(self, z):
        if self.use_lu:
            W = self._assemble_W()
            log_det = torch.sum(self.log_S)
        else:
            W = self.W
            log_det = torch.slogdet(self.W)[1]
        W = W.view(self.num_channels, self.num_channels, 1, 1)
        
        z_ = torch.nn.functional.conv2d(z, W)
        log_det = log_det * z.size(2) * z.size(3)
        return z_, log_det

class ActNorm(AffineConstFlow):
    """
    An AffineConstFlow but with a data-dependent initialization,
    where on the very first batch we clever initialize the s,t so that the output
    is unit gaussian. As described in Glow paper.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dep_init_done_cpu = torch.tensor(0.)
        self.register_buffer('data_dep_init_done', self.data_dep_init_done_cpu)

    def forward(self, z):
        # first batch is used for initialization, c.f. batchnorm
        if not self.data_dep_init_done > 0.:
            assert self.s is not None and self.t is not None
            s_init = -torch.log(z.std(dim=self.batch_dims, keepdim=True) + 1e-6)
            self.s.data = s_init.data
            self.t.data = (-z.mean(dim=self.batch_dims, keepdim=True) * torch.exp(self.s)).data
            self.data_dep_init_done = torch.tensor(1.)
        return super().forward(z)

    def inverse(self, z):
        # first batch is used for initialization, c.f. batchnorm
        if not self.data_dep_init_done:
            assert self.s is not None and self.t is not None
            s_init = torch.log(z.std(dim=self.batch_dims, keepdim=True) + 1e-6)
            self.s.data = s_init.data
            self.t.data = z.mean(dim=self.batch_dims, keepdim=True).data
            self.data_dep_init_done = torch.tensor(1.)
        return super().inverse(z)