import numpy as np
import torch
import transformers
from torch import nn
from transformers import AutoFeatureExtractor, AutoModel

# from TTS.utils.audio import TorchSTFT
from TTS.encoder.models.base_encoder import BaseEncoder


class SSLEncoder(BaseEncoder):
    """Implementation of an encoder based on Self-supervised (SSL) pretrained models (like Hubert and Wav2vec)."""

    # pylint: disable=W0102
    def __init__(
        self,
        ssl_model_name_or_path="ntu-spml/distilhubert",
        freeze_feature_extractor=True,
        freeze_ssl_model=False,
        encoder_type="ASP",
        proj_dim=512,
        use_torch_spec=True,
        use_whole_audio=False,
        use_layers_weighted_sum=False,
    ):
        super(SSLEncoder, self).__init__()

        self.encoder_type = encoder_type
        self.freeze_ssl_model = freeze_ssl_model
        self.use_torch_spec = use_torch_spec
        self.use_whole_audio = use_whole_audio
        self.use_layers_weighted_sum = use_layers_weighted_sum

        self.ssl_model = AutoModel.from_pretrained(ssl_model_name_or_path, layerdrop=0.0)
        self.ssl_feature_extractor = AutoFeatureExtractor.from_pretrained(
            ssl_model_name_or_path, feature_size=1, padding_value=0.0, do_normalize=True, return_attention_mask=True
        )
        self.sampling_rate = self.ssl_feature_extractor.sampling_rate

        if freeze_feature_extractor:
            self.ssl_model.feature_extractor._freeze_parameters()

        if use_layers_weighted_sum:
            self.ssl_model = self.ssl_model.eval()
            # create a dummy input
            paired_wavs = torch.randn((1, self.sampling_rate)).to(self.ssl_model.device)
            with torch.no_grad():
                # get the ssl output to check the numbers of layers
                features = self.ssl_model(paired_wavs, output_hidden_states=True).hidden_states
                self.layer_num = len(features)

            self.weights = nn.Parameter(torch.zeros(self.layer_num))

            self.ssl_model = self.ssl_model.train()
            print("> SSL encoder is using the weighted sum of all hidden layers !")

        ssl_out_dim = self.ssl_model.config.hidden_size
        # pre_fc = nn.Linear(config.hidden_size, config.hidden_size)

        self.attention = nn.Sequential(
            nn.Conv1d(ssl_out_dim, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, ssl_out_dim, kernel_size=1),
            nn.Softmax(dim=2),
        )

        if self.encoder_type == "SAP":
            out_dim = ssl_out_dim
        elif self.encoder_type == "ASP":
            out_dim = ssl_out_dim * 2
        else:
            raise ValueError("Undefined encoder")

        self.fc = nn.Linear(out_dim, proj_dim)

        # use the model in eval mode if it is frozen
        if freeze_ssl_model:
            self.ssl_model = self.ssl_model.eval()

    def forward(self, x, l2_norm=False):
        """Forward pass of the model.

        Args:
            x (Tensor): Raw waveform signal or spectrogram frames. If input is a waveform, `torch_spec` must be `True`
                to compute the spectrogram on-the-fly.
            l2_norm (bool): Whether to L2-normalize the outputs.

        Shapes:
            - x: :math:`(N, 1, T_{in})` or :math:`(N, D_{spec}, T_{in})`
        """
        # prepare the waveform
        x = x.squeeze(1)

        if self.freeze_ssl_model:
            with torch.no_grad():
                ssl_out = self.ssl_model(input_values=x, output_hidden_states=self.use_layers_weighted_sum)
        else:
            ssl_out = self.ssl_model(input_values=x, output_hidden_states=self.use_layers_weighted_sum)

        if not self.use_layers_weighted_sum:
            x = ssl_out.last_hidden_state
        else:
            x = self._weighted_sum(ssl_out.hidden_states)

        x = x.transpose(1, 2)
        # x = x.reshape(x.size()[0], -1, x.size()[-1])
        w = self.attention(x)

        if self.encoder_type == "SAP":
            x = torch.sum(x * w, dim=2)
        elif self.encoder_type == "ASP":
            mu = torch.sum(x * w, dim=2)
            sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-5))
            x = torch.cat((mu, sg), 1)

        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        if l2_norm:
            x = torch.nn.functional.normalize(x, p=2, dim=1)

        return x

    def compute_embedding(self, x, num_frames=None, num_eval=None, return_mean=True, l2_norm=True):
        """
        Generate embeddings for a batch of utterances
        x: 1xTxD
        """
        if x.ndim == 1:
            x = x.unsqueeze(1)

        embeddings = self.inference(x, l2_norm=l2_norm)

        if return_mean:
            embeddings = torch.mean(embeddings, dim=0, keepdim=True)

        return embeddings

    def _weighted_sum(self, feature):
        assert self.layer_num == len(feature), (
            "If you run into this error, there is a great chance"
            " you are finetuning the upstream with wav2vec2's transformer blocks"
            " in weighted-sum mode (default), including wav2vec2, hubert, and decoar2."
            " These models use the layerdrop technique which causes the different number"
            " of layer forwards between different model forwards, resulting in different"
            " number of hidden states for different model forwards. Hence, finetuning"
            " these upstreams is essentially incompatible with weight-sum mode unless"
        )
        stacked_feature = torch.stack(feature, dim=0)

        _, *origin_shape = stacked_feature.shape
        stacked_feature = stacked_feature.view(self.layer_num, -1)
        norm_weights = torch.nn.functional.softmax(self.weights, dim=-1)
        weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
        weighted_feature = weighted_feature.view(*origin_shape)

        return weighted_feature
