from dataclasses import asdict, dataclass
from coqpit import Coqpit, check_argument

@dataclass
class StyleEncoderConfig(Coqpit):
    """Defines the Generic Style Encoder Config

    Args:    
        # TODO Args
    """
    # Style Encoder Type           
    se_type: str = "re"                     # Possibilites: "lookup", "re", "gst", "vae", "vaeflow" or "diffusion".
    num_styles: int = 5                     # Number of styles in dataset.
    num_speakers: int = 4                   # Number of speaker in the dataset. 
    
    # Inputs
    num_mel: int = 80                       # Dimension of the reference mel-spectrogram. Except if se_type == lookup.

    # Output 
    style_embedding_dim: int = 128          # Output dimension of the style embedding
    use_nonlinear_proj: bool = False        # Use a nonlinear projection (style_embedding_dim, style_embedding dim) and a tanh activation.
    use_proj_linear: bool = False           # Use a linear projection before decoder (Useful for matching dims in agg_type = "sum")
    proj_dim: int = 512                     # Projection dimension.

    # Supervised Training
    use_supervised_style: bool = False      # Enables the Style Manager.
    use_guided_style: bool = False          # Enables the Style Classification Layer.
    use_guided_speaker: bool = False        # Enables the Speaker Classification Layer.

    # Aggregation
    agg_type: str = "sum"                   # Possibilities: "concat", "sum", or "adain".
    agg_norm: bool = False                  # If agg_type == sum, you can rather than normalizing or not

    # Losses
    start_loss_at: int = 0                  # Iteration that the style loss should start propagate 
    content_orthogonal_loss: bool = False   # whether use othogonal loss between style and content embeddings
    speaker_orthogonal_loss: bool = False   # whether use othogonal loss between speaker and content embeddings
    orthogonal_loss: bool = False           # Use orthogonal loss
    orthogonal_loss_alpha: float = 1.0      # Use weight for orthogonal loss

    # GST-SE Additional Configs
    gst_style_input_weights: dict = None
    gst_num_heads: int = 4
    gst_num_style_tokens: int = 10

    # VAE-Based General Configs
    vae_latent_dim: int = 128               # Dim of mean and logvar
    use_cyclical_annealing: bool = True     # Whether use or not annealing (recommended true), only linear implemented
    vae_loss_alpha: float = 1.0             # Default alpha value (term of KL loss)
    vae_cycle_period: int = 5000            # iteration period to apply a new annealing cycle

    # VAEFLOW-SE Additional Configs
    vaeflow_intern_dim: int = 300
    vaeflow_number_of_flows: int = 16

    # Diffusion-SE Additional Configs
    diff_num_timesteps: int = 25 
    diff_schedule_type: str = 'cosine'
    diff_loss_type: str = 'l1' 
    diff_ref_online: bool = True 
    diff_step_dim: int = 128
    diff_in_out_ch: int = 1 
    diff_num_heads: int = 1 
    diff_hidden_channels: int = 128 
    diff_num_blocks: int = 5
    diff_dropout: float = 0.1
    diff_loss_alpha: float = 0.75

    def check_values(
        self,
    ):
        """Check config fields"""
        c = asdict(self)
        super().check_values()
        check_argument("se_type", c, restricted=True, enum_list=["lookup", "re", "gst", "vae", "diffusion", "vaeflow"])
        check_argument("agg_type", c, restricted=True, enum_list=["sum", "concat", "adain"])
        check_argument("num_mel", c, restricted=False)
        check_argument("style_embedding_dim", c, restricted=True, min_val=0, max_val=1000)
        check_argument("use_speaker_embedding", c, restricted=False)
        check_argument("gst_style_input_weights", c, restricted=False)
        check_argument("gst_num_heads", c, restricted=True, min_val=2, max_val=10)
        check_argument("gst_num_style_tokens", c, restricted=True, min_val=1, max_val=1000)
        check_argument("vae_latent_dim", c, restricted=True, min_val=0, max_val=1000)
        check_argument("vaeflow_intern_dim", c, restricted=True, min_val=0, max_val=1000)
        check_argument("vaeflow_number_of_flows", c, restricted=True, min_val=0, max_val=1000)
        check_argument("diff_num_timesteps", c, restricted=True, min_val=0, max_val=5000)
        check_argument("diff_schedule_type", c, restricted=True, enum_list=["cosine", "linear"])
        check_argument("diff__step", c, restricted=True, min_val=0, max_val=self.num_timesteps)
        check_argument("diff_loss_type", c, restricted=True, enum_list=["l1", "mse"])
        check_argument("diff_ref_online", c, restricted=True, enum_list=[True, False])
        check_argument("diff_step_dim", c, restricted=True, min_val=0, max_val=1000)
        check_argument("diff_in_out_ch", c, restricted=True, min_val=1, max_val=1)
        check_argument("diff_num_heads", c, restricted=False)
        check_argument("diff_hidden_channels", c, restricted=True, min_val=0, max_val=2048)
        check_argument("diff_num_blocks", c, restricted=True, min_val=0, max_val=100)
        check_argument("diff_dropout", c, restricted=False)
        check_argument("diff_loss_alpha", c, restricted=False)

        # Hierarchical Dependencies
        if (c["use_guided_style"] == True) or (c["se_type"]=="lookup"):
            assert c["use_supervised_style"]  == True