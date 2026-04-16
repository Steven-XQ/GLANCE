# Developed by Junyi Ma
# Diff-IP2D: Diffusion-Based Hand-Object Interaction Prediction on Egocentric Videos
# https://github.com/IRMVLab/Diff-IP2D
# We thank OCT (Liu et al.), Diffuseq (Gong et al.), and USST (Bao et al.) for providing the codebases.

from diffip2d import gaussian_diffusion as gd
from diffip2d.gaussian_diffusion import HOIDiffusion, space_timesteps
from diffip2d.transformer_model import TransformerNetModel, MADT, RL_HOITransformerNetModel
from diffip2d.pre_encoder import SideFusionEncoder, GazeSideFusionEncoder, MotionEncoder
from diffip2d.post_decoder import TrajDecoder

def create_network_and_diffusion(
    hidden_t_dim,
    hidden_dim,
    vocab_size,
    config_name,
    use_plm_init,
    dropout,
    diffusion_steps,
    noise_schedule,
    learn_sigma,
    timestep_respacing,
    predict_xstart,
    rescale_timesteps,
    sigma_small,
    rescale_learned_sigmas,
    use_kl,
    sf_encoder_hidden,
    traj_decoder_hidden1,
    traj_decoder_hidden2,
    motion_encoder_hidden,
    madt_depth,
    feat_num=3,   # global hand object
    traj_dim=2,   # 2D traj on egocentric video
    homo_dim=3,   # homography matrix
    use_gaze=False,
    T_max=20,
    gaze_coord_only=False,
    gaze_fusion_only=False,
    gaze_alpha_clamp=0.0,
    gaze_last_n_blocks=0,
    gaze_fixed_delta=0,
    gaze_bias_init_delta=0,
    gaze_bias_init_amp=2.0,
    gaze_before_motion=False,
    **kwargs,
):

    if use_gaze:
        from diffip2d.gaze_modules import GazeEncoder
        gaze_encoder = GazeEncoder(output_dim=hidden_dim, coord_only=gaze_coord_only)
        sf_encoder = GazeSideFusionEncoder(input_dims=feat_num * hidden_dim, output_dims=hidden_dim,
                                           encoder_hidden_dims=sf_encoder_hidden, gaze_dim=hidden_dim)
    else:
        gaze_encoder = None
        sf_encoder = SideFusionEncoder(input_dims=feat_num * hidden_dim, output_dims=hidden_dim,
                                       encoder_hidden_dims=sf_encoder_hidden)

    # When gaze_fusion_only, gaze is used only in SideFusionEncoder gating,
    # not in MADT cross-attention (to avoid trajectory regression).
    use_gaze_in_madt = use_gaze and not gaze_fusion_only

    traj_decoder = TrajDecoder(input_dims=hidden_dim, output_dims=traj_dim,
                               encoder_hidden_dims1=traj_decoder_hidden1,
                               encoder_hidden_dims2=traj_decoder_hidden2)
    motion_encoder = MotionEncoder(input_dims=homo_dim * homo_dim, output_dims=hidden_dim,
                                   encoder_hidden_dims=motion_encoder_hidden)
    denoised_model = MADT(
        input_dims=hidden_dim,
        output_dims=(hidden_dim if not learn_sigma else hidden_dim*2),
        hidden_t_dim=hidden_t_dim,
        dropout=dropout,
        depth=madt_depth,
        use_gaze=use_gaze_in_madt,
        T_max=T_max,
        gaze_last_n_blocks=gaze_last_n_blocks,
        gaze_alpha_clamp=gaze_alpha_clamp,
        gaze_fixed_delta=gaze_fixed_delta,
        gaze_bias_init_delta=gaze_bias_init_delta,
        gaze_bias_init_amp=gaze_bias_init_amp,
        gaze_before_motion=gaze_before_motion,
    )

    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if not timestep_respacing:
        timestep_respacing = [diffusion_steps]
    diffusion = HOIDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        rescale_timesteps=rescale_timesteps,
        predict_xstart=predict_xstart,
        learn_sigmas = learn_sigma,
        sigma_small = sigma_small,
        use_kl = use_kl,
        rescale_learned_sigmas=rescale_learned_sigmas
    )

    return sf_encoder, denoised_model, diffusion, traj_decoder, motion_encoder, gaze_encoder
