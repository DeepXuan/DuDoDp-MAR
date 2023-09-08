import argparse
import inspect

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .unet import SuperResModel, UNetModel, EncoderUNetModel, SinoUnet, SinoMLP

NUM_CLASSES = 1000 # 1000

def diffusion_defaults():
    """
    Defaults for image and classifier training. 
    We set the default output to xstart -> eps is difficult to predict at very low SNR levels
    """
    return dict(
        learn_sigma=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        model_mean_type="xstart",
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        snr_splits="",
        weight_schedule="sqrt_snr"
    )


def classifier_defaults():
    """
    Defaults for classifier models.
    """
    return dict(
        image_size=64,
        classifier_use_fp16=False,
        classifier_width=128,
        classifier_depth=2,
        classifier_attention_resolutions="32,16,8",  # 16
        classifier_use_scale_shift_norm=True,  # False
        classifier_resblock_updown=True,  # False
        classifier_pool="attention",
    )


def model_and_diffusion_defaults():
    """
    Defaults for image training - which includes using patching (p=4) and classifier-free guidance.
    """
    res = dict(
        model_name='Unet',
        sino_size=641,
        image_size=64,
        in_channels=3,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="16,8",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        view_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,

        patch_size=4,
        classifier_free=True,
        snr_splits="",
        weight_schedule="sqrt_snr",

        img_size=None,
        p_size=None,
        in_chans=None,
        embed_dim=None,
        depth=None,
        n_heads=None,
        mlp_ratio=None,
        qkv_bias=None,
        qk_scale=None

    )
    res.update(diffusion_defaults())
    return res


def classifier_and_diffusion_defaults():
    res = classifier_defaults()
    res.update(diffusion_defaults())
    return res


def create_model_and_diffusion(
    model_name=None,
    sino_size=None,
    image_size=None,
    in_channels=None,
    class_cond=None,
    view_cond=None,
    learn_sigma=None,
    num_channels=None,
    num_res_blocks=None,

    patch_size=None,
    classifier_free=None,

    channel_mult=None,
    num_heads=None,
    num_head_channels=None,
    num_heads_upsample=None,
    attention_resolutions=None,
    dropout=None,
    diffusion_steps=None,
    noise_schedule=None,
    timestep_respacing=None,
    use_kl=None,
    model_mean_type=None,
    rescale_timesteps=None,
    rescale_learned_sigmas=None,
    use_checkpoint=None,
    use_scale_shift_norm=None,
    resblock_updown=None,
    use_fp16=None,
    use_new_attention_order=None,
    snr_splits=None,
    weight_schedule=None,

    img_size=None,
    p_size=None,
    in_chans=None,
    embed_dim=None,
    depth=None,
    n_heads=None,
    mlp_ratio=None,
    qkv_bias=None,
    qk_scale=None
):
    model = create_model(
        model_name=model_name,
        sino_size=sino_size,
        image_size=image_size,
        in_channels=in_channels,
        num_channels=num_channels,
        num_res_blocks=num_res_blocks,
        patch_size=patch_size,
        classifier_free=classifier_free,
        channel_mult=channel_mult,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        view_cond=view_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        use_new_attention_order=use_new_attention_order,

        img_size=img_size,
        p_size=p_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        depth=depth,
        n_heads=n_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        qk_scale=qk_scale,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        model_mean_type=model_mean_type,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
        snr_splits=snr_splits,
        weight_schedule=weight_schedule
    )
    return model, diffusion


def create_model(
    model_name,
    sino_size,
    image_size, #real image size, not the dimensions of the image after patches have been extracted.
    in_channels,
    num_channels,
    num_res_blocks,

    patch_size=4,
    classifier_free=True,

    channel_mult="",
    learn_sigma=False,
    class_cond=False,
    view_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,

    img_size=None,
    p_size=None,
    in_chans=None,
    embed_dim=None,
    depth=None,
    n_heads=None,
    mlp_ratio=None,
    qkv_bias=None,
    qk_scale=None
):
    assert image_size%patch_size == 0, "patch size must evenly divide image size."

    input_res = image_size//patch_size

    if channel_mult == "":
        if input_res == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif input_res == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif input_res == 64:
            channel_mult = (1, 2, 3, 4)
        elif input_res == 32:
            channel_mult = (1, 2, 3)
        elif input_res == 16:
            channel_mult == (1, 2)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(float(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(input_res // int(res))
    if model_name == 'Unet':
        return UNetModel(
            image_size=image_size,
            in_channels=in_channels, # 3
            model_channels=num_channels,
            out_channels=(in_channels if not learn_sigma else in_channels*2), # 3, 6
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=dropout,
            channel_mult=channel_mult,
            patch_size=patch_size,
            classifier_free=classifier_free,
            num_classes=(NUM_CLASSES if class_cond else None),
            view_cond=view_cond,
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
        )
    elif model_name == 'SinoUnet':
        return SinoUnet(
            sino_size=sino_size,
            image_size=image_size,
            in_channels=in_channels, # 3
            model_channels=num_channels,
            out_channels=(in_channels if not learn_sigma else in_channels*2), # 3, 6
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=dropout,
            channel_mult=channel_mult,
            patch_size=patch_size,
            classifier_free=classifier_free,
            num_classes=(NUM_CLASSES if class_cond else None),
            view_cond=view_cond,
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
        )
    elif model_name == 'SinoMLP':
        return SinoMLP(
            sino_size=sino_size,
            image_size=image_size,
            in_channels=in_channels, # 3
            model_channels=num_channels,
            out_channels=(in_channels if not learn_sigma else in_channels*2), # 3, 6
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=dropout,
            channel_mult=channel_mult,
            patch_size=patch_size,
            classifier_free=classifier_free,
            num_classes=(NUM_CLASSES if class_cond else None),
            view_cond=view_cond,
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
        )
    elif model_name == 'UViTSino':
        from .uvit import UViTSino
        return UViTSino(img_size=(640, 641), in_chans=1, out_chans=(in_channels if not learn_sigma else in_channels*2),
                         embed_dim=1024, depth=20, num_heads=16, mlp_ratio=4., qkv_bias=False, qk_scale=None)
    elif model_name == 'UViT':
        from .uvit import UViT
        return UViT(out_chans=(in_channels if not learn_sigma else in_channels*2), embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None) # 3, 6)
    elif model_name == 'UViTSinoSlice':
        from .uvit import UViTSinoSlice
        return UViTSinoSlice(img_size=img_size, patch_size=p_size, in_chans=in_chans, out_chans=(in_channels if not learn_sigma else in_channels*2),  
                             embed_dim=embed_dim, depth=depth, num_heads=n_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=None if qk_scale=='None' else qk_scale)
        # return UViTSinoSlice(out_chans=(in_channels if not learn_sigma else in_channels*2))


def create_classifier_and_diffusion(
    image_size,
    classifier_use_fp16,
    classifier_width,
    classifier_depth,
    classifier_attention_resolutions,
    classifier_use_scale_shift_norm,
    classifier_resblock_updown,
    classifier_pool,
    learn_sigma,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    model_mean_type,
    rescale_timesteps,
    rescale_learned_sigmas,
):
    classifier = create_classifier(
        image_size,
        classifier_use_fp16,
        classifier_width,
        classifier_depth,
        classifier_attention_resolutions,
        classifier_use_scale_shift_norm,
        classifier_resblock_updown,
        classifier_pool,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        model_mean_type=model_mean_type,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return classifier, diffusion


def create_classifier(
    image_size,
    classifier_use_fp16,
    classifier_width,
    classifier_depth,
    classifier_attention_resolutions,
    classifier_use_scale_shift_norm,
    classifier_resblock_updown,
    classifier_pool,
):
    if image_size == 512:
        channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
    elif image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 128:
        channel_mult = (1, 1, 2, 3, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    for res in classifier_attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return EncoderUNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=classifier_width,
        out_channels=1000,
        num_res_blocks=classifier_depth,
        attention_resolutions=tuple(attention_ds),
        channel_mult=channel_mult,
        use_fp16=classifier_use_fp16,
        num_head_channels=64,
        use_scale_shift_norm=classifier_use_scale_shift_norm,
        resblock_updown=classifier_resblock_updown,
        pool=classifier_pool,
    )


def sr_model_and_diffusion_defaults():
    res = model_and_diffusion_defaults()
    res["large_size"] = 256
    res["small_size"] = 64
    arg_names = inspect.getfullargspec(sr_create_model_and_diffusion)[0]
    for k in res.copy().keys():
        if k not in arg_names:
            del res[k]
    return res


def sr_create_model_and_diffusion(
    large_size,
    small_size,
    class_cond,
    learn_sigma,
    num_channels,
    num_res_blocks,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    patch_size,
    classifier_free,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    model_mean_type,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
):
    model = sr_create_model(
        large_size,
        small_size,
        num_channels,
        num_res_blocks,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        patch_size=patch_size,
        classifier_free=classifier_free,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        model_mean_type=model_mean_type,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion


def sr_create_model(
    large_size,
    small_size,
    num_channels,
    num_res_blocks,
    learn_sigma,
    class_cond,
    use_checkpoint,
    attention_resolutions,
    patch_size,
    classifier_free,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
    resblock_updown,
    use_fp16,
):
    _ = small_size  # hack to prevent unused variable

    if large_size == 512:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif large_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif large_size == 64:
        channel_mult = (1, 2, 3, 4)
    else:
        raise ValueError(f"unsupported large size: {large_size}")

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(large_size // int(res))

    return SuperResModel(
        image_size=large_size,
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        patch_size=patch_size,
        classifier_free=classifier_free,
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
    )


def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    model_mean_type='xstart',
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
    snr_splits="",
    weight_schedule="sqrt_snr"
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]

    if model_mean_type.lower() in ['eps', 'epsilon']:
        model_mean_type = gd.ModelMeanType.EPSILON
    elif model_mean_type.lower() in ['xstart', 'x0', 'xo']:
        model_mean_type = gd.ModelMeanType.START_X
    elif model_mean_type == 'v':
        model_mean_type = gd.ModelMeanType.V
    else:
        raise NotImplementedError()

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=model_mean_type,
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        snr_splits=snr_splits,
        weight_schedule=weight_schedule
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

def add_dict_to_dict(add_dict, default_dict):
    for k, v in default_dict.items():
        if k in add_dict.keys():
            continue
        else:
            add_dict[k] = v
    return add_dict


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}

def dict_to_dict(args, keys):
    return {k: args[k] for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
