"""
Initial Latent AdaIN (color & tone initialization) for training-free style
transfer, from:

  "Style Injection in Diffusion: A Training-free Approach for Adapting
  Large-scale Diffusion Models for Style Transfer" (StyleID), CVPR 2024.
  https://arxiv.org/abs/2312.09008

This module implements only the *Initial Latent AdaIN* component of that
paper (Eq. in Sec 3.3 of the paper) -- not the attention-injection /
attention-temperature-scaling components, which are a separate mechanism
operating inside the model's self-attention layers and are not implemented
here.

What Initial Latent AdaIN actually does:
  1. Take a content input and a style input (both real images, or here,
     real video clips) and DDIM-invert each into the noise latent the
     model would have produced them from (a deterministic, training-free
     operation -- see GaussianDiffusion.ddim_reverse_sample_loop).
  2. Renormalize the *content* noise latent's per-channel mean/std to match
     the *style* noise latent's per-channel mean/std (this is exactly
     AdaIN, applied to the initial latent rather than to intermediate
     features as in the original AdaIN paper).
  3. Run ordinary (forward) DDIM sampling starting from this renormalized
     latent, conditioned on whatever text prompt you want.

Because low-frequency color/tone information is concentrated in the coarse
statistics of the initial latent, this single step transfers the *style
image's overall color and tone* onto the *content image's structure* before
any denoising happens -- without training anything.

Video generalization: the original paper operates on single images shaped
(C, H, W), so AdaIN statistics are computed per-channel over (H, W). GenTron
latents are (B, C, F, H, W) -- one extra frame dimension. We compute
statistics over (F, H, W) jointly, i.e. one global color/tone statistic per
channel for the whole clip, which is the direct video analogue of "one
statistic per channel for the whole image." This is a generalization we're
making explicitly, not something asserted by the original (image-only)
paper.
"""

import torch


def adain_latent(content_latent: torch.Tensor, style_latent: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Apply AdaIN: renormalize `content_latent` so its per-channel mean/std
    match `style_latent`'s per-channel mean/std, while keeping the
    content's own spatial (and temporal, for video) structure.

    AdaIN(c, s) = std(s) * (c - mean(c)) / std(c) + mean(s)

    Args:
        content_latent: (B, C, F, H, W) or (B, C, H, W) -- the initial noise
            latent recovered from inverting the content input.
        style_latent: same shape as content_latent -- the initial noise
            latent recovered from inverting the style input.
        eps: small constant to avoid division by zero for near-constant
            channels.

    Returns:
        A tensor the same shape as content_latent, with the style's
        per-channel mean/std transferred onto the content's structure.
    """
    assert content_latent.shape == style_latent.shape, (
        f"content_latent and style_latent must have the same shape to compute "
        f"matching per-channel statistics, got {content_latent.shape} vs {style_latent.shape}"
    )

    # Reduce over every dim except batch (0) and channel (1): that's (F, H, W)
    # for video latents, or (H, W) for image latents -- either way "every
    # spatial/temporal position", matching the original AdaIN's per-channel,
    # per-sample statistics.
    reduce_dims = tuple(range(2, content_latent.dim()))

    content_mean = content_latent.mean(dim=reduce_dims, keepdim=True)
    content_std = content_latent.std(dim=reduce_dims, keepdim=True)
    style_mean = style_latent.mean(dim=reduce_dims, keepdim=True)
    style_std = style_latent.std(dim=reduce_dims, keepdim=True)

    normalized_content = (content_latent - content_mean) / (content_std + eps)
    return normalized_content * (style_std + eps) + style_mean


@torch.no_grad()
def invert_to_noise(diffusion, model, latent, model_kwargs, device=None, progress=False):
    """
    Convenience wrapper: DDIM-invert a real (VAE-encoded) latent into the
    noise the model would have produced it from. Thin wrapper around
    GaussianDiffusion.ddim_reverse_sample_loop so callers don't need to
    import the diffusion internals directly.

    Args:
        diffusion: a GaussianDiffusion / SpacedDiffusion instance (e.g. from
            diffusion.create_diffusion(...)).
        model: the denoising model (e.g. a GenTronT2V/GenTronT2I instance).
        latent: the real, VAE-encoded latent to invert -- (B, C, F, H, W)
            for video or (B, C, H, W) for images.
        model_kwargs: conditioning dict (e.g. dict(y=text_embeddings,
            mask=attention_mask)) -- must match whatever the model expects.
        device: device to run inversion on; defaults to the model's device.
        progress: show a tqdm progress bar over inversion steps.

    Returns:
        The inverted noise latent, same shape as `latent`.
    """
    return diffusion.ddim_reverse_sample_loop(
        model,
        latent,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        device=device,
        progress=progress,
    )
