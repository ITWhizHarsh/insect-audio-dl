"""Model D - Audio Spectrogram Transformer (Gong, Chung & Glass, Interspeech 2021).

The log-mel image is cut into 16 x 16 patches, each patch becomes a 768-d
token, and 12 transformer blocks let every token attend to every other one.
A CNN only relates two parts of the spectrogram once its receptive field has
grown large enough to cover both; self-attention can link a pulse at one
time and frequency to its harmonics or to the next pulse of the same call
anywhere in the 5 s window from the first layer on. The price is a weak
inductive bias: a transformer has to learn locality from data.

That is why the AudioSet-finetuned weights matter. With ~86 M parameters and
only a few thousand training chunks, a randomly initialised AST would mostly
memorise the training set. The released model was trained on ImageNet and
then on ~2 M AudioSet clips, including the classes Insect and Cricket, so its
patch filters and attention patterns already describe sound; we only adapt
it. The weights are those of HuggingFace 'MIT/ast-finetuned-audioset-10-10-0.4593',
loaded into our own compact implementation (HF's version crashes in backward
on Apple MPS). The AudioSet 527-way classifier is replaced by a new head.

Differences from the pretraining setup and how they are handled:
- AST saw 10 s clips (1024 frames, a 12 x 101 patch grid). Our chunks have
  431 frames, so the time axis of the position embeddings is cut to its
  centre, as in AST's own code; a different stride interpolates it bilinearly.
- AST's input was Kaldi fbank at 16 kHz normalised to mean 0, std 0.5. Ours
  is 44.1 kHz log-mel (50 Hz - 22.05 kHz) standardised to std 1, so forward
  multiplies the input by 0.5. The mel axis still spans a wider frequency
  range than in pretraining; the low-level patch filters transfer, the rest
  is left to fine-tuning.

Cost on an 8 GB laptop versus a cloud GPU (one 128 x 431 chunk):
- stride 10 (overlapping patches, as pretrained) gives 12 x 42 = 504 tokens;
  stride 16 (no overlap) gives 8 x 26 = 208, about 2.5x less compute and
  activation memory, at some accuracy cost because the weights were learned
  with overlap.
- patchout (PaSST, Koutini et al. 2022) drops a random fraction of patch
  tokens per example during training only. It regularises like dropout and
  makes training steps cheaper; evaluation always uses every token.
- freeze_blocks=k keeps the patch embedding, the cls / distillation tokens,
  the position embeddings and the first k blocks fixed and runs them without
  autograd, so they store no activations, gradients or optimiser state
  (k = 6 halves the trainable parameters to 42.6 M). checkpoint=True
  recomputes block activations in the backward pass instead of storing them
  (~30% more compute, much less memory).
On a cloud GPU we fine-tune everything at stride 10; on the laptop a higher
stride, patchout and frozen lower blocks make a run fit.

About 85.7 M parameters with a 40-class head at stride 10 (the original 10 s
backbone has 86.2 M); 85.1 M of them are in the 12 blocks.
"""
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

HF_REPO = "MIT/ast-finetuned-audioset-10-10-0.4593"
DIM, DEPTH, HEADS, MLP_DIM, PATCH = 768, 12, 12, 3072, 16
PRE_GRID = (12, 101)          # patch grid of the pretrained model: 128 mel bins x 1024 frames, stride 10


def n_patches(size, stride):
    return (size - PATCH) // stride + 1


class Block(nn.Module):
    """Pre-norm ViT block: x + attention(norm(x)), then x + MLP(norm(x))."""

    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(DIM, eps=1e-12)
        self.qkv = nn.Linear(DIM, 3 * DIM)
        self.proj = nn.Linear(DIM, DIM)
        self.norm2 = nn.LayerNorm(DIM, eps=1e-12)
        self.fc1 = nn.Linear(DIM, MLP_DIM)
        self.fc2 = nn.Linear(MLP_DIM, DIM)

    def forward(self, x):                      # x: (B, N, DIM)
        B, N, _ = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(B, N, 3, HEADS, DIM // HEADS).permute(2, 0, 3, 1, 4)
        x = x + self.proj(attention(q, k, v).transpose(1, 2).reshape(B, N, DIM))
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x))))


def attention(q, k, v):
    """softmax(q k^T / sqrt(d)) v for each head -> (B, HEADS, N, DIM / HEADS).

    On Apple MPS keep it in float32: float16 autocast gives NaN there in torch 2.5."""
    return F.scaled_dot_product_attention(q, k, v)


class AST(nn.Module):
    def __init__(self, n_classes, n_mels=128, n_frames=431, stride=10, patchout=0.0,
                 freeze_blocks=0, pretrained=True, dropout=0.0, checkpoint=False):
        super().__init__()
        self.grid = (n_patches(n_mels, stride), n_patches(n_frames, stride))
        self.patchout, self.freeze_blocks, self.checkpoint = patchout, freeze_blocks, checkpoint
        self.embed_dim = DIM

        self.patch = nn.Conv2d(1, DIM, PATCH, stride=(stride, stride))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, DIM))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, DIM))
        self.pos_embed = nn.Parameter(torch.zeros(1, 2 + self.grid[0] * self.grid[1], DIM))
        self.blocks = nn.ModuleList([Block() for _ in range(DEPTH)])
        self.norm = nn.LayerNorm(DIM, eps=1e-12)
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(DIM, n_classes))

        if pretrained:
            self.load_audioset(n_mels, n_frames, stride)
        else:
            for p in (self.cls_token, self.dist_token, self.pos_embed):
                nn.init.trunc_normal_(p, std=0.02)

        if freeze_blocks:
            for m in (self.patch, *self.blocks[:freeze_blocks]):
                m.requires_grad_(False)
            for p in (self.cls_token, self.dist_token, self.pos_embed):
                p.requires_grad = False

    def load_audioset(self, n_mels, n_frames, stride):
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        sd = from_hf(load_file(hf_hub_download(HF_REPO, "model.safetensors")))
        pos = sd["pos_embed"]
        sd["pos_embed"] = torch.cat([pos[:, :2], adapt_pos_embed(pos[:, 2:], n_mels, n_frames, stride)], 1)
        missing, unexpected = self.load_state_dict(sd, strict=False)
        assert not unexpected and all(k.startswith("fc.") for k in missing), (missing, unexpected)

    def tokens(self, x):
        """(B, 1, mels, frames) -> (B, 2 + kept patches, DIM): patch tokens with positions, patchout, cls tokens."""
        p = self.patch(x * 0.5)                # x * 0.5: AST was pretrained on inputs with std 0.5
        if p.requires_grad:
            p.register_hook(lambda g: g.contiguous())   # MPS conv backward fails on the transposed grad
        h = p.flatten(2).transpose(1, 2) + self.pos_embed[:, 2:]        # freq-major token order, as in HF
        if self.training and self.patchout > 0:
            B, N, _ = h.shape
            keep = max(1, round((1 - self.patchout) * N))
            idx = torch.rand(B, N, device=h.device).argsort(1)[:, :keep]
            h = h.gather(1, idx.unsqueeze(-1).expand(-1, -1, DIM))
        special = torch.cat([self.cls_token, self.dist_token], 1) + self.pos_embed[:, :2]
        return torch.cat([special.expand(len(h), -1, -1), h], 1)

    def embed(self, x):                        # x: (B, 1, mels, frames)
        k = self.freeze_blocks
        with torch.no_grad() if k and self.training else nullcontext():
            h = self.tokens(x)
            for blk in self.blocks[:k]:
                h = blk(h)
        for blk in self.blocks[k:]:
            if self.checkpoint and self.training:
                h = torch.utils.checkpoint.checkpoint(blk, h, use_reentrant=False)
            else:
                h = blk(h)
        h = self.norm(h)
        return (h[:, 0] + h[:, 1]) / 2         # mean of cls and distillation token (HF pooler_output)

    def forward(self, x):
        return self.fc(self.embed(x))


def adapt_pos_embed(pos, n_mels, n_frames, stride):
    """Fit the pretrained (1, 12 * 101, DIM) patch position embeddings to our input size and stride.

    First the 12 x 101 grid is brought to the stride-10 grid of our input
    (centre slice if shorter, bilinear stretch if longer, per axis, as in
    AST's own code), then it is interpolated to the grid of the actual stride.
    """
    g = pos.reshape(1, *PRE_GRID, -1).permute(0, 3, 1, 2)          # (1, DIM, 12, 101)
    g = fit_axis(fit_axis(g, n_patches(n_frames, 10), 3), n_patches(n_mels, 10), 2)
    target = (n_patches(n_mels, stride), n_patches(n_frames, stride))
    if tuple(g.shape[2:]) != target:
        g = F.interpolate(g, size=target, mode="bilinear", align_corners=False)
    return g.flatten(2).transpose(1, 2)        # (1, f * t, DIM)


def fit_axis(g, n, axis):
    old = g.shape[axis]
    if n <= old:
        start = old // 2 - n // 2
        return g.narrow(axis, start, n)
    size = list(g.shape[2:])
    size[axis - 2] = n
    return F.interpolate(g, size=size, mode="bilinear", align_corners=False)


def from_hf(sd):
    """Rename the tensors of HF's ASTForAudioClassification to this module's names (q, k, v fused)."""
    p = "audio_spectrogram_transformer."
    out = {
        "patch.weight": sd[p + "embeddings.patch_embeddings.projection.weight"],
        "patch.bias": sd[p + "embeddings.patch_embeddings.projection.bias"],
        "cls_token": sd[p + "embeddings.cls_token"],
        "dist_token": sd[p + "embeddings.distillation_token"],
        "pos_embed": sd[p + "embeddings.position_embeddings"],
        "norm.weight": sd[p + "layernorm.weight"],
        "norm.bias": sd[p + "layernorm.bias"],
    }
    names = {"norm1": "layernorm_before", "proj": "attention.output.dense", "norm2": "layernorm_after",
             "fc1": "intermediate.dense", "fc2": "output.dense"}
    for i in range(DEPTH):
        b = f"{p}encoder.layer.{i}."
        for t in ("weight", "bias"):
            out[f"blocks.{i}.qkv.{t}"] = torch.cat(
                [sd[f"{b}attention.attention.{n}.{t}"] for n in ("query", "key", "value")])
            for ours, theirs in names.items():
                out[f"blocks.{i}.{ours}.{t}"] = sd[f"{b}{theirs}.{t}"]
    return out
