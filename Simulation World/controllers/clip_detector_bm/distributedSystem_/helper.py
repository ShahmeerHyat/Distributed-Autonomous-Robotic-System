import torch
from dataclasses import dataclass

@dataclass
class AttentionMeta:
    num_heads : int
    embed_dim : int

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads


def split_heads(tensor: torch.Tensor, head_assignments: list[int]) -> list[torch.Tensor]:
    """
    Splits (B, H, S, head_dim) along H by arbitrary head counts per device.
    
    head_assignments: e.g. [7, 5] means device 0 gets 7 heads, device 1 gets 5.
    Must sum to total heads.
    """
    assert sum(head_assignments) == tensor.shape[1], \
        f"head_assignments {head_assignments} must sum to {tensor.shape[1]}"
    
    return list(torch.split(tensor, head_assignments, dim=1))


def split_weights(out_proj_weight: torch.Tensor, head_assignments: list[int], head_dim: int) -> list[torch.Tensor]:
    """
    Splits out_proj weight [embed_dim, embed_dim] column-wise to match head assignments.
    Each device gets [embed_dim, assigned_heads * head_dim].
    """
    sizes = [h * head_dim for h in head_assignments]
    return list(torch.split(out_proj_weight, sizes, dim=1))


def merge_heads(head_parts: list[torch.Tensor], embed_dim: int) -> torch.Tensor:
    """
    Merges variable-sized head outputs back into hidden state.
    
    Each part : (B, H_i, S, head_dim)  -- H_i can differ per device
    Output    : (B, S, embed_dim)
    """
    # (B, H_total, S, head_dim)
    combined = torch.cat(head_parts, dim=1)
    B, H, S, hd = combined.shape

    # (B, S, embed_dim)
    return combined.transpose(1, 2).contiguous().view(B, S, embed_dim)


def allocate_heads(num_heads: int, scores: dict[str, float]) -> dict[str, int]:
    """
    Distributes heads across devices proportional to their scores.
    Handles rounding by giving remainder to highest scored device.
    
    scores: {"gpu_laptop": 0.7, "cpu_laptop": 0.3}
    Returns: {"gpu_laptop": 8, "cpu_laptop": 4}  for num_heads=12
    """
    total_score = sum(scores.values())
    normalized  = {d: s / total_score for d, s in scores.items()}
    
    # Floor allocation first
    allocation  = {d: int(w * num_heads) for d, w in normalized.items()}
    remainder   = num_heads - sum(allocation.values())
    
    # Give leftover heads to highest scored device
    if remainder > 0:
        best = max(normalized, key=normalized.get)
        allocation[best] += remainder
    
    return allocation

# ─────────────────────────────────────────────────────────────────────────────
# Model metadata  (CLIP vision encoder only)
# ─────────────────────────────────────────────────────────────────────────────

def get_model_metadata(vision_model) -> dict:
    """
    Extracts architecture constants from clip.vision_model.

    CLIP CLIPEncoderLayer layout:
        layer.self_attn          → CLIPAttention (q_proj / k_proj / v_proj / out_proj)
        layer.mlp.fc1 / mlp.fc2 → CLIPEncoderMLP
        layer.layer_norm1/2      → LayerNorm

    Returns
    -------
    dict with keys: embed_dim, num_heads, head_dim, mlp_hidden_dim, seq_length
    """
    layer = vision_model.encoder.layers[0]
    attn  = layer.self_attn   # CLIPAttention

    # q_proj weight shape: (embed_dim, embed_dim)  — square for self-attention
    embed_dim     = attn.q_proj.weight.shape[1]
    num_heads     = attn.num_heads
    mlp_hidden    = layer.mlp.fc1.weight.shape[0]   # fc1: (mlp_hidden, embed)

    # CLIP: positional embedding lives in vision_model.embeddings
    seq_length = vision_model.embeddings.position_embedding.weight.shape[0]

    return {
        "embed_dim":      embed_dim,
        "num_heads":      num_heads,
        "head_dim":       embed_dim // num_heads,
        "mlp_hidden_dim": mlp_hidden,
        "seq_length":     seq_length,   # 50 for ViT-B/32 (49 patches + 1 CLS)
    }


# ─────────────────────────────────────────────────────────────────────────────
# Head-space helpers
# ─────────────────────────────────────────────────────────────────────────────

def to_head_space(x: torch.Tensor, h_range: range, num_heads: int, head_dim: int) -> torch.Tensor:
    """
    Slice a projected tensor into a specific head range and reshape for
    multi-head attention math.

    Parameters
    ----------
    x        : (B, S, embed_dim)   — output of q_proj / k_proj / v_proj
    h_range  : range(start, end)   — which heads to keep
    num_heads: total number of attention heads
    head_dim : embed_dim // num_heads

    Returns
    -------
    (B, len(h_range), S, head_dim)  — ready for scaled dot-product attention
    """
    B, S, _ = x.shape
    x = x.view(B, S, num_heads, head_dim)     # (B, S, H, hd)
    x = x[:, :, h_range, :]                   # (B, S, H_slice, hd)
    return x.transpose(1, 2)                  # (B, H_slice, S, hd)

# ─────────────────────────────────────────────────────────────────────────────
# MLP merge helper  (used by master after collecting partial MLP outputs)
# ─────────────────────────────────────────────────────────────────────────────

def sum_mlp_parts(parts: list) -> torch.Tensor:
    """
    Each element of `parts` is a (B, S, embed_dim) tensor representing the
    contribution of a neuron slice: gelu(x @ w1_slice.t() + b1_slice) @ w2_slice.t()

    Summing them reconstructs the full fc2 output (bias is added by caller).
    """
    non_empty = [p for p in parts if p is not None and p.numel() > 0]
    if not non_empty:
        raise RuntimeError("sum_mlp_parts: all parts are empty — no MLP work was done.")
    return torch.stack(non_empty, dim=0).sum(dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# Legacy helpers kept so nothing explodes if imported elsewhere
# ─────────────────────────────────────────────────────────────────────────────

def merge_n_projections(projections):
    """Kept for backwards compatibility. Not used in the CLIP ATTN path."""
    all_q, all_k, all_v = [], [], []
    for p in projections:
        if p.numel() == 0:
            continue
        q, k, v = torch.chunk(p, 3, dim=-1)
        all_q.append(q); all_k.append(k); all_v.append(v)
    if not all_q:
        return torch.tensor([])
    return torch.cat(
        [torch.cat(all_q, -1), torch.cat(all_k, -1), torch.cat(all_v, -1)], -1
    )