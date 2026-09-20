import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ==========================================
# 1. LLaMA core components: RMSNorm & RoPE
# ==========================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.cos(freqs), torch.sin(freqs)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
):
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    freqs_cos = freqs_cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim/2]
    freqs_sin = freqs_sin.unsqueeze(0).unsqueeze(2)

    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# ==========================================
# 2. Core components of LLaMA: Attention & SwiGLU
# ==========================================
class LlamaAttention(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.1):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        self.dropout_p = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, freqs_cos, freqs_sin):
        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cos[:seqlen], freqs_sin[:seqlen])

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)

        mask = torch.tril(torch.ones(seqlen, seqlen, device=x.device)).view(1, 1, seqlen, seqlen)
        scores = scores.masked_fill(mask == 0, float("-inf"))

        scores = F.softmax(scores, dim=-1)
        scores = F.dropout(scores, p=self.dropout_p, training=self.training)

        output = torch.matmul(scores, xv)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        return self.resid_dropout(self.wo(output))


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, hidden_dim, dropout=0.1):
        super().__init__()
        self.attention = LlamaAttention(dim, n_heads, dropout)
        self.feed_forward = SwiGLU(dim, hidden_dim, dropout)
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

    def forward(self, x, freqs_cos, freqs_sin):
        h = x + self.attention(self.attention_norm(x), freqs_cos, freqs_sin)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


# ==========================================
# 3. DarwinLLaMA Master Model (Two-Condition Pure Version: EST + SAScore)
# ==========================================
class DarwinLLaMA(nn.Module):
    """
    Pure two-condition mechanism
    1. Continuous property (EST, SAScore) -> prefix embedding (length 2)
    2. During training, prop_mask can be passed in to control which properties are effective
    3. For masked properties, use learnable null prefix
    """
    def __init__(
        self,
        vocab_size,
        prop_len=2,  
        dim=512,
        n_layers=4,
        n_heads=8,
        max_seq_len=300,
        dropout=0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.prop_len = prop_len

        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.dropout = nn.Dropout(dropout)

        # ==========================================
        # 🌟 Thoroughly purify it into a dual-attribute mapping layer
        # ==========================================
        if self.prop_len == 2:
            self.est_proj = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
            self.sa_proj = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
            self.null_prefix = nn.Parameter(torch.zeros(2, dim))
        elif self.prop_len != 0:
            raise ValueError("🚀 This model has been hard-coded for dual-condition (EST, SAScore) specific use, and prop_len must be either 2 or 0!")

        hidden_dim = int(2 * (4 * dim) / 3)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, hidden_dim, dropout)
            for _ in range(n_layers)
        ])

        self.norm = RMSNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)

        # Weight tying
        self.tok_embeddings.weight = self.output.weight

        freqs_cos, freqs_sin = precompute_freqs_cis(dim // n_heads, max_seq_len * 2)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        self.apply(self._init_weights)

        # Make the null prefix initially closer to the "empty signal"
        if hasattr(self, "null_prefix"):
            nn.init.normal_(self.null_prefix, mean=0.0, std=0.02)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _build_prefix_embeddings(
        self,
        props: Optional[torch.Tensor],
        prop_mask: Optional[torch.Tensor]
    ):
        """
        props:     [B, 2]
        prop_mask: [B, 2], 1 indicates that the property is visible, and 0 indicates that it has been masked
        return:
            prop_emb:   [B, 2, dim]
            prefix_len: 2
        """
        if self.prop_len == 0 or props is None:
            return None, 0

        bsz = props.size(0)
        device = props.device

        if prop_mask is None:
            prop_mask = torch.ones_like(props, device=device)

        # ==========================================
        # 🌟 彻底净化为双属性前缀拼接
        # ==========================================
        if self.prop_len == 2:
            est_emb = self.est_proj(props[:, 0:1]).unsqueeze(1)  # [B, 1, D]
            sa_emb  = self.sa_proj(props[:, 1:2]).unsqueeze(1)   # [B, 1, D]

            prop_emb = torch.cat([est_emb, sa_emb], dim=1)       # [B, 2, D]

            mask_expand = prop_mask.unsqueeze(-1).float()        # [B, 2, 1]
            null_prefix = self.null_prefix.unsqueeze(0).expand(bsz, -1, -1)  # [B, 2, D]

            prop_emb = mask_expand * prop_emb + (1.0 - mask_expand) * null_prefix
            return prop_emb, 2
        else:
             raise ValueError("🚀 This model has been hard-coded as dual-condition dedicated, and the input feature dimension is incorrect!")

    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        props: Optional[torch.Tensor] = None,
        prop_mask: Optional[torch.Tensor] = None
    ):
        bsz, seqlen = tokens.shape

        h = self.tok_embeddings(tokens)
        h = self.dropout(h)

        prefix_len = 0
        prop_emb, prefix_len = self._build_prefix_embeddings(props, prop_mask)

        if prop_emb is not None:
            h = torch.cat([prop_emb, h], dim=1)

        for layer in self.layers:
            h = layer(h, self.freqs_cos, self.freqs_sin)

        h = self.norm(h)
        logits = self.output(h)

        loss = None
        if targets is not None:
            valid_logits = logits[:, prefix_len:, :].contiguous()
            loss = F.cross_entropy(
                valid_logits.reshape(-1, self.vocab_size),
                targets.reshape(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        batch_size,
        char_dict,
        device,
        props=None,
        prop_mask=None,
        init_seq=None,
        banned_tokens_step_0=None,
        max_length=240,
        temperature=0.7,
        top_k=5
    ):
        self.eval()
        start_idx = char_dict['^']
        end_idx = char_dict['>']

        if init_seq is not None:
            idx = init_seq.repeat(batch_size, 1).to(device)
            current_len = idx.shape[1]
        else:
            idx = torch.full((batch_size, 1), start_idx, dtype=torch.long, device=device)
            current_len = 1

        finish = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for step in range(max_length - current_len):
            logits, _ = self.forward(
                tokens=idx,
                props=props,
                prop_mask=prop_mask
            )
            next_token_logits = logits[:, -1, :]

            if step == 0 and banned_tokens_step_0 is not None:
                for ban_tok in banned_tokens_step_0:
                    next_token_logits[:, ban_tok] = float('-inf')

            next_token_logits = next_token_logits / temperature

            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
            probs = F.softmax(top_k_logits, dim=-1)

            next_token_idx = torch.multinomial(probs, num_samples=1)
            next_token = torch.gather(top_k_indices, -1, next_token_idx)

            next_token[finish.unsqueeze(-1)] = end_idx
            idx = torch.cat((idx, next_token), dim=1)

            eos_sampled = (next_token.squeeze(-1) == end_idx)
            finish = torch.logical_or(finish, eos_sampled)

            if torch.all(finish):
                break

        return idx[:, 1:]