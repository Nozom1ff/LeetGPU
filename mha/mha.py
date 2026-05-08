import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_state, casual_mask=None, kv_cache=None, use_cache=False):
        batch_size = hidden_state.size(0)
        query = self.q_proj(hidden_state)
        key = self.k_proj(hidden_state)
        value = self.v_proj(hidden_state)

        query = query.view(batch_size, -1, self.num_heads, self.head_dim)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim)

        if kv_cache is not None:
            past_k, pask_v = kv_cache
            key = torch.cat([pask_k, key], dim=2)
            value = torch.cat([pask_v, value], dim=2)
        new_kv_cache = (key, value) if use_cache else None
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(
            self.head_dim, dtype=torch.float32
        )

        if casual_mask is not None:
            attention_scores += casual_mask * 1e-9
        attention_probs = F.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_probs, value)

        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.num_heads * self.head_dim)
        )
        output = self.o_proj(output)

        return (output, new_kv_cache) if use_cache else output
