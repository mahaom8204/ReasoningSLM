import os, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from dataclasses import dataclass

# ⚡ CPU OPTIMIZATIONS
torch.set_num_threads(os.cpu_count() // 2) 
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "./compile_cache"

@dataclass
class Config:
    n_blocks          : int   = 20       
    vocab_size        : int   = 32000    
    hidden_size       : int   = 512
    max_seq_len       : int   = 1024     
    d_state           : int   = 16
    d_conv            : int   = 4
    mamba_expand      : int   = 2   
    n_heads           : int   = 16       
    n_experts         : int   = 8        
    top_k             : int   = 2
    expert_hidden     : int   = 2048     
    aux_loss_coef     : float = 0.001
    infini_seg_len    : int   = 256      
    infini_mem_dim    : int   = 512
    dropout           : float = 0.0 
    pad_id            : int   = 0
    rope_base         : float = 10000.0  

# ── Optimized Model Architecture with Caching ──────────────────────────
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, seq_len=None, offset=0):
        if seq_len is None: seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq) + offset
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().unsqueeze(0), emb.sin().unsqueeze(0)

def apply_rotary_pos_emb(q, k, cos, sin):
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    if q.dim() == 4:
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps, self.w = eps, nn.Parameter(torch.ones(d))
    def forward(self, x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w

class MambaLayer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        D, inner = cfg.hidden_size, cfg.hidden_size * cfg.mamba_expand
        self.dt_rank = math.ceil(D / 16) 
        self.in_proj  = nn.Linear(D, inner * 2, bias=False)
        self.conv1d   = nn.Conv1d(inner, inner, cfg.d_conv, padding=cfg.d_conv-1, groups=inner)
        self.x_proj   = nn.Linear(inner, self.dt_rank + cfg.d_state*2, bias=False)
        self.dt_proj  = nn.Linear(self.dt_rank, inner, bias=True)
        self.A_log    = nn.Parameter(torch.log(torch.arange(1, cfg.d_state+1).float().unsqueeze(0).expand(inner, -1)))
        self.D_ssm    = nn.Parameter(torch.ones(inner))
        self.out      = nn.Linear(inner, D, bias=False)
        self._inner, self._d_state = inner, cfg.d_state

    def forward(self, x, cache=None, layer_id=None):
        B, L, D = x.shape
        
        if cache is not None and layer_id is not None:
            if layer_id not in cache['mamba_conv']:
                cache['mamba_conv'][layer_id] = torch.zeros(B, self._inner, self.conv1d.kernel_size[0], device=x.device)
                cache['mamba_ssm'][layer_id] = torch.zeros(B, self._inner, self._d_state, device=x.device)
            
            # ⚡ FIX: Mathematically correct O(1) Decoding Math with Broadcasting
            if L == 1: 
                conv_state = cache['mamba_conv'][layer_id]
                xi, z = self.in_proj(x).chunk(2, dim=-1)
                
                # Update Conv Memory
                conv_state = torch.cat([conv_state[:, :, 1:], xi.transpose(1, 2)], dim=-1)
                cache['mamba_conv'][layer_id] = conv_state
                
                # Apply fast grouped 1D Convolution
                xc = F.conv1d(conv_state, self.conv1d.weight, self.conv1d.bias, groups=self._inner)
                xc = F.silu(xc).transpose(1, 2) # Shape: [B, 1, Inner]
                
                dt, Bs, Cs = self.x_proj(xc).split([self.dt_rank, self._d_state, self._d_state], -1)
                dt = F.softplus(self.dt_proj(dt)) # Shape: [B, 1, Inner]
                
                # Matrix Broadcasting for Memory Update
                A = -torch.exp(self.A_log.float()) # Shape: [Inner, 16]
                dA = torch.exp(dt.unsqueeze(-1) * A) # Shape: [B, 1, Inner, 16]
                dB = dt.unsqueeze(-1) * Bs.unsqueeze(2) # Shape: [B, 1, Inner, 16]
                
                h = cache['mamba_ssm'][layer_id]
                h = h * dA[:, 0] + dB[:, 0] * xc[:, 0].unsqueeze(-1)
                cache['mamba_ssm'][layer_id] = h
                
                y = torch.einsum('bdn,bn->bd', h, Cs[:, 0]).unsqueeze(1)
                return self.out(y * F.silu(z))

        # Prefill Step (Process entire prompt at once)
        xi, z = self.in_proj(x).chunk(2, dim=-1)
        xc = F.silu(self.conv1d(xi.transpose(1,2))[...,:L].transpose(1,2))
        dt, Bs, Cs = self.x_proj(xc).split([self.dt_rank, self._d_state, self._d_state], -1)
        dt = F.softplus(self.dt_proj(dt))
        dA = torch.exp(torch.einsum('bld,dn->bldn', dt, -torch.exp(self.A_log.float())))
        dB = torch.einsum('bld,bln->bldn', dt, Bs)
        h, ys = torch.zeros(B, self._inner, self._d_state, device=x.device, dtype=x.dtype), []
        for i in range(L):
            h = h * dA[:,i] + dB[:,i] * xc[:,i].unsqueeze(-1)
            ys.append(torch.einsum('bdn,bn->bd', h, Cs[:,i]))
            
        if cache is not None and layer_id is not None:
            pad_len = max(0, self.conv1d.kernel_size[0] - L)
            if pad_len > 0:
                pad = torch.zeros(B, self._inner, pad_len, device=x.device)
                cache['mamba_conv'][layer_id] = torch.cat([pad, xi.transpose(1, 2)], dim=-1)
            else:
                cache['mamba_conv'][layer_id] = xi[:, -self.conv1d.kernel_size[0]:].transpose(1, 2)
            cache['mamba_ssm'][layer_id] = h
            
        return self.out((torch.stack(ys, 1) + xc * self.D_ssm) * F.silu(z))

class SelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_heads, self.hd = cfg.n_heads, cfg.hidden_size // cfg.n_heads
        self.qkv, self.proj = nn.Linear(cfg.hidden_size, 3 * cfg.hidden_size, bias=False), nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        
    def forward(self, x, cos, sin, cache=None, layer_id=None):
        B, L, D = x.shape
        q, k, v = self.qkv(x).reshape(B, L, 3, self.n_heads, self.hd).permute(2,0,3,1,4).unbind(0)
        
        if cache is not None and layer_id is not None:
            if layer_id in cache['attn_k']:
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
                k = torch.cat([cache['attn_k'][layer_id], k], dim=2)
                v = torch.cat([cache['attn_v'][layer_id], v], dim=2)
                cache['attn_k'][layer_id], cache['attn_v'][layer_id] = k, v
            else:
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
                cache['attn_k'][layer_id], cache['attn_v'][layer_id] = k, v
        else:
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        is_causal = q.size(2) > 1 # Only apply causal mask on prefill
        y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        return self.proj(y.transpose(1,2).reshape(B, L, D))

class InfiniBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        D, self.seg, self.mdim = cfg.hidden_size, cfg.infini_seg_len, cfg.infini_mem_dim
        self.q, self.k, self.v = nn.Linear(D, D, bias=False), nn.Linear(D, D, bias=False), nn.Linear(D, D, bias=False)
        self.mp = nn.Linear(self.mdim, D, bias=False)
        self.beta, self.out = nn.Parameter(torch.zeros(1)), nn.Linear(D, D, bias=False)
        
    def forward(self, x, cos, sin, cache=None, layer_id=None):
        B, L, D = x.shape
        Q, K, V = self.q(x), self.k(x), self.v(x)
        
        if cache is not None and layer_id is not None:
            if layer_id in cache['infini_k']:
                Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
                K_full = torch.cat([cache['infini_k'][layer_id], K], dim=1)
                V_full = torch.cat([cache['infini_v'][layer_id], V], dim=1)
                cache['infini_k'][layer_id], cache['infini_v'][layer_id] = K_full, V_full
            else:
                Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
                K_full, V_full = K, V
                cache['infini_k'][layer_id], cache['infini_v'][layer_id] = K, V
        else:
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
            K_full, V_full = K, V

        is_causal = Q.size(1) > 1
        A_local = F.scaled_dot_product_attention(Q.unsqueeze(1), K_full.unsqueeze(1), V_full.unsqueeze(1), is_causal=is_causal).squeeze(1)
        
        M = torch.zeros(B, self.mdim, D, device=x.device)
        z = torch.ones(B, self.mdim, device=x.device)
        sQ = F.elu(Q) + 1
        Amem = self.mp(torch.einsum('bsd,bde->bse', sQ[..., :self.mdim], M) / (sQ[..., :self.mdim] * z.unsqueeze(1)).sum(-1, keepdim=True).clamp(min=1e-6))
        beta = torch.sigmoid(self.beta)
        
        return self.out(beta * Amem + (1 - beta) * A_local)

class Expert(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.gate, self.up, self.down = nn.Linear(cfg.hidden_size, cfg.expert_hidden, bias=False), nn.Linear(cfg.hidden_size, cfg.expert_hidden, bias=False), nn.Linear(cfg.expert_hidden, cfg.hidden_size, bias=False)
    def forward(self, x): return self.down(F.silu(self.gate(x)) * self.up(x))

class MoEFFN(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_exp, self.top_k = cfg.n_experts, cfg.top_k
        self.router, self.experts = nn.Linear(cfg.hidden_size, cfg.n_experts, bias=False), nn.ModuleList([Expert(cfg) for _ in range(cfg.n_experts)])
    def forward(self, x):
        B, L, D = x.shape
        xf, logits = x.reshape(-1, D), self.router(x.reshape(-1, D))
        probs = F.softmax(logits, dim=-1)
        topk_w, topk_i = probs.topk(self.top_k, dim=-1)
        topk_w = topk_w / topk_w.sum(-1, keepdim=True)
        out = torch.zeros_like(xf)
        for k in range(self.top_k):
            ei, ew = topk_i[:,k], topk_w[:,k].unsqueeze(-1)
            active_experts = torch.unique(ei)
            for e in active_experts:
                m = (ei == e)
                out[m] += ew[m] * self.experts[e](xf[m])
        return out.reshape(B, L, D), 0.0

class Block(nn.Module):
    def __init__(self, cfg: Config, layer_type: str):
        super().__init__()
        self.layer_type = layer_type
        self.norm1 = RMSNorm(cfg.hidden_size)
        if layer_type == 'mamba': self.core = MambaLayer(cfg)
        elif layer_type == 'attn': self.core = SelfAttention(cfg)
        elif layer_type == 'infini': self.core = InfiniBlock(cfg)
        self.norm2 = RMSNorm(cfg.hidden_size)
        self.moe = MoEFFN(cfg)

    def forward(self, x, cos=None, sin=None, cache=None, layer_id=None):
        res = x
        x_norm = self.norm1(x)
        if self.layer_type == 'mamba': x = self.core(x_norm, cache=cache, layer_id=layer_id)
        else: x = self.core(x_norm, cos, sin, cache=cache, layer_id=layer_id)
        x = res + x
        moe_out, _ = self.moe(self.norm2(x))
        return x + moe_out

class ReasoningSLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.rotary_attn = RotaryEmbedding(cfg.hidden_size // cfg.n_heads, base=cfg.rope_base)
        self.rotary_infini = RotaryEmbedding(cfg.hidden_size, base=cfg.rope_base)
        
        layers = []
        for _ in range(4): layers.extend(['mamba', 'mamba', 'attn', 'mamba', 'infini'])
        self.blocks = nn.ModuleList([Block(cfg, ltype) for ltype in layers])
        self.norm_f = RMSNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(self, input_ids, internal_loops=1, cache=None, offset=0):
        x = self.embed(input_ids)
        cos_attn, sin_attn = self.rotary_attn(x, seq_len=input_ids.shape[1], offset=offset)
        cos_inf, sin_inf = self.rotary_infini(x, seq_len=input_ids.shape[1], offset=offset)
        
        for loop in range(internal_loops):
            loop_res = x if internal_loops > 1 else 0
            for i, block in enumerate(self.blocks):
                c, s = (cos_attn, sin_attn) if block.layer_type == 'attn' else ((cos_inf, sin_inf) if block.layer_type == 'infini' else (None, None))
                x = block(x, c, s, cache=cache, layer_id=i)
            if internal_loops > 1:
                x = x + loop_res 
                
        return self.lm_head(self.norm_f(x))

# ── Smart Cache Generation Engine ──────────────────────────────────────
@torch.inference_mode()
def generate(model, tokenizer, prompt_text, is_deep_thinking=False, max_tokens=150, temperature=0.2, top_p=0.9):
    model.eval()
    
    formatted_prompt = f"### User: {prompt_text}\n\n### Assistant:"
    if is_deep_thinking:
        formatted_prompt += " <think>"
        print("\n[🧠 Deep Thinking Protocol Initiated...]")
        
    input_ids = tokenizer.encode(formatted_prompt).ids
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device="cpu")
    eos_id = tokenizer.token_to_id('<eos>') or 2
    
    loops = 2 if is_deep_thinking else 1
    in_thought_block = is_deep_thinking
    generated_tokens = []
    
    cache = {'mamba_conv': {}, 'mamba_ssm': {}, 'attn_k': {}, 'attn_v': {}, 'infini_k': {}, 'infini_v': {}}
    
    print("\nAssistant: ", end="")
    if is_deep_thinking:
        print("\n  [Thinking]: ", end="")
    
    # ⚡ PHASE 1: PREFILL
    logits = model(input_tensor, internal_loops=loops, cache=cache, offset=0)
    next_token_logits = logits[0, -1, :] / temperature
    
    probs = F.softmax(next_token_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    seq_length = input_tensor.shape[1]
    
    # ⚡ PHASE 2: FAST O(1) DECODE
    for step in range(max_tokens):
        token_id = next_token.item()
        if token_id == eos_id: break
            
        generated_tokens.append(token_id)
        token_str = tokenizer.decode([token_id])
        
        if token_str.strip() == "<think>":
            in_thought_block = True
            print("\n  [Thinking]: ", end="", flush=True)
        elif token_str.strip() == "</think>":
            in_thought_block = False
            print("\n\n[✅ Response]: ", end="", flush=True)
        else:
            if in_thought_block: print(f"\033[90m{token_str}\033[0m", end="", flush=True) 
            else: print(token_str, end="", flush=True)
        
        if len(generated_tokens) > 15 and len(set(generated_tokens[-10:])) <= 3:
            print("\n[Auto-Stopped: Repetition]")
            break
            
        logits = model(next_token.unsqueeze(0), internal_loops=1, cache=cache, offset=seq_length)
        seq_length += 1
        
        next_token_logits = logits[0, -1, :] / temperature
        
        for prev_id in set(generated_tokens[-15:]):
            next_token_logits[prev_id] -= 1.0 
        
        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
        next_token_logits[indices_to_remove] = float('-inf')
        
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
            
    print("\n")

if __name__ == "__main__":
    TOK_PATH = "reasoning_slm_tokenizer.json" 
    CKPT_PATH = "vi_demo_model_tuned.pt"      
    
    print("Loading CPU Inference Engine (FP32 Precision + Smart Cache)...")
    cfg = Config()
    tokenizer = Tokenizer.from_file(TOK_PATH)
    cfg.pad_id = tokenizer.token_to_id('<pad>') or 0
    
    model = ReasoningSLM(cfg)
    
    print(f"Loading Fine-Tuned Weights from {CKPT_PATH}...")
    torch.serialization.add_safe_globals([Config])
    checkpoint = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}

    for k, v in state_dict.items():
        name = k.replace("module.", "")
        if "core.mamba." in name: name = name.replace("core.mamba.", "core.")
        if name.endswith(".D"): name = name.replace(".D", ".D_ssm")
        if name.endswith(".out_proj.weight"): name = name.replace(".out_proj.weight", ".out.weight")
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict, strict=True)
    print("✅ Weights successfully translated and loaded!")
    
    print("\n" + "="*50)
    print("  DeepSynapse-M8 | Vi Customer Service Agent")
    print("  Modes: 'normal' | 'deep' | 'quit'")
    print("="*50)
    
    while True:
        mode = input("\nSelect Mode (normal/deep/quit): ").strip().lower()
        if mode in ['quit', 'exit']: break
        if mode not in ['normal', 'deep']: continue
            
        user_input = input("User: ")
        is_deep = (mode == 'deep')
        
        start_time = time.time()
        generate(model, tokenizer, user_input, is_deep_thinking=is_deep, max_tokens=250)
        print(f"⏱️ Generation time: {time.time() - start_time:.2f}s")