import torch
import torch.nn as nn
import time

# Перевірка пристрою
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Використовується пристрій: {device}")

# Параметри
d_model = 512
num_heads = 8
batch_size = 2
seq_len = 4096          # довжина контексту
runs = 5                # скільки разів запускати для середнього часу

# 1. Базова Multi-Head Attention (MHA)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.proj_q = nn.Linear(d_model, d_model)
        self.proj_k = nn.Linear(d_model, d_model)
        self.proj_v = nn.Linear(d_model, d_model)
        self.proj_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch, seq, _ = x.shape
        q = self.proj_q(x).view(batch, seq, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        k = self.proj_k(x).view(batch, seq, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        v = self.proj_v(x).view(batch, seq, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = nn.functional.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous().view(batch, seq, -1)
        return self.proj_o(out)

# 2. Grouped Query Attention (GQA)
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_k = d_model // num_heads
        self.proj_q = nn.Linear(d_model, d_model)
        self.proj_k = nn.Linear(d_model, num_kv_heads * self.d_k)
        self.proj_v = nn.Linear(d_model, num_kv_heads * self.d_k)
        self.proj_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch, seq, _ = x.shape
        q = self.proj_q(x).view(batch, seq, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        k = self.proj_k(x).view(batch, seq, self.num_kv_heads, self.d_k).permute(0, 2, 1, 3)
        v = self.proj_v(x).view(batch, seq, self.num_kv_heads, self.d_k).permute(0, 2, 1, 3)
        k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = nn.functional.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous().view(batch, seq, -1)
        return self.proj_o(out)

# 3. Multi-Latent Attention (MLA)
class MultiLatentAttention(nn.Module):
    def __init__(self, d_model, num_heads, compression_ratio=4):
        super().__init__()
        self.num_heads = num_heads
        self.latent_dim = d_model // compression_ratio
        self.proj_q = nn.Linear(d_model, d_model)
        self.proj_k_latent = nn.Linear(d_model, self.latent_dim)
        self.proj_v_latent = nn.Linear(d_model, self.latent_dim)
        self.proj_k = nn.Linear(self.latent_dim, d_model)
        self.proj_v = nn.Linear(self.latent_dim, d_model)
        self.proj_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch, seq, _ = x.shape
        q = self.proj_q(x).view(batch, seq, self.num_heads, -1).permute(0, 2, 1, 3)
        k_lat = self.proj_k_latent(x)
        v_lat = self.proj_v_latent(x)
        k = self.proj_k(k_lat).view(batch, seq, self.num_heads, -1).permute(0, 2, 1, 3)
        v = self.proj_v(v_lat).view(batch, seq, self.num_heads, -1).permute(0, 2, 1, 3)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
        attn = nn.functional.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous().view(batch, seq, -1)
        return self.proj_o(out)

# Функція для вимірювання
def measure_model(model, name):
    model = model.to(device)
    model.eval()

    x = torch.randn(batch_size, seq_len, d_model, device=device)

    # Скидаємо статистику пікової пам'яті (тільки для CUDA)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    mem_before = 0.0
    if device.type == 'cuda':
        mem_before = torch.cuda.memory_allocated() / (1024 ** 2)

    total_time = 0.0
    for _ in range(runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        total_time += time.perf_counter() - start

    avg_time = total_time / runs

    # Пікова пам'ять
    peak_mem_mb = 0.0
    if device.type == 'cuda':
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        current_mem_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    else:
        # Fallback для CPU (менш точний)
        import psutil
        import os
        process = psutil.Process(os.getpid())
        peak_mem_mb = process.memory_info().rss / (1024 ** 2)
        current_mem_mb = peak_mem_mb  # приблизно

    print(f"\n{name}:")
    print(f"  Середній час forward: {avg_time:.4f} с")
    if device.type == 'cuda':
        print(f"  Пам'ять на початку:   {mem_before:.1f} MB")
        print(f"  Пікова пам'ять (CUDA): {peak_mem_mb:.1f} MB")
    else:
        print(f"  Приблизна пам'ять процесу (RSS): {peak_mem_mb:.1f} MB")

    return avg_time, peak_mem_mb

# Запуск експериментів
print("\nЗапуск порівняння (seq_len = {}, batch = {})".format(seq_len, batch_size))

mha = MultiHeadAttention(d_model, num_heads)
gqa = GroupedQueryAttention(d_model, num_heads, num_kv_heads=4)     # 4 групи KV
mla = MultiLatentAttention(d_model, num_heads, compression_ratio=4)

measure_model(mha, "Стандартна MHA")
measure_model(gqa, "Grouped Query Attention (GQA)")
measure_model(mla, "Multi-Latent Attention (MLA)")

# Простий тест точності (імітація Needle-in-a-Haystack)
def simple_needle_test(model, pos=seq_len//2):
    model.eval()
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    with torch.no_grad():
        out = model(x)
    # Імітуємо "точність" - чи близький вихід до певного патерну (dummy)
    target = torch.zeros_like(out)
    target[:, pos, :] = 1.0
    acc = (out - target).abs().mean().item()  # чим менше - тим "краще" (dummy метрика)
    return 100 - min(acc * 1000, 100)  # просто для прикладу

print("\nПриблизна точність на dummy Needle-in-a-Haystack (чим вище — тим краще):")
print("MHA:", simple_needle_test(mha))
print("GQA:", simple_needle_test(gqa))
print("MLA:", simple_needle_test(mla))