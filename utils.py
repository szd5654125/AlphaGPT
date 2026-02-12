from solana.rpc.async_api import AsyncClient
from solders.pubkey import Pubkey
from execution.config import ExecutionConfig
import torch


async def get_mint_decimals(mint_str: str, client: AsyncClient) -> int:
    if mint_str == ExecutionConfig.SOL_MINT:
        return 9
    try:
        pubkey = Pubkey.from_string(mint_str)
        resp = await client.get_account_info(pubkey)
        if resp.value is None:
            return 6
        resp_parsed = await client.get_account_info_json_parsed(pubkey)
        decimals = resp_parsed.value.data.parsed['info']['decimals']
        return int(decimals)
    except Exception:
        return 6

def shift1(x, fill=0.0):
    # x: [N, T]
    pad = torch.full((x.shape[0], 1), fill, device=x.device, dtype=x.dtype)
    return torch.cat([pad, x[:, :-1]], dim=1)


def _fmt_bytes(n: int) -> str:
    for u in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if n < 1024 or u == "TiB":
            return f"{n:.2f}{u}"
        n /= 1024
    return f"{n:.2f}B"

def tensor_nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()

def cuda_snapshot(tag: str, device: torch.device | None = None, extra: str = ""):
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if device.type != "cuda" or not torch.cuda.is_available():
        print(f"[MEM][{tag}] device={device} (cuda unavailable) {extra}")
        return

    idx = device.index if device.index is not None else torch.cuda.current_device()

    # 让统计更“真实”（尤其是你刚做完大张量搬运时）
    torch.cuda.synchronize(idx)

    free, total = torch.cuda.mem_get_info(idx)
    alloc = torch.cuda.memory_allocated(idx)
    reserved = torch.cuda.memory_reserved(idx)
    max_alloc = torch.cuda.max_memory_allocated(idx)

    print(
        f"[MEM][{tag}] cuda:{idx} "
        f"free={_fmt_bytes(free)} total={_fmt_bytes(total)} "
        f"alloc={_fmt_bytes(alloc)} reserved={_fmt_bytes(reserved)} peak={_fmt_bytes(max_alloc)} "
        f"{extra}"
    )

def dict_tensors_snapshot(tag: str, d: dict, device: torch.device | None = None, topk: int = 20):
    items = []
    total = 0
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            nb = tensor_nbytes(v)
            total += nb
            items.append((nb, k, tuple(v.shape), str(v.dtype), str(v.device)))
    items.sort(reverse=True)

    print(f"[TENS][{tag}] tensors={len(items)} total={_fmt_bytes(total)}")
    for nb, k, shape, dtype, dev in items[:topk]:
        print(f"  - {k:12s} {shape} {dtype} {dev} size={_fmt_bytes(nb)}")

    if device is not None:
        cuda_snapshot(tag + "::cuda", device=device)
