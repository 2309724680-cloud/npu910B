"""按目标 token 数构造 prompt，并控制 Prefix Cache 复用。

硬约束：禁止所有请求复用同一 Prompt，否则 Prefix Cache
命中率虚高、TTFT 偏低，得到的容量数据不成立。默认 unique 模式给每个请求
一段唯一前缀，shared / mixed 用于专门测 Prefix Cache 收益。
"""
from __future__ import annotations

import random

import httpx

from .config import TargetConfig

# 中文语料池。用中文而非 lorem ipsum，因为中文 token 密度与生产流量接近，
# 且能暴露 tokenizer 对多字节字符的处理差异。
_CORPUS = (
    "在分布式推理系统中，调度器需要在吞吐与延迟之间做出权衡。",
    "键值缓存的容量决定了单节点可以承载的最大并发会话数量。",
    "张量并行将单层权重切分到多张加速卡上，通过集合通信汇总结果。",
    "专家并行把不同专家分布到不同设备，路由器决定每个词元激活哪些专家。",
    "投机解码用小模型先草拟若干候选词元，再由主模型一次性校验。",
    "长上下文场景下注意力计算的开销随序列长度平方增长。",
    "量化把权重从十六位浮点压缩到八位整数，显著降低显存占用。",
    "预填充阶段是计算密集的，解码阶段则受显存带宽限制。",
    "连续批处理允许新请求随时加入正在执行的批次，提升设备利用率。",
    "当排队请求数持续增长时，说明系统已越过可持续服务的容量上限。",
)


class TokenCounter:
    """token 计数。优先用服务端 /tokenize，保证与服务端计费口径一致。

    /tokenize 不可用时退化为字符数估算，此时记录的 input_tokens 是近似值，
    报告里必须标注来源，不能与服务端 usage 混为一谈。
    """

    def __init__(self, target: TargetConfig, client: httpx.Client | None = None):
        self.target = target
        self._client = client or httpx.Client(timeout=30.0)
        self._own_client = client is None
        self.available = target.has_tokenize
        self._cache: dict[str, int] = {}

    def count(self, text: str) -> int:
        if not self.available:
            # 中文约 1.5 字/token，英文约 4 字符/token，混合按 1.7 折中
            return max(1, int(len(text) / 1.7))
        if text in self._cache:
            return self._cache[text]
        try:
            r = self._client.post(
                self.target.tokenize_url,
                headers=self.target.headers(),
                json={"model": self.target.model, "prompt": text},
            )
            r.raise_for_status()
            n = int(r.json()["count"])
        except Exception:
            self.available = False
            return max(1, int(len(text) / 1.7))
        self._cache[text] = n
        return n

    def close(self) -> None:
        if self._own_client:
            self._client.close()


def _grow_to(target_tokens: int, counter: TokenCounter,
             seed_text: str, rng: random.Random) -> str:
    """拼接语料直到达到目标 token 数。

    先按估算比例批量拼接再微调，避免每加一句都调一次 /tokenize
    （4K 输入会产生上百次网络往返）。
    """
    text = seed_text
    n = counter.count(text)
    if n >= target_tokens:
        return text

    # 用当前样本估算每句 token 数，批量补足
    probe = _CORPUS[0]
    per = max(1, counter.count(probe))
    need = target_tokens - n
    batch = max(1, need // per)
    parts = [text]
    for _ in range(batch):
        parts.append(rng.choice(_CORPUS))
    text = "".join(parts)
    n = counter.count(text)

    # 微调：不足则逐句加，超出则截字符
    guard = 0
    while n < target_tokens and guard < 200:
        text += rng.choice(_CORPUS)
        n = counter.count(text)
        guard += 1
    while n > target_tokens and guard < 400:
        cut = max(1, int(len(text) * 0.02))
        text = text[:-cut]
        n = counter.count(text)
        guard += 1
    return text


class PromptFactory:
    """按场景生成 prompt。

    unique：每个请求一段唯一随机前缀，前缀不可复用
    shared：所有请求共用同一前缀，用于测 Prefix Cache 上限收益
    mixed：按 shared_prefix_ratio 混合，逼近真实会话分布
    """

    def __init__(self, counter: TokenCounter, seed: int = 0):
        self.counter = counter
        self.rng = random.Random(seed)
        self._shared_cache: dict[int, str] = {}

    def _unique_tag(self, idx: int) -> str:
        # 高熵前缀，确保 block 级哈希不命中
        salt = self.rng.getrandbits(48)
        return f"[会话{idx:06d}-{salt:012x}]"

    def _shared_body(self, target_tokens: int) -> str:
        if target_tokens not in self._shared_cache:
            r = random.Random(12345)  # 固定种子，跨请求完全一致
            self._shared_cache[target_tokens] = _grow_to(
                target_tokens, self.counter, "", r
            )
        return self._shared_cache[target_tokens]

    def build(self, idx: int, target_tokens: int, prefix_mode: str,
              shared_ratio: float) -> tuple[str, bool]:
        """返回 (prompt, expect_prefix_reuse)。

        expect_prefix_reuse 只表示构造意图，实际命中与否要看服务端
        prefix_cache_hits_total，两者不可互相替代。
        """
        use_shared = (
            prefix_mode == "shared"
            or (prefix_mode == "mixed" and self.rng.random() < shared_ratio)
        )

        if use_shared:
            body = self._shared_body(target_tokens)
            return body, True

        tag = self._unique_tag(idx)
        remain = max(1, target_tokens - self.counter.count(tag))
        body = _grow_to(remain, self.counter, tag, self.rng)
        return body, False


def make_messages(prompt: str) -> list[dict[str, str]]:
    """统一走 chat 接口。system 留空以免额外前缀污染 prefix cache 对比。"""
    return [{"role": "user", "content": prompt}]
