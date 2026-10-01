"""Data-parallel vLLM generation: one model replica per GPU group, sharded prompts."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any


def visible_gpu_ids() -> list[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if raw.strip():
        return [g.strip() for g in raw.split(",") if g.strip()]
    import torch

    return [str(i) for i in range(torch.cuda.device_count())]


def count_visible_gpus() -> int:
    return len(visible_gpu_ids())


def resolve_data_parallel_replicas(
    n_gpus: int,
    tensor_parallel_size: int,
    requested: int,
) -> int:
    """0 = single engine; -1 = auto (use every GPU group); else explicit replica count."""
    if requested == 0:
        return 1
    if requested < 0:
        if n_gpus < tensor_parallel_size:
            return 1
        return n_gpus // tensor_parallel_size
    if requested * tensor_parallel_size > n_gpus:
        raise ValueError(
            f"data_parallel_replicas={requested} x tensor_parallel_size={tensor_parallel_size} "
            f"requires {requested * tensor_parallel_size} GPUs but only {n_gpus} visible"
        )
    return requested


def shard_indices(n_items: int, num_shards: int) -> list[list[int]]:
    shards: list[list[int]] = [[] for _ in range(num_shards)]
    for idx in range(n_items):
        shards[idx % num_shards].append(idx)
    return shards


def _generate_on_loaded_llm(
    llm: Any,
    prompt_token_ids: list[list[int]],
    samples_per_prompt: list[int],
    base_sampling: dict[str, Any],
) -> list[Any]:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    if len(samples_per_prompt) != len(prompt_token_ids):
        raise ValueError("samples_per_prompt length must match prompt_token_ids")

    token_prompts = [TokensPrompt(prompt_token_ids=ids) for ids in prompt_token_ids]
    unique_ns = sorted(set(samples_per_prompt))
    if len(unique_ns) == 1:
        sampling = SamplingParams(n=unique_ns[0], **base_sampling)
        return llm.generate(token_prompts, sampling)

    outputs: list[Any | None] = [None] * len(prompt_token_ids)
    for n in unique_ns:
        idxs = [i for i, count in enumerate(samples_per_prompt) if count == n]
        sampling = SamplingParams(n=n, **base_sampling)
        batch = llm.generate([token_prompts[i] for i in idxs], sampling)
        for i, out in zip(idxs, batch):
            outputs[i] = out
    if any(x is None for x in outputs):
        raise RuntimeError("Missing outputs after grouped generation")
    return outputs  # type: ignore[return-value]


def _vllm_replica_worker(payload: dict[str, Any]) -> list[tuple[int, Any]]:
    """Spawn-safe worker: load vLLM on a GPU group and generate one prompt shard."""
    import time

    os.environ["CUDA_VISIBLE_DEVICES"] = payload["cuda_visible_devices"]
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

    from vllm import LLM

    base_sampling = dict(payload["base_sampling"])
    try:
        stagger = int(str(payload["cuda_visible_devices"]).split(",")[0].strip()) * 2
    except ValueError:
        stagger = 0
    if stagger:
        time.sleep(stagger)

    llm = None
    last_err: Exception | None = None
    for attempt in range(1, 6):
        try:
            llm = LLM(
                model=payload["checkpoint"],
                skip_tokenizer_init=True,
                tensor_parallel_size=payload["tensor_parallel_size"],
                gpu_memory_utilization=payload["gpu_memory_utilization"],
                enforce_eager=True,
                compilation_config=0,
                max_model_len=payload["max_model_len"],
                seed=payload["seed"],
            )
            break
        except Exception as e:
            last_err = e
            print(
                f"replica GPU {payload['cuda_visible_devices']} LLM init "
                f"attempt {attempt}/5 failed: {e}",
                flush=True,
            )
            time.sleep(10 * attempt)
    if llm is None:
        raise RuntimeError(
            f"replica GPU {payload['cuda_visible_devices']} failed LLM init after 5 attempts"
        ) from last_err
    outputs = _generate_on_loaded_llm(
        llm,
        payload["prompt_token_ids"],
        payload["samples_per_prompt"],
        base_sampling,
    )
    return [(payload["prompt_indices"][i], out) for i, out in enumerate(outputs)]


def generate_with_data_parallel_replicas(
    *,
    checkpoint: str,
    prompt_token_ids: list[list[int]],
    tensor_parallel_size: int,
    num_replicas: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    seed: int,
    sampling_kwargs: dict[str, Any],
    samples_per_prompt: list[int] | None = None,
) -> list[Any]:
    """
    Run one or more vLLM replicas over disjoint prompt shards.

    Returns vLLM RequestOutput objects in the same order as prompt_token_ids.
    """
    if samples_per_prompt is None:
        if "n" not in sampling_kwargs:
            raise ValueError("sampling_kwargs must include 'n' or pass samples_per_prompt")
        samples_per_prompt = [int(sampling_kwargs["n"])] * len(prompt_token_ids)
    base_sampling = {k: v for k, v in sampling_kwargs.items() if k != "n"}

    if num_replicas <= 1:
        from vllm import LLM

        llm = LLM(
            model=checkpoint,
            skip_tokenizer_init=True,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True,
            compilation_config=0,
            max_model_len=max_model_len,
            seed=seed,
        )
        return _generate_on_loaded_llm(llm, prompt_token_ids, samples_per_prompt, base_sampling)

    gpu_ids = visible_gpu_ids()
    shards = shard_indices(len(prompt_token_ids), num_replicas)
    payloads = []
    for replica_idx, prompt_indices in enumerate(shards):
        if not prompt_indices:
            continue
        start = replica_idx * tensor_parallel_size
        replica_gpus = gpu_ids[start : start + tensor_parallel_size]
        payloads.append(
            {
                "cuda_visible_devices": ",".join(replica_gpus),
                "checkpoint": checkpoint,
                "tensor_parallel_size": tensor_parallel_size,
                "gpu_memory_utilization": gpu_memory_utilization,
                "max_model_len": max_model_len,
                "seed": seed + replica_idx,
                "base_sampling": base_sampling,
                "prompt_indices": prompt_indices,
                "prompt_token_ids": [prompt_token_ids[i] for i in prompt_indices],
                "samples_per_prompt": [samples_per_prompt[i] for i in prompt_indices],
            }
        )

    print(
        f"Data-parallel vLLM: {len(payloads)} replicas x tp={tensor_parallel_size} "
        f"over {len(prompt_token_ids)} prompts"
    )
    ordered: list[Any | None] = [None] * len(prompt_token_ids)
    with ProcessPoolExecutor(max_workers=len(payloads)) as pool:
        futures = [pool.submit(_vllm_replica_worker, p) for p in payloads]
        for fut in as_completed(futures):
            for prompt_idx, out in fut.result():
                ordered[prompt_idx] = out

    if any(x is None for x in ordered):
        raise RuntimeError("Data-parallel generation missing outputs for some prompts")
    return ordered  # type: ignore[return-value]
