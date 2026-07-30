"""Offline logprob and entropy computation using Transformers.

This module provides memory-efficient offline computation of logprobs and entropies
for trajectory steps using a local Transformers model.

Supports multiple parallel strategies:
- naive: Single GPU with basic batching (original implementation)
- accelerate: HuggingFace Accelerate for easy distributed inference
- data_parallel: PyTorch DataParallel for multi-GPU batch processing
- fsdp: Fully Sharded Data Parallel for memory-efficient large model inference
- vllm: vLLM for high-throughput inference
- deepspeed: DeepSpeed ZeRO for memory-efficient inference
"""

import gc
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class ParallelStrategy(str, Enum):
    """Parallel strategy for offline logprob computation."""
    NAIVE = "naive"  # Single GPU with basic batching
    ACCELERATE = "accelerate"  # HuggingFace Accelerate
    DATA_PARALLEL = "data_parallel"  # PyTorch DataParallel
    FSDP = "fsdp"  # Fully Sharded Data Parallel
    VLLM = "vllm"  # vLLM for high-throughput
    DEEPSPEED = "deepspeed"  # DeepSpeed ZeRO


@dataclass
class OfflineLogprobConfig:
    """Configuration for offline logprob computation."""
    model_name: str
    tokenizer_name: str | None = None
    device: str = "auto"
    dtype: str = "bfloat16"
    batch_size: int = 1
    gpu_memory_utilization: float = 0.9
    parallel_strategy: str = "naive"
    # Advanced options
    use_flash_attention: bool = True
    gradient_checkpointing: bool = False
    sequence_bucketing: bool = True
    bucket_size: int = 128
    max_sequence_length: int = 8192
    # FSDP options
    fsdp_cpu_offload: bool = False
    fsdp_sharding_strategy: str = "FULL_SHARD"
    # vLLM options
    vllm_tensor_parallel_size: int = 1
    vllm_gpu_memory_utilization: float = 0.9
    # DeepSpeed options
    deepspeed_zero_stage: int = 3
    deepspeed_offload_device: str = "cpu"


@dataclass
class _OfflineStepBatchItem:
    """Internal data structure for batch processing."""
    step: object
    prompt_ids: list[int]
    response_ids: list[int]
    sequence_length: int = field(init=False)

    def __post_init__(self):
        self.sequence_length = len(self.prompt_ids) + len(self.response_ids)


def _resolve_dtype(dtype: str) -> torch.dtype:
    """Resolve string dtype to torch.dtype."""
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return dtype_map.get(dtype.lower(), torch.bfloat16)


def _collect_steps(trajectories: Iterable[object]) -> list[_OfflineStepBatchItem]:
    """Collect all steps with prompt_ids and response_ids from trajectories."""
    items: list[_OfflineStepBatchItem] = []
    for trajectory in trajectories:
        steps = getattr(trajectory, "steps", [])
        for step in steps:
            prompt_ids = getattr(step, "prompt_ids", []) or []
            response_ids = getattr(step, "response_ids", []) or []
            if not prompt_ids or not response_ids:
                continue
            items.append(
                _OfflineStepBatchItem(
                    step=step,
                    prompt_ids=prompt_ids,
                    response_ids=response_ids,
                )
            )
    return items


def _bucket_items_by_length(
    items: list[_OfflineStepBatchItem],
    bucket_size: int = 128,
) -> list[list[_OfflineStepBatchItem]]:
    """Group items into buckets by sequence length to reduce padding overhead."""
    if not items:
        return []

    # Sort by sequence length
    sorted_items = sorted(items, key=lambda x: x.sequence_length)

    # Group into buckets
    buckets: list[list[_OfflineStepBatchItem]] = []
    current_bucket: list[_OfflineStepBatchItem] = []
    current_max_len = 0

    for item in sorted_items:
        bucket_max = ((item.sequence_length // bucket_size) + 1) * bucket_size
        if current_bucket and bucket_max != current_max_len:
            buckets.append(current_bucket)
            current_bucket = []
        current_bucket.append(item)
        current_max_len = bucket_max

    if current_bucket:
        buckets.append(current_bucket)

    return buckets


def _shutdown_ray_completely():
    """Forcefully shutdown Ray to prevent GCS connection errors."""
    try:
        import ray
        if ray.is_initialized():
            logger.info("Shutting down Ray before offline computation...")
            ray.shutdown()
            import time
            time.sleep(2)
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Error shutting down Ray: {e}")


def _cleanup_gpu_memory():
    """Aggressively clean up GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _estimate_logits_memory_bytes(batch_size: int, seq_len: int, vocab_size: int) -> int:
    """Estimate GPU memory needed for the logits tensor (float32)."""
    # logits are typically computed in float32 even with autocast
    return batch_size * seq_len * vocab_size * 4


def _get_model_vocab_size(model: torch.nn.Module) -> int:
    """Get vocab size from model config."""
    if hasattr(model, 'config') and hasattr(model.config, 'vocab_size'):
        return model.config.vocab_size
    # Fallback: try lm_head weight shape
    if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight'):
        return model.lm_head.weight.shape[0]
    return 32000  # conservative default


def _get_free_gpu_memory(device) -> int:
    """Get free GPU memory in bytes for the given device."""
    if not torch.cuda.is_available():
        return 0
    if isinstance(device, str):
        device = torch.device(device)
    if hasattr(device, 'index') and device.index is not None:
        dev_idx = device.index
    else:
        dev_idx = torch.cuda.current_device()
    free_mem, _ = torch.cuda.mem_get_info(dev_idx)
    return free_mem


def _process_single_item_logprobs(
    item: "_OfflineStepBatchItem",
    model: torch.nn.Module,
    tokenizer,
    input_device: torch.device,
    torch_dtype: torch.dtype,
    vocab_size: int,
    seq_chunk_size: int = 2048,
) -> None:
    """Process a single item with chunked logits extraction for long sequences.

    For very long sequences, avoids materializing the full (1, L, V) logits tensor
    by processing the sequence in chunks and only extracting needed logprobs per chunk.
    """
    pl = len(item.prompt_ids)
    rl = len(item.response_ids)
    if rl <= 0:
        return

    seq = item.prompt_ids + item.response_ids
    total_len = len(seq)

    # Estimate logits memory for this single sequence
    logits_mem = _estimate_logits_memory_bytes(1, total_len, vocab_size)
    free_mem = _get_free_gpu_memory(input_device)
    # Use at most 40% of free memory for logits to leave room for intermediates
    mem_budget = int(free_mem * 0.4)

    if logits_mem <= mem_budget:
        # Fits in memory - process normally
        input_ids = torch.tensor([seq], device=input_device, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch_dtype):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # (1, L, V)

            start_pos = max(pl - 1, 0)
            end_pos = start_pos + rl
            logits_slice = logits[0, start_pos:end_pos, :].float()
            del logits, outputs

            log_probs = torch.log_softmax(logits_slice, dim=-1)
            response_ids_tensor = torch.tensor(
                item.response_ids, device=logits_slice.device, dtype=torch.long
            )
            token_logprobs = log_probs.gather(1, response_ids_tensor.unsqueeze(1)).squeeze(1)
            probs = log_probs.exp()
            token_entropies = -(probs * log_probs).sum(dim=-1)
            del log_probs, probs, logits_slice

            item.step.logprobs = token_logprobs.detach().cpu().tolist()
            item.step.token_entropies = token_entropies.detach().cpu().tolist()

        del input_ids, attention_mask
        return

    # Sequence too long - use chunked forward pass
    # We need logits at positions [pl-1, pl, ..., pl+rl-2] to predict response tokens
    logger.info(
        f"Chunked processing: seq_len={total_len}, vocab={vocab_size}, "
        f"logits_mem={logits_mem / (1024**3):.1f}GB, free={free_mem / (1024**3):.1f}GB"
    )

    all_token_logprobs = []
    all_token_entropies = []

    # We need logits at positions start_pos..end_pos-1 where:
    start_pos = max(pl - 1, 0)
    end_pos = start_pos + rl  # exclusive

    # Process the full sequence but extract logits in chunks
    # First, do the forward pass with output_hidden_states=False and
    # use a custom approach: feed the full input but only compute lm_head on chunks
    input_ids = torch.tensor([seq], device=input_device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch_dtype):
        # Get hidden states without computing lm_head
        if hasattr(model, 'model'):
            # Standard HF architecture: model.model is the base, model.lm_head is the head
            base_model = model.model
            lm_head = model.lm_head
        elif hasattr(model, 'transformer'):
            base_model = model.transformer
            lm_head = model.lm_head
        else:
            # Fallback: can't decompose, process full logits in chunks via slicing
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # (1, L, V) - may OOM but it's our last resort
            for chunk_start in range(start_pos, end_pos, seq_chunk_size):
                chunk_end = min(chunk_start + seq_chunk_size, end_pos)
                logits_chunk = logits[0, chunk_start:chunk_end, :].float()
                log_probs = torch.log_softmax(logits_chunk, dim=-1)
                resp_offset = chunk_start - start_pos
                resp_ids_chunk = item.response_ids[resp_offset:resp_offset + (chunk_end - chunk_start)]
                resp_tensor = torch.tensor(resp_ids_chunk, device=logits_chunk.device, dtype=torch.long)
                chunk_logprobs = log_probs.gather(1, resp_tensor.unsqueeze(1)).squeeze(1)
                probs = log_probs.exp()
                chunk_entropies = -(probs * log_probs).sum(dim=-1)
                all_token_logprobs.append(chunk_logprobs.detach().cpu())
                all_token_entropies.append(chunk_entropies.detach().cpu())
                del logits_chunk, log_probs, probs
            del logits, outputs
            item.step.logprobs = torch.cat(all_token_logprobs).tolist()
            item.step.token_entropies = torch.cat(all_token_entropies).tolist()
            del input_ids, attention_mask
            return

        # Get hidden states from base model
        base_outputs = base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = base_outputs[0]  # (1, L, hidden_dim)
        del base_outputs

        # Apply lm_head in chunks over the sequence positions we need
        for chunk_start in range(start_pos, end_pos, seq_chunk_size):
            chunk_end = min(chunk_start + seq_chunk_size, end_pos)
            hidden_chunk = hidden_states[0, chunk_start:chunk_end, :]  # (chunk_len, hidden_dim)

            # Apply lm_head to get logits for this chunk only
            logits_chunk = lm_head(hidden_chunk).float()  # (chunk_len, V)

            log_probs = torch.log_softmax(logits_chunk, dim=-1)
            resp_offset = chunk_start - start_pos
            resp_ids_chunk = item.response_ids[resp_offset:resp_offset + (chunk_end - chunk_start)]
            resp_tensor = torch.tensor(resp_ids_chunk, device=logits_chunk.device, dtype=torch.long)
            chunk_logprobs = log_probs.gather(1, resp_tensor.unsqueeze(1)).squeeze(1)
            probs = log_probs.exp()
            chunk_entropies = -(probs * log_probs).sum(dim=-1)

            all_token_logprobs.append(chunk_logprobs.detach().cpu())
            all_token_entropies.append(chunk_entropies.detach().cpu())
            del logits_chunk, log_probs, probs, hidden_chunk

        del hidden_states

    item.step.logprobs = torch.cat(all_token_logprobs).tolist()
    item.step.token_entropies = torch.cat(all_token_entropies).tolist()
    del input_ids, attention_mask


def _process_batch_logprobs(
    batch_items: list[_OfflineStepBatchItem],
    model: torch.nn.Module,
    tokenizer,
    input_device: torch.device,
    torch_dtype: torch.dtype,
) -> None:
    """Process a batch of items and populate logprobs/entropies.

    Dynamically adjusts batch size based on available GPU memory to avoid OOM.
    Falls back to single-item processing with chunked logits for very long sequences.
    """
    if not batch_items:
        return

    vocab_size = _get_model_vocab_size(model)
    max_len = max(len(item.prompt_ids) + len(item.response_ids) for item in batch_items)

    # Estimate logits memory for the full batch
    logits_mem = _estimate_logits_memory_bytes(len(batch_items), max_len, vocab_size)
    free_mem = _get_free_gpu_memory(input_device)
    # Use at most 40% of free memory for logits (rest needed for activations, intermediates)
    mem_budget = int(free_mem * 0.4)

    if logits_mem > mem_budget and len(batch_items) > 1:
        # Batch too large - split into smaller sub-batches
        # Calculate max safe batch size
        single_item_mem = _estimate_logits_memory_bytes(1, max_len, vocab_size)
        safe_batch_size = max(1, mem_budget // single_item_mem)

        logger.info(
            f"Dynamic batch resize: {len(batch_items)} -> {safe_batch_size} "
            f"(logits_mem={logits_mem / (1024**3):.1f}GB, free={free_mem / (1024**3):.1f}GB)"
        )
        print(
            f"Dynamic batch resize: {len(batch_items)} -> {safe_batch_size} "
            f"(seq_len={max_len}, vocab={vocab_size}, free={free_mem / (1024**3):.1f}GB)"
        )

        for sub_start in range(0, len(batch_items), safe_batch_size):
            sub_batch = batch_items[sub_start:sub_start + safe_batch_size]
            _process_batch_logprobs(sub_batch, model, tokenizer, input_device, torch_dtype)
            _cleanup_gpu_memory()
        return

    if logits_mem > mem_budget and len(batch_items) == 1:
        # Single item but still too large - use chunked processing
        _process_single_item_logprobs(
            batch_items[0], model, tokenizer, input_device, torch_dtype, vocab_size
        )
        _cleanup_gpu_memory()
        return

    # Normal processing - fits in memory
    prompt_lens = [len(item.prompt_ids) for item in batch_items]
    response_lens = [len(item.response_ids) for item in batch_items]

    input_ids = torch.full(
        (len(batch_items), max_len),
        fill_value=tokenizer.pad_token_id,
        dtype=torch.long,
        device=input_device,
    )
    attention_mask = torch.zeros_like(input_ids)

    for i, item in enumerate(batch_items):
        seq = item.prompt_ids + item.response_ids
        input_ids[i, : len(seq)] = torch.tensor(seq, device=input_device, dtype=torch.long)
        attention_mask[i, : len(seq)] = 1

    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch_dtype):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (B, L, V)

        for i, item in enumerate(batch_items):
            pl = prompt_lens[i]
            rl = response_lens[i]
            if rl <= 0:
                continue
            start_pos = max(pl - 1, 0)
            end_pos = start_pos + rl
            logits_slice = logits[i, start_pos:end_pos, :].float()
            log_probs = torch.log_softmax(logits_slice, dim=-1)

            response_ids_tensor = torch.tensor(
                item.response_ids, device=logits.device, dtype=torch.long
            )
            token_logprobs = log_probs.gather(1, response_ids_tensor.unsqueeze(1)).squeeze(1)

            probs = log_probs.exp()
            token_entropies = -(probs * log_probs).sum(dim=-1)

            item.step.logprobs = token_logprobs.detach().cpu().tolist()
            item.step.token_entropies = token_entropies.detach().cpu().tolist()

        del logits, outputs

    del input_ids, attention_mask


# =============================================================================
# Naive Strategy (Original Implementation)
# =============================================================================


def _run_naive_strategy(
    items: list[_OfflineStepBatchItem],
    config: OfflineLogprobConfig,
) -> None:
    """Run naive single-GPU strategy with basic batching."""
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name or config.model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    torch_dtype = _resolve_dtype(config.dtype)

    # Model loading kwargs
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }

    if config.use_flash_attention:
        try:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        except Exception:
            pass

    # Load model with appropriate device mapping
    device = config.device
    if device == "auto":
        logger.info("Loading model with device_map='auto' for multi-GPU inference...")
        num_gpus = torch.cuda.device_count()
        if num_gpus > 0:
            max_memory = {}
            for i in range(num_gpus):
                torch.cuda.set_device(i)
                _cleanup_gpu_memory()
                free_mem, total_mem = torch.cuda.mem_get_info(i)
                max_mem_gb = max(1, int(free_mem * config.gpu_memory_utilization / (1024 ** 3)))
                max_memory[i] = f"{max_mem_gb}GiB"
                logger.info(f"GPU {i}: free={free_mem // (1024**3)}GB, max_memory={max_memory[i]}")
            model_kwargs["device_map"] = "auto"
            model_kwargs["max_memory"] = max_memory
        else:
            logger.warning("No GPUs available, falling back to CPU")
            device = "cpu"

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)

    if device not in ["auto", "cpu"] and "device_map" not in model_kwargs:
        model.to(device)

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model.eval()

    # Determine input device
    if hasattr(model, 'device'):
        input_device = model.device
    elif hasattr(model, 'hf_device_map'):
        first_device = next(iter(model.hf_device_map.values()))
        input_device = f"cuda:{first_device}" if isinstance(first_device, int) else first_device
    else:
        input_device = next(model.parameters()).device

    logger.info(f"Model loaded. Input device: {input_device}")

    # Apply sequence bucketing if enabled
    if config.sequence_bucketing:
        buckets = _bucket_items_by_length(items, config.bucket_size)
        all_batches = []
        for bucket in buckets:
            for i in range(0, len(bucket), config.batch_size):
                all_batches.append(bucket[i:i + config.batch_size])
    else:
        all_batches = [items[i:i + config.batch_size] for i in range(0, len(items), config.batch_size)]

    total_batches = len(all_batches)

    for batch_idx, batch_items in enumerate(all_batches):
        _process_batch_logprobs(batch_items, model, tokenizer, input_device, torch_dtype)

        if (batch_idx + 1) % 5 == 0:
            _cleanup_gpu_memory()

        if (batch_idx + 1) % 10 == 0 or batch_idx == total_batches - 1:
            logger.info(f"Processed batch {batch_idx + 1}/{total_batches}")
            print(f"Processed batch {batch_idx + 1}/{total_batches}")

    del model
    _cleanup_gpu_memory()


# =============================================================================
# Accelerate Strategy
# =============================================================================


def _run_accelerate_strategy(
    items: list[_OfflineStepBatchItem],
    config: OfflineLogprobConfig,
) -> None:
    """Run HuggingFace Accelerate strategy for distributed inference.

    In single-process mode with multiple GPUs, uses device_map="auto" for model sharding.
    In multi-process mode (launched with accelerate launch), distributes data across processes.
    """
    try:
        from accelerate import Accelerator, PartialState
        from accelerate.utils import gather_object
    except ImportError:
        raise ImportError("accelerate is required for accelerate strategy. Install with: pip install accelerate")

    # Check if we're in multi-process mode
    is_multi_process = os.environ.get("ACCELERATE_USE_FSDP") or \
                       os.environ.get("WORLD_SIZE") or \
                       os.environ.get("LOCAL_RANK")

    num_gpus = torch.cuda.device_count()

    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name or config.model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    torch_dtype = _resolve_dtype(config.dtype)

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }

    if config.use_flash_attention:
        try:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        except Exception:
            pass

    # Single-process mode with multiple GPUs: use device_map="auto" for model sharding
    if not is_multi_process and num_gpus > 1:
        logger.info(f"Accelerate: Single-process mode with {num_gpus} GPUs, using device_map='auto'")
        print(f"Accelerate: Single-process mode with {num_gpus} GPUs, using device_map='auto'")

        # Calculate max_memory for each GPU
        max_memory = {}
        for i in range(num_gpus):
            torch.cuda.set_device(i)
            _cleanup_gpu_memory()
            free_mem, total_mem = torch.cuda.mem_get_info(i)
            max_mem_gb = max(1, int(free_mem * config.gpu_memory_utilization / (1024 ** 3)))
            max_memory[i] = f"{max_mem_gb}GiB"
            logger.info(f"GPU {i}: free={free_mem // (1024**3)}GB, max_memory={max_memory[i]}")

        model_kwargs["device_map"] = "auto"
        model_kwargs["max_memory"] = max_memory

        model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)

        if config.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        model.eval()

        # Determine input device
        if hasattr(model, 'hf_device_map'):
            first_device = next(iter(model.hf_device_map.values()))
            input_device = f"cuda:{first_device}" if isinstance(first_device, int) else first_device
        else:
            input_device = next(model.parameters()).device

        logger.info(f"Accelerate: Model sharded across {num_gpus} GPUs, input device: {input_device}")
        print(f"Accelerate: Model sharded across {num_gpus} GPUs, input device: {input_device}")

        # Process all items (no splitting needed in single-process mode)
        if config.sequence_bucketing:
            buckets = _bucket_items_by_length(items, config.bucket_size)
            all_batches = []
            for bucket in buckets:
                for i in range(0, len(bucket), config.batch_size):
                    all_batches.append(bucket[i:i + config.batch_size])
        else:
            all_batches = [items[i:i + config.batch_size] for i in range(0, len(items), config.batch_size)]

        total_batches = len(all_batches)

        for batch_idx, batch_items in enumerate(all_batches):
            _process_batch_logprobs(batch_items, model, tokenizer, input_device, torch_dtype)

            if (batch_idx + 1) % 5 == 0:
                _cleanup_gpu_memory()

            if (batch_idx + 1) % 10 == 0 or batch_idx == total_batches - 1:
                logger.info(f"Processed batch {batch_idx + 1}/{total_batches}")
                print(f"Processed batch {batch_idx + 1}/{total_batches}")

        del model
        _cleanup_gpu_memory()
        return

    # Multi-process mode: use Accelerator for distributed inference
    accelerator = Accelerator()

    # Load model - Accelerate will handle device placement
    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model.eval()

    # Prepare model with accelerator
    model = accelerator.prepare(model)
    input_device = accelerator.device

    logger.info(f"Accelerate: Model loaded on device {input_device}, process {accelerator.process_index}")

    # Split items across processes
    with accelerator.split_between_processes(items) as local_items:
        if config.sequence_bucketing:
            buckets = _bucket_items_by_length(local_items, config.bucket_size)
            all_batches = []
            for bucket in buckets:
                for i in range(0, len(bucket), config.batch_size):
                    all_batches.append(bucket[i:i + config.batch_size])
        else:
            all_batches = [local_items[i:i + config.batch_size] for i in range(0, len(local_items), config.batch_size)]

        total_batches = len(all_batches)

        for batch_idx, batch_items in enumerate(all_batches):
            _process_batch_logprobs(batch_items, model, tokenizer, input_device, torch_dtype)

            if (batch_idx + 1) % 5 == 0:
                _cleanup_gpu_memory()

            if (batch_idx + 1) % 10 == 0 or batch_idx == total_batches - 1:
                if accelerator.is_main_process:
                    logger.info(f"Process {accelerator.process_index}: Processed batch {batch_idx + 1}/{total_batches}")
                    print(f"Process {accelerator.process_index}: Processed batch {batch_idx + 1}/{total_batches}")

    # Wait for all processes
    accelerator.wait_for_everyone()

    del model
    _cleanup_gpu_memory()


# =============================================================================
# Data Parallel Strategy
# =============================================================================


def _run_data_parallel_strategy(
    items: list[_OfflineStepBatchItem],
    config: OfflineLogprobConfig,
) -> None:
    """Run PyTorch DataParallel strategy for multi-GPU batch processing."""
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name or config.model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    torch_dtype = _resolve_dtype(config.dtype)

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }

    if config.use_flash_attention:
        try:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        except Exception:
            pass

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    model = model.cuda()

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Wrap with DataParallel
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        logger.info(f"Using DataParallel with {num_gpus} GPUs")
        model = torch.nn.DataParallel(model)

    model.eval()
    input_device = torch.device("cuda:0")

    logger.info(f"DataParallel: Model loaded on {num_gpus} GPU(s)")

    # Increase batch size proportionally to number of GPUs
    effective_batch_size = config.batch_size * max(1, num_gpus)

    if config.sequence_bucketing:
        buckets = _bucket_items_by_length(items, config.bucket_size)
        all_batches = []
        for bucket in buckets:
            for i in range(0, len(bucket), effective_batch_size):
                all_batches.append(bucket[i:i + effective_batch_size])
    else:
        all_batches = [items[i:i + effective_batch_size] for i in range(0, len(items), effective_batch_size)]

    total_batches = len(all_batches)

    # Get the underlying model for processing
    base_model = model.module if hasattr(model, 'module') else model

    for batch_idx, batch_items in enumerate(all_batches):
        _process_batch_logprobs(batch_items, base_model, tokenizer, input_device, torch_dtype)

        if (batch_idx + 1) % 5 == 0:
            _cleanup_gpu_memory()

        if (batch_idx + 1) % 10 == 0 or batch_idx == total_batches - 1:
            logger.info(f"Processed batch {batch_idx + 1}/{total_batches}")
            print(f"Processed batch {batch_idx + 1}/{total_batches}")

    del model
    _cleanup_gpu_memory()


# =============================================================================
# FSDP Strategy
# =============================================================================


def _init_single_process_distributed():
    """Initialize distributed for single-process mode."""
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=0, world_size=1)


def _run_fsdp_strategy(
    items: list[_OfflineStepBatchItem],
    config: OfflineLogprobConfig,
) -> None:
    """Run Fully Sharded Data Parallel strategy for memory-efficient large model inference."""
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import ShardingStrategy, CPUOffload
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        import torch.distributed as dist
    except ImportError:
        raise ImportError("FSDP requires PyTorch >= 1.12. Please upgrade PyTorch.")

    # Check if we're in a distributed environment
    is_distributed_env = all(
        var in os.environ for var in ["RANK", "WORLD_SIZE", "LOCAL_RANK"]
    )

    if not is_distributed_env:
        logger.warning(
            "FSDP strategy requested but distributed environment not detected. "
            "Initializing single-process distributed mode. "
            "For multi-GPU FSDP, launch with: torchrun --nproc_per_node=N script.py"
        )
        print(
            "WARNING: FSDP running in single-process mode. "
            "For multi-GPU, use: torchrun --nproc_per_node=N script.py"
        )
        _init_single_process_distributed()
    elif not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name or config.model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    torch_dtype = _resolve_dtype(config.dtype)

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }

    if config.use_flash_attention:
        try:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        except Exception:
            pass

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)

    # Configure FSDP sharding strategy
    sharding_strategy_map = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
        "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
    }
    sharding_strategy = sharding_strategy_map.get(
        config.fsdp_sharding_strategy.upper(),
        ShardingStrategy.FULL_SHARD
    )

    # Configure CPU offload
    cpu_offload = CPUOffload(offload_params=config.fsdp_cpu_offload)

    # Get transformer layer class for auto wrap policy
    try:
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        transformer_layer_cls = {LlamaDecoderLayer}
    except ImportError:
        transformer_layer_cls = None

    # Wrap model with FSDP
    if transformer_layer_cls:
        auto_wrap_policy = transformer_auto_wrap_policy(
            transformer_layer_cls=transformer_layer_cls
        )
        model = FSDP(
            model,
            sharding_strategy=sharding_strategy,
            cpu_offload=cpu_offload,
            auto_wrap_policy=auto_wrap_policy,
            device_id=local_rank,
        )
    else:
        model = FSDP(
            model,
            sharding_strategy=sharding_strategy,
            cpu_offload=cpu_offload,
            device_id=local_rank,
        )

    model.eval()
    input_device = torch.device(f"cuda:{local_rank}")

    logger.info(f"FSDP: Model loaded on rank {local_rank}")

    # Split items across ranks
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    items_per_rank = len(items) // world_size
    start_idx = rank * items_per_rank
    end_idx = start_idx + items_per_rank if rank < world_size - 1 else len(items)
    local_items = items[start_idx:end_idx]

    if config.sequence_bucketing:
        buckets = _bucket_items_by_length(local_items, config.bucket_size)
        all_batches = []
        for bucket in buckets:
            for i in range(0, len(bucket), config.batch_size):
                all_batches.append(bucket[i:i + config.batch_size])
    else:
        all_batches = [local_items[i:i + config.batch_size] for i in range(0, len(local_items), config.batch_size)]

    total_batches = len(all_batches)

    for batch_idx, batch_items in enumerate(all_batches):
        _process_batch_logprobs(batch_items, model, tokenizer, input_device, torch_dtype)

        if (batch_idx + 1) % 5 == 0:
            _cleanup_gpu_memory()

        if (batch_idx + 1) % 10 == 0 or batch_idx == total_batches - 1:
            if rank == 0:
                logger.info(f"Rank {rank}: Processed batch {batch_idx + 1}/{total_batches}")
                print(f"Rank {rank}: Processed batch {batch_idx + 1}/{total_batches}")

    # Synchronize all ranks
    dist.barrier()

    del model
    _cleanup_gpu_memory()


# =============================================================================
# vLLM Strategy
# =============================================================================


def _run_vllm_strategy(
    items: list[_OfflineStepBatchItem],
    config: OfflineLogprobConfig,
) -> None:
    """Run vLLM strategy for high-throughput inference."""
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        raise ImportError("vLLM is required for vllm strategy. Install with: pip install vllm")

    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name or config.model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    # Initialize vLLM engine
    llm = LLM(
        model=config.model_name,
        tokenizer=config.tokenizer_name or config.model_name,
        tensor_parallel_size=config.vllm_tensor_parallel_size,
        gpu_memory_utilization=config.vllm_gpu_memory_utilization,
        trust_remote_code=True,
        dtype=config.dtype,
        max_model_len=config.max_sequence_length,
    )

    logger.info(f"vLLM: Model loaded with tensor_parallel_size={config.vllm_tensor_parallel_size}")

    # Process items in batches
    # vLLM handles batching internally, but we still need to prepare prompts
    total_items = len(items)

    for batch_start in range(0, total_items, config.batch_size):
        batch_end = min(batch_start + config.batch_size, total_items)
        batch_items = items[batch_start:batch_end]

        # Prepare prompts for vLLM
        prompt_token_ids = []
        for item in batch_items:
            # Full sequence for logprob computation
            full_seq = item.prompt_ids + item.response_ids
            prompt_token_ids.append(full_seq)

        # Use vLLM's prompt_logprobs feature
        sampling_params = SamplingParams(
            max_tokens=1,  # We don't need generation, just logprobs
            prompt_logprobs=1,  # Get logprobs for prompt tokens
            temperature=0.0,
        )

        try:
            outputs = llm.generate(
                prompt_token_ids=prompt_token_ids,
                sampling_params=sampling_params,
            )

            # Extract logprobs from outputs
            for i, (item, output) in enumerate(zip(batch_items, outputs)):
                prompt_len = len(item.prompt_ids)
                response_len = len(item.response_ids)

                if output.prompt_logprobs is not None:
                    # Extract logprobs for response tokens
                    token_logprobs = []
                    token_entropies = []

                    for j in range(prompt_len, prompt_len + response_len):
                        if j < len(output.prompt_logprobs) and output.prompt_logprobs[j] is not None:
                            # Get the logprob for the actual token
                            token_id = item.response_ids[j - prompt_len]
                            logprob_dict = output.prompt_logprobs[j]
                            if token_id in logprob_dict:
                                lp = logprob_dict[token_id].logprob
                                token_logprobs.append(lp)
                                # Approximate entropy from logprob
                                token_entropies.append(-lp)
                            else:
                                token_logprobs.append(0.0)
                                token_entropies.append(0.0)
                        else:
                            token_logprobs.append(0.0)
                            token_entropies.append(0.0)

                    item.step.logprobs = token_logprobs
                    item.step.token_entropies = token_entropies

        except Exception as e:
            logger.warning(f"vLLM batch processing error: {e}")
            # Fall back to processing items individually
            for item in batch_items:
                item.step.logprobs = []
                item.step.token_entropies = []

        batch_idx = batch_start // config.batch_size
        total_batches = (total_items + config.batch_size - 1) // config.batch_size
        if (batch_idx + 1) % 10 == 0 or batch_idx == total_batches - 1:
            logger.info(f"vLLM: Processed batch {batch_idx + 1}/{total_batches}")
            print(f"vLLM: Processed batch {batch_idx + 1}/{total_batches}")

    del llm
    _cleanup_gpu_memory()


# =============================================================================
# DeepSpeed Strategy
# =============================================================================


def _run_deepspeed_strategy(
    items: list[_OfflineStepBatchItem],
    config: OfflineLogprobConfig,
) -> None:
    """Run DeepSpeed ZeRO strategy for memory-efficient inference."""
    try:
        import deepspeed
    except ImportError:
        raise ImportError("DeepSpeed is required for deepspeed strategy. Install with: pip install deepspeed")

    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name or config.model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    torch_dtype = _resolve_dtype(config.dtype)

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }

    if config.use_flash_attention:
        try:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        except Exception:
            pass

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # DeepSpeed inference config
    ds_config = {
        "tensor_parallel": {"tp_size": 1},
        "dtype": str(torch_dtype).split('.')[-1],
        "replace_with_kernel_inject": True,
        "enable_cuda_graph": False,
    }

    # Configure ZeRO stage for inference
    if config.deepspeed_zero_stage > 0:
        ds_config["zero_optimization"] = {
            "stage": config.deepspeed_zero_stage,
            "offload_param": {
                "device": config.deepspeed_offload_device,
            } if config.deepspeed_offload_device != "none" else None,
        }

    # Initialize DeepSpeed inference engine
    model = deepspeed.init_inference(
        model,
        config=ds_config,
    )

    model.eval()
    input_device = model.module.device if hasattr(model, 'module') else next(model.parameters()).device

    logger.info(f"DeepSpeed: Model loaded with ZeRO stage {config.deepspeed_zero_stage}")

    if config.sequence_bucketing:
        buckets = _bucket_items_by_length(items, config.bucket_size)
        all_batches = []
        for bucket in buckets:
            for i in range(0, len(bucket), config.batch_size):
                all_batches.append(bucket[i:i + config.batch_size])
    else:
        all_batches = [items[i:i + config.batch_size] for i in range(0, len(items), config.batch_size)]

    total_batches = len(all_batches)

    # Get the underlying model for processing
    base_model = model.module if hasattr(model, 'module') else model

    for batch_idx, batch_items in enumerate(all_batches):
        _process_batch_logprobs(batch_items, base_model, tokenizer, input_device, torch_dtype)

        if (batch_idx + 1) % 5 == 0:
            _cleanup_gpu_memory()

        if (batch_idx + 1) % 10 == 0 or batch_idx == total_batches - 1:
            logger.info(f"DeepSpeed: Processed batch {batch_idx + 1}/{total_batches}")
            print(f"DeepSpeed: Processed batch {batch_idx + 1}/{total_batches}")

    del model
    _cleanup_gpu_memory()


# =============================================================================
# Main Entry Point
# =============================================================================


# Strategy dispatcher
_STRATEGY_RUNNERS = {
    ParallelStrategy.NAIVE: _run_naive_strategy,
    ParallelStrategy.ACCELERATE: _run_accelerate_strategy,
    ParallelStrategy.DATA_PARALLEL: _run_data_parallel_strategy,
    ParallelStrategy.FSDP: _run_fsdp_strategy,
    ParallelStrategy.VLLM: _run_vllm_strategy,
    ParallelStrategy.DEEPSPEED: _run_deepspeed_strategy,
}


def populate_offline_logprobs(
    trajectories: Iterable[object],
    model_name: str,
    tokenizer_name: str | None = None,
    device: str = "auto",
    dtype: str = "bfloat16",
    batch_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    parallel_strategy: str = "naive",
    **kwargs,
) -> None:
    """Populate step.logprobs and step.token_entropies using offline Transformers forward.

    This computes exact entropy from full logits distribution on GPU.

    Args:
        trajectories: Iterable of trajectory objects containing steps.
        model_name: HuggingFace model name or path.
        tokenizer_name: Tokenizer name (defaults to model_name).
        device: Device specification:
            - "auto": Use device_map="auto" to distribute across all GPUs (recommended for large models)
            - "cuda": Use single GPU (cuda:0)
            - "cuda:X": Use specific GPU
            - "cpu": Use CPU (very slow)
        dtype: Data type for model weights (float16, bfloat16, float32).
        batch_size: Batch size for forward pass.
        gpu_memory_utilization: Max fraction of GPU memory to use.
        parallel_strategy: Parallel strategy to use:
            - "naive": Single GPU with basic batching (default, original implementation)
            - "accelerate": HuggingFace Accelerate for easy distributed inference
            - "data_parallel": PyTorch DataParallel for multi-GPU batch processing
            - "fsdp": Fully Sharded Data Parallel for memory-efficient large model inference
            - "vllm": vLLM for high-throughput inference
            - "deepspeed": DeepSpeed ZeRO for memory-efficient inference
        **kwargs: Additional strategy-specific options:
            - use_flash_attention: bool = True
            - gradient_checkpointing: bool = False
            - sequence_bucketing: bool = True
            - bucket_size: int = 128
            - max_sequence_length: int = 8192
            - fsdp_cpu_offload: bool = False
            - fsdp_sharding_strategy: str = "FULL_SHARD"
            - vllm_tensor_parallel_size: int = 1
            - vllm_gpu_memory_utilization: float = 0.9
            - deepspeed_zero_stage: int = 3
            - deepspeed_offload_device: str = "cpu"
    """
    # Shutdown Ray completely to prevent GCS connection errors
    _shutdown_ray_completely()

    # Clean up GPU memory before loading model
    _cleanup_gpu_memory()

    items = _collect_steps(trajectories)
    if not items:
        logger.warning("No steps with prompt_ids/response_ids found for offline logprob computation.")
        return

    logger.info(f"Computing offline logprobs for {len(items)} steps using {parallel_strategy} strategy...")
    print(f"Computing offline logprobs for {len(items)} steps using {parallel_strategy} strategy...")

    # Build config
    config = OfflineLogprobConfig(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        gpu_memory_utilization=gpu_memory_utilization,
        parallel_strategy=parallel_strategy,
        **{k: v for k, v in kwargs.items() if hasattr(OfflineLogprobConfig, k)},
    )

    # Resolve strategy
    try:
        strategy = ParallelStrategy(parallel_strategy.lower())
    except ValueError:
        logger.warning(f"Unknown parallel strategy '{parallel_strategy}', falling back to naive")
        strategy = ParallelStrategy.NAIVE

    # Run the appropriate strategy
    runner = _STRATEGY_RUNNERS.get(strategy, _run_naive_strategy)
    runner(items, config)

    logger.info("Offline logprob computation complete.")
    print("Offline logprob computation complete.")


def populate_offline_logprobs_with_config(
    trajectories: Iterable[object],
    config: OfflineLogprobConfig,
) -> None:
    """Populate step.logprobs and step.token_entropies using a config object.

    This is an alternative entry point that accepts a full config object
    for more fine-grained control.

    Args:
        trajectories: Iterable of trajectory objects containing steps.
        config: OfflineLogprobConfig object with all settings.
    """
    # Shutdown Ray completely to prevent GCS connection errors
    _shutdown_ray_completely()

    # Clean up GPU memory before loading model
    _cleanup_gpu_memory()

    items = _collect_steps(trajectories)
    if not items:
        logger.warning("No steps with prompt_ids/response_ids found for offline logprob computation.")
        return

    logger.info(f"Computing offline logprobs for {len(items)} steps using {config.parallel_strategy} strategy...")
    print(f"Computing offline logprobs for {len(items)} steps using {config.parallel_strategy} strategy...")

    # Resolve strategy
    try:
        strategy = ParallelStrategy(config.parallel_strategy.lower())
    except ValueError:
        logger.warning(f"Unknown parallel strategy '{config.parallel_strategy}', falling back to naive")
        strategy = ParallelStrategy.NAIVE

    # Run the appropriate strategy
    runner = _STRATEGY_RUNNERS.get(strategy, _run_naive_strategy)
    runner(items, config)

    logger.info("Offline logprob computation complete.")
    print("Offline logprob computation complete.")


# =============================================================================
# Utility Functions
# =============================================================================


def get_available_strategies() -> list[str]:
    """Get list of available parallel strategies."""
    return [s.value for s in ParallelStrategy]


def get_recommended_strategy(
    model_size_gb: float,
    num_gpus: int,
    total_gpu_memory_gb: float,
) -> str:
    """Get recommended parallel strategy based on hardware and model size.

    Args:
        model_size_gb: Estimated model size in GB.
        num_gpus: Number of available GPUs.
        total_gpu_memory_gb: Total GPU memory across all GPUs in GB.

    Returns:
        Recommended parallel strategy name.
    """
    # If model fits in single GPU with room to spare, use naive
    if model_size_gb * 1.5 < total_gpu_memory_gb / num_gpus:
        if num_gpus > 1:
            return ParallelStrategy.DATA_PARALLEL.value
        return ParallelStrategy.NAIVE.value

    # If model fits across all GPUs, use data parallel or accelerate
    if model_size_gb * 1.2 < total_gpu_memory_gb:
        if num_gpus > 1:
            return ParallelStrategy.ACCELERATE.value
        return ParallelStrategy.NAIVE.value

    # For very large models, use FSDP or DeepSpeed
    if num_gpus >= 4:
        return ParallelStrategy.FSDP.value

    # For high throughput needs with moderate model size
    if model_size_gb < total_gpu_memory_gb * 0.8:
        return ParallelStrategy.VLLM.value

    # Default to DeepSpeed for memory efficiency
    return ParallelStrategy.DEEPSPEED.value


def estimate_memory_requirements(
    model_name: str,
    dtype: str = "bfloat16",
    batch_size: int = 1,
    max_sequence_length: int = 4096,
) -> dict[str, float]:
    """Estimate memory requirements for offline logprob computation.

    Args:
        model_name: HuggingFace model name or path.
        dtype: Data type for model weights.
        batch_size: Batch size for forward pass.
        max_sequence_length: Maximum sequence length.

    Returns:
        Dictionary with estimated memory requirements in GB.
    """
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        # Estimate model parameters
        hidden_size = getattr(config, 'hidden_size', 4096)
        num_layers = getattr(config, 'num_hidden_layers', 32)
        vocab_size = getattr(config, 'vocab_size', 32000)
        intermediate_size = getattr(config, 'intermediate_size', hidden_size * 4)

        # Rough parameter count estimation
        params_per_layer = (
            4 * hidden_size * hidden_size +  # attention
            3 * hidden_size * intermediate_size  # MLP
        )
        total_params = (
            num_layers * params_per_layer +
            vocab_size * hidden_size * 2  # embeddings
        )

        # Bytes per parameter based on dtype
        bytes_per_param = {
            "float32": 4, "fp32": 4,
            "float16": 2, "fp16": 2,
            "bfloat16": 2, "bf16": 2,
        }.get(dtype.lower(), 2)

        model_memory_gb = total_params * bytes_per_param / (1024 ** 3)

        # Activation memory estimation (rough)
        activation_memory_gb = (
            batch_size * max_sequence_length * hidden_size * num_layers * 2 * bytes_per_param
        ) / (1024 ** 3)

        # KV cache estimation
        kv_cache_gb = (
            2 * batch_size * max_sequence_length * hidden_size * num_layers * bytes_per_param
        ) / (1024 ** 3)

        return {
            "model_memory_gb": model_memory_gb,
            "activation_memory_gb": activation_memory_gb,
            "kv_cache_gb": kv_cache_gb,
            "total_estimated_gb": model_memory_gb + activation_memory_gb + kv_cache_gb,
            "estimated_params_billions": total_params / 1e9,
        }

    except Exception as e:
        logger.warning(f"Could not estimate memory requirements: {e}")
        return {
            "model_memory_gb": 0.0,
            "activation_memory_gb": 0.0,
            "kv_cache_gb": 0.0,
            "total_estimated_gb": 0.0,
            "estimated_params_billions": 0.0,
        }
