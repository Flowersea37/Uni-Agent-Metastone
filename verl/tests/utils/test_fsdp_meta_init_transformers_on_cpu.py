"""Regression for Transformers parameter markers under FSDP meta initialization."""

import torch
from transformers import AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel

from verl.utils.fsdp_utils import get_init_weight_context_manager


def test_meta_init_accepts_transformers_initialized_parameter(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    init_context = get_init_weight_context_manager(use_meta_tensor=True)

    with init_context():
        module = torch.nn.Linear(2, 2)
        loaded = torch.nn.Parameter(torch.ones(2))
        loaded._is_hf_initialized = True
        module.loaded = loaded
        module.register_buffer("example_buffer", torch.ones(2))

    assert module.weight.device.type == "meta"
    assert module.loaded.device.type == "meta"
    assert module.loaded._is_hf_initialized is True
    assert module.example_buffer.device.type == "cpu"


def test_from_pretrained_loads_under_fsdp_meta_context(monkeypatch, tmp_path):
    config = GPT2Config(vocab_size=32, n_positions=16, n_ctx=16, n_embd=16, n_layer=1, n_head=2)
    GPT2LMHeadModel(config).save_pretrained(tmp_path, safe_serialization=True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)

    with get_init_weight_context_manager(use_meta_tensor=True)():
        model = AutoModelForCausalLM.from_pretrained(tmp_path, config=config)

    assert {param.device.type for param in model.parameters()} == {"meta"}
