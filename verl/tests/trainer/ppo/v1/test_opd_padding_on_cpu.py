"""Regression test for distillation fields in a synthetic replay-buffer sample."""

import torch

from verl.trainer.ppo.padding_utils import construct_minimal_padding_template


def test_padding_template_resizes_teacher_fields_to_synthetic_sequence():
    source = {
        "prompts": torch.tensor([1, 2]),
        "responses": torch.tensor([3, 4, 5]),
        "position_ids": torch.arange(5),
        "teacher_logprobs": torch.randn(5, 1),
        "teacher_ids": torch.arange(5, dtype=torch.int32).unsqueeze(-1),
    }

    sample, tag = construct_minimal_padding_template(source, {"seq_len": 5}, eos_token_id=7)

    assert sample["input_ids"].tolist() == [7, 7]
    assert sample["teacher_logprobs"].shape == (2, 1)
    assert sample["teacher_ids"].shape == (2, 1)
    assert torch.count_nonzero(sample["teacher_logprobs"]) == 0
    assert torch.count_nonzero(sample["teacher_ids"]) == 0
    assert tag["seq_len"] == 2
    assert tag["is_padding"] is True
