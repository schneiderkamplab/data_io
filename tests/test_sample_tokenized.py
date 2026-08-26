from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

import sample_tokenized


def _write_task(root: Path, name: str) -> None:
    task = root / name
    task.mkdir()
    inst_len = np.array([2, 2, 5, 5], dtype=np.int64)
    resp_len = np.array([2, 2, 2, 2], dtype=np.int64)
    inst_start = np.array([0, 4, 8, 15], dtype=np.int64)
    resp_start = inst_start + inst_len
    np.save(task / "inst_start.npy", inst_start)
    np.save(task / "inst_len.npy", inst_len)
    np.save(task / "resp_start.npy", resp_start)
    np.save(task / "resp_len.npy", resp_len)
    np.save(task / "tokens.npy", np.arange(22, dtype=np.uint8))


def test_skip_unmatched_and_no_repeat_bins_reset_each_epoch(tmp_path: Path, monkeypatch) -> None:
    tokenized = tmp_path / "tokenized"
    tokenized.mkdir()
    (tokenized / "tokenizer_info.json").write_text(json.dumps({"vocab_size": 100}))
    _write_task(tokenized, "included_task")
    _write_task(tokenized, "unmatched_task")

    prefix_config = tmp_path / "prefix.yaml"
    prefix_config.write_text(yaml.safe_dump([{"prefix": "included_", "repeat": 3}]))
    output = tmp_path / "sampled"
    bins = (
        "[{min: 0, max: 5, fraction: 1.0, no_repeat: true}, "
        "{min: 5, max: 10, fraction: 1.0, no_repeat: true}]"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sample_tokenized.py",
            f"tokenized_path={tokenized}",
            f"output_path={output}",
            f"prefix_config_path={prefix_config}",
            "epochs=2",
            "context_size=10",
            "concat_workers=1",
            "skip_unmatched=true",
            f"length_bins={bins}",
        ],
    )

    sample_tokenized.main()

    for epoch in range(2):
        starts = np.load(output / f"epoch_{epoch}" / "inst_start.npy")
        assert len(starts) == 4
        assert sorted(starts.tolist()) == [0, 4, 8, 15]
    assert np.load(output / "tokens.npy").shape == (22,)
