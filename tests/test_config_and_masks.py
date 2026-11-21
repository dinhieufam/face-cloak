import random
from pathlib import Path

import pytest
import torch

from facecloak.config import get_paths, load_config
from facecloak.masks import attacker_view, jitter_mask

REPO = Path(__file__).resolve().parents[1]
EXPS = sorted((REPO / "configs" / "experiments").glob("*.yaml"))


@pytest.mark.parametrize("path", EXPS, ids=[p.stem for p in EXPS])
def test_every_experiment_config_loads(path):
    cfg = load_config(path)
    assert cfg["exp_name"] == path.stem
    assert cfg["cloak"]["eps"] == [16] and cfg["cloak"]["n_iters"] == 200   # shared protocol is inherited
    assert cfg["inpaint"]["prompt_ids"] == [0] and cfg["inpaint"]["seeds"] == [0, 1]


def test_experiments_change_one_knob_each():
    """E* = single-change experiments vs. base.yaml. C* = documented combinations of E* winners (exempt)."""
    base = load_config(REPO / "configs" / "base.yaml")["cloak"]
    for path in EXPS:
        if path.stem.startswith("C"):
            assert "combin" in path.read_text().split("\n", 1)[0].lower(), f"{path.stem}: say what it combines"
            continue
        assert path.stem.startswith("E"), path.stem
        diff = {k for k, v in load_config(path)["cloak"].items() if base[k] != v}
        assert len(diff) <= 2, (path.stem, diff)  # E2 sets mask_aware + its jitter


def test_overrides_and_quick_test(tmp_path, monkeypatch):
    monkeypatch.setenv("FACECLOAK_OUTPUT_ROOT", str(tmp_path))
    cfg = load_config(REPO / "configs/experiments/E1_pattern_target.yaml",
                      ["cloak.n_iters=7", "cloak.eps=[8,16]", "quick_test=true"])
    assert cfg["exp_name"] == "E1_pattern_target_quick"
    assert cfg["cloak"]["n_iters"] == 50 and cfg["cloak"]["eps"] == [16]      # quick test wins
    assert cfg["data"]["max_images"] == 2
    p = get_paths(cfg)
    assert p.cloak_dir == tmp_path / "E1_pattern_target_quick" / "cloak_out"


def test_colab_and_local_roots(monkeypatch):
    monkeypatch.delenv("FACECLOAK_OUTPUT_ROOT", raising=False)
    cfg = load_config(REPO / "configs/experiments/E0_baseline.yaml")
    for where in ("colab", "local"):  # Colab clones the repo, so outputs land in <repo>/runs there too
        monkeypatch.setenv("FACECLOAK_ENV", where)
        assert get_paths(cfg).root == REPO / "runs"


def test_attacker_view_is_mid_gray_outside_face():
    x = torch.rand(1, 3, 16, 16)
    m = torch.zeros(1, 1, 16, 16)
    m[..., 4:12, 4:12] = 1
    v = attacker_view(x, m)
    assert torch.all(v[..., :4, :] == 0.5)
    assert torch.equal(v[..., 4:12, 4:12], x[..., 4:12, 4:12])


def test_jitter_mask_dilates_or_erodes():
    m = torch.zeros(1, 1, 32, 32)
    m[..., 10:22, 10:22] = 1
    sizes = {jitter_mask(m, 3, random.Random(s)).sum().item() for s in range(30)}
    assert min(sizes) < m.sum().item() < max(sizes)
    assert jitter_mask(m, 0) is m
