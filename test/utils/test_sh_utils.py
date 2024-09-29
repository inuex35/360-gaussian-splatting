import pytest
import torch
from utils.sh_utils import eval_sh, RGB2SH, SH2RGB

def test_eval_sh_deg_0():
    sh = torch.tensor([[[1.0]]])
    dirs = torch.tensor([[0.0, 0.0, 1.0]])
    result = eval_sh(0, sh, dirs)
    expected = torch.tensor([0.28209479177387814])
    assert torch.allclose(result, expected), f"Expected {expected}, but got {result}"

def test_eval_sh_deg_1():
    sh = torch.tensor([[[1.0, 0.5, 0.5, 0.5]]])
    dirs = torch.tensor([[0.0, 0.0, 1.0]])
    result = eval_sh(1, sh, dirs)
    expected = torch.tensor([0.5263950477253381])
    assert torch.allclose(result, expected), f"Expected {expected}, but got {result}"

"""
def test_eval_sh_deg_2():
    sh = torch.tensor([[[1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]])
    dirs = torch.tensor([[0.0, 0.0, 1.0]])
    result = eval_sh(2, sh, dirs)
    expected = torch.tensor([[0.8418]])
    torch.testing.assert_allclose(result.squeeze(), expected.squeeze(), atol=1e-5)
"""

def test_RGB2SH():
    rgb = torch.tensor([0.5, 0.5, 0.5])
    result = RGB2SH(rgb)
    expected = torch.tensor([0.0, 0.0, 0.0])
    assert torch.allclose(result, expected), f"Expected {expected}, but got {result}"

def test_SH2RGB():
    sh = torch.tensor([0.0, 0.0, 0.0])
    result = SH2RGB(sh)
    expected = torch.tensor([0.5, 0.5, 0.5])
    assert torch.allclose(result, expected), f"Expected {expected}, but got {result}"

if __name__ == '__main__':
    pytest.main()