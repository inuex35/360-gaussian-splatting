import pytest
import torch
from utils.image_utils import mse, psnr

def test_mse():
    img1 = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]])
    img2 = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]])
    expected_mse = torch.tensor([[[[1.0]]]])
    assert torch.allclose(mse(img1, img2), expected_mse)

def test_psnr():
    img1 = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]])
    img2 = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]])
    expected_psnr = 20 * torch.log10(torch.tensor(1.0) / torch.sqrt(torch.tensor(1.0)))
    assert torch.allclose(psnr(img1, img2), expected_psnr)

def test_mse_same_image():
    img1 = torch.tensor([[[[0.5, 0.5], [0.5, 0.5]]]])
    img2 = torch.tensor([[[[0.5, 0.5], [0.5, 0.5]]]])
    expected_mse = torch.tensor([[[[0.0]]]])
    assert torch.allclose(mse(img1, img2), expected_mse)

def test_psnr_same_image():
    img1 = torch.tensor([[[[0.5, 0.5], [0.5, 0.5]]]])
    img2 = torch.tensor([[[[0.5, 0.5], [0.5, 0.5]]]])
    expected_psnr = torch.tensor(float('inf'))
    assert torch.isinf(psnr(img1, img2))

if __name__ == "__main__":
    pytest.main()