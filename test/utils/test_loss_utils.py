import pytest
import torch
from utils.loss_utils import latitude_weight, l1_loss, l2_loss, ssim

def test_latitude_weight():
    height = 10
    weight = latitude_weight(height)
    assert weight.shape == (3, height, 1)
    assert torch.allclose(weight[0], weight[1]) and torch.allclose(weight[1], weight[2])

def test_l1_loss():
    network_output = torch.tensor([1.0, 2.0, 3.0])
    gt = torch.tensor([1.0, 2.0, 3.0])
    assert l1_loss(network_output, gt).item() == 0.0

    gt = torch.tensor([0.0, 0.0, 0.0])
    assert l1_loss(network_output, gt).item() == 2.0

def test_l2_loss():
    network_output = torch.tensor([1.0, 2.0, 3.0])
    gt = torch.tensor([1.0, 2.0, 3.0])
    assert l2_loss(network_output, gt).item() == 0.0

    gt = torch.tensor([0.0, 0.0, 0.0])
    assert l2_loss(network_output, gt).item() == 4.666666507720947

def test_ssim():
    img1 = torch.ones((1, 1, 11, 11))
    img2 = torch.ones((1, 1, 11, 11))
    assert ssim(img1, img2).item() == 1.0

    img2 = torch.zeros((1, 1, 11, 11))
    assert ssim(img1, img2).item() < 1.0

if __name__ == "__main__":
    pytest.main()