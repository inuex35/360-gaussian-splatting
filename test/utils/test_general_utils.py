import unittest
import torch
import numpy as np
from PIL import Image
import sys
import io

from utils.general_utils import (
    inverse_sigmoid, PILtoTorch, get_expon_lr_func, strip_lowerdiag,
    strip_symmetric, build_rotation, build_scaling_rotation, safe_state
)

class TestGeneralUtils(unittest.TestCase):

    def test_inverse_sigmoid(self):
        x = torch.tensor([0.1, 0.5, 0.9])
        expected = torch.log(x / (1 - x))
        result = inverse_sigmoid(x)
        self.assertTrue(torch.allclose(result, expected))

    def test_PILtoTorch(self):
        pil_image = Image.new('RGB', (100, 100), color = 'red')
        resolution = (50, 50)
        result = PILtoTorch(pil_image, resolution)
        self.assertEqual(result.shape, (3, 50, 50))
        self.assertTrue(torch.all(result[0] == 1.0))  # Red channel should be 1.0

    def test_get_expon_lr_func(self):
        lr_init = 0.1
        lr_final = 0.01
        max_steps = 1000
        lr_func = get_expon_lr_func(lr_init, lr_final, max_steps=max_steps)
        self.assertAlmostEqual(lr_func(0), lr_init)
        self.assertAlmostEqual(lr_func(max_steps), lr_final)

    def test_strip_lowerdiag(self):
        L = torch.tensor([[[1, 0, 0], [2, 3, 0], [4, 5, 6]]], dtype=torch.float, device="cuda")
        expected = torch.tensor([[1, 0, 0, 3, 0, 6]], dtype=torch.float, device="cuda")
        result = strip_lowerdiag(L)
        self.assertTrue(torch.allclose(result, expected))

    def test_strip_symmetric(self):
        sym = torch.tensor([[[1, 0, 0], [2, 3, 0], [4, 5, 6]]], dtype=torch.float, device="cuda")
        expected = torch.tensor([[1, 0, 0, 3, 0, 6]], dtype=torch.float, device="cuda")
        result = strip_symmetric(sym)
        self.assertTrue(torch.allclose(result, expected))

    def test_build_rotation(self):
        r = torch.tensor([[1, 0, 0, 0]], dtype=torch.float, device="cuda")
        expected = torch.eye(3, device="cuda").unsqueeze(0)
        result = build_rotation(r)
        self.assertTrue(torch.allclose(result, expected))

    def test_build_scaling_rotation(self):
        s = torch.tensor([[1, 2, 3]], dtype=torch.float, device="cuda")
        r = torch.tensor([[1, 0, 0, 0]], dtype=torch.float, device="cuda")
        expected = torch.diag_embed(s)
        result = build_scaling_rotation(s, r)
        self.assertTrue(torch.allclose(result, expected))

    def test_safe_state(self):
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        safe_state(False)
        print("This should appear in logs")
        
        sys.stdout = sys.__stdout__
        
        output = captured_output.getvalue()
        self.assertIn("This should appear in logs", output)

if __name__ == '__main__':
    unittest.main()