import pytest
import torch
import numpy as np
from utils.graphics_utils import (
    geom_transform_points,
    getWorld2View,
    getWorld2View2,
    getProjectionMatrix,
    fov2focal,
    focal2fov
)

@pytest.fixture
def test_data():
    points = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    transf_matrix = torch.eye(4, dtype=torch.float32)
    return points, transf_matrix

def test_geom_transform_points(test_data):
    points, transf_matrix = test_data
    transformed_points = geom_transform_points(points, transf_matrix)
    
    expected_points = points
    assert torch.allclose(transformed_points, expected_points), f"Expected {expected_points}, but got {transformed_points}"

def test_getWorld2View():
    R = np.eye(3)
    t = np.array([1.0, 2.0, 3.0])
    
    world2view_matrix = getWorld2View(R, t)
    expected_matrix = np.array([
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 2.0],
        [0.0, 0.0, 1.0, 3.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    assert np.allclose(world2view_matrix, expected_matrix), f"Expected {expected_matrix}, but got {world2view_matrix}"

def test_getWorld2View2():
    R = np.eye(3)
    t = np.array([1.0, 2.0, 3.0])
    translate = np.array([0.0, 0.0, 0.0])
    scale = 1.0
    
    world2view_matrix = getWorld2View2(R, t, translate, scale)
    
    expected_matrix = np.array([
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 2.0],
        [0.0, 0.0, 1.0, 3.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    assert np.allclose(world2view_matrix, expected_matrix), f"Expected {expected_matrix}, but got {world2view_matrix}"

def test_getProjectionMatrix():
    znear = 0.1
    zfar = 100.0
    fovX = np.radians(90.0)
    fovY = np.radians(90.0)
    
    projection_matrix = getProjectionMatrix(znear, zfar, fovX, fovY)
    
    assert projection_matrix.shape == (4, 4), "Projection matrix should be 4x4"
    assert projection_matrix[0, 0] != 0.0, "First diagonal element should not be zero"
    assert projection_matrix[3, 2] == 1.0, "Fourth column, third row should be 1"

def test_fov2focal():
    fov = np.radians(90.0)
    pixels = 800
    focal_length = fov2focal(fov, pixels)
    
    expected_focal_length = pixels / (2 * np.tan(fov / 2))
    assert np.isclose(focal_length, expected_focal_length), f"Expected {expected_focal_length}, but got {focal_length}"

def test_focal2fov():
    focal = 500.0
    pixels = 1000
    fov = focal2fov(focal, pixels)
    
    expected_fov = 2 * np.arctan(pixels / (2 * focal))
    assert np.isclose(fov, expected_fov), f"Expected {expected_fov}, but got {fov}"
