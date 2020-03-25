"""
Tests for augmented flows

Author: Michael Williams
"""
import pytest
import numpy as np
import torch

def forward_transform(x):
    """Intialise the Transform module and pass x"""
    from flows import Transform
    model = Transform(1, 1, 16)
    x = torch.Tensor(x)
    _ = model(x)
    return True

def test_transform():
    """Test the Transform module"""
    x = np.random.randn(1, 1).astype(np.float32)
    assert forward_transform(x)

def forward_backward_block(x):
    """
    Consecutive forward and backward pass of a point
    """
    from flows import AugmentedBlock
    block = AugmentedBlock(1,1)
    feature = torch.Tensor(x)
    augment = torch.randn_like(feature)
    y, z, _ = block(feature, augment, mode='forward')
    feature_out, augment_out, _ = block(y, z, mode='generate')
    return feature_out.detach().cpu().numpy()

def test_block_invertibility():
    """
    Test the invertibility of the AugmentedBlock
    """
    x = np.random.randn(1, 1).astype(np.float32)
    np.testing.assert_array_equal(x, forward_backward_block(x))

def forward_backward_sequential(x):
    """
    Consecutive forward and backward pass of a point through AugmentedSequential
    """
    from flows import AugmentedBlock, AugmentedSequential
    blocks = []
    for n in range(4):
        blocks += [AugmentedBlock(1, 1)]
    model = AugmentedSequential(*blocks)
    feature = torch.Tensor(x)
    augment = torch.randn_like(feature)
    y, z, _ = model(feature, augment, mode='forward')
    feature_out, augment_out, _ = model(y, z, mode='generate')
    return feature_out.detach().cpu().numpy()

def test_sequential_invertibility():
    """
    Test the invertibility of the AugmentedSequential
    """
    x = np.random.randn(1, 1).astype(np.float32)
    np.testing.assert_array_equal(x, forward_backward_sequential(x))
