import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def test_numpy():
    """Test if numpy is working"""
    arr = np.array([1, 2, 3])
    assert arr.mean() == 2

def test_matplotlib():
    """Test if matplotlib is working"""
    plt.figure()
    plt.plot([1, 2, 3], [1, 2, 3])
    plt.close()

def test_file_operations():
    """Test if file operations are working"""
    test_dir = Path('test_dir')
    test_dir.mkdir(exist_ok=True)
    
    test_file = test_dir / 'test.txt'
    test_file.write_text('test')
    
    assert test_file.exists()
    assert test_file.read_text() == 'test'
    
    # Clean up
    test_file.unlink()
    test_dir.rmdir() 