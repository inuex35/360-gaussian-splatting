import os
import pytest
from utils.system_utils import mkdir_p, searchForMaxIteration

def test_mkdir_p(tmpdir):
    # Test creating a new directory
    new_dir = tmpdir.mkdir("new_dir")
    mkdir_p(new_dir)
    assert new_dir.exists() and new_dir.isdir()

    # Test creating an existing directory (should not raise an exception)
    mkdir_p(new_dir)
    assert new_dir.exists() and new_dir.isdir()

def test_searchForMaxIteration(tmpdir):
    # Create some dummy files with iteration numbers
    iterations = [1, 2, 3, 10, 20]
    for i in iterations:
        tmpdir.join(f"file_{i}").ensure()

    # Test finding the maximum iteration
    max_iter = searchForMaxIteration(tmpdir)
    assert max_iter == max(iterations)

    # Test with no files in the directory
    empty_dir = tmpdir.mkdir("empty_dir")
    mkdir_p(empty_dir)
    with pytest.raises(ValueError):
        searchForMaxIteration(empty_dir)

if __name__ == "__main__":
    pytest.main()