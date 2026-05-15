"""Test conftest - handles importing the pipeline scripts as modules.

The scripts (prepff, paramsgen, topolgen) have no .py extension and use
relative paths. This conftest uses importlib to load them for unit testing.
"""

import os
import sys
import tempfile
import pytest
from importlib.machinery import SourceFileLoader


def _import_script(script_name):
    """Import a pipeline script as a Python module.

    Uses SourceFileLoader because the scripts have no .py extension.
    """
    script_path = os.path.join(
        os.path.dirname(__file__), '..', 'scripts', script_name
    )
    try:
        mod = SourceFileLoader(script_name, script_path).load_module()
    except ImportError:
        # Allow tests to run without AIMNet/CUDA
        mod = SourceFileLoader(script_name, script_path).load_module()
    return mod


@pytest.fixture(scope='session')
def prepff():
    """Load the prepff module."""
    return _import_script('prepff')


@pytest.fixture(scope='session')
def paramsgen():
    """Load the paramsgen module."""
    return _import_script('paramsgen')


@pytest.fixture(scope='session')
def topolgen():
    """Load the topolgen module."""
    return _import_script('topolgen')


@pytest.fixture
def tmp_smiles_file():
    """Create a temporary SMILES file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.smiles', delete=False) as f:
        f.write('N[C@@H](C)C(=O)O\n')  # L-alanine
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def tmp_dir():
    """Create a temporary working directory."""
    old_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as d:
        os.chdir(d)
        yield d
        os.chdir(old_cwd)
