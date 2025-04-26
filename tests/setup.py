"""
Setup file for pytest to ensure the src package is importable during tests.
"""
from setuptools import setup, find_packages

# Setup with development install for testing
setup(
    name="vipunen-tests",
    version="0.1.0",
    description="Tests for Vipunen project",
    author="Topi Jarvinen",
    packages=find_packages(),
) 