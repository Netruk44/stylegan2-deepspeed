import sys
from setuptools import setup, find_packages

sys.path[0:0] = ['stylegan2_deepspeed']
from version import __version__

setup(
  name = 'stylegan2_deepspeed',
  packages = find_packages(),
  entry_points={
    'console_scripts': [
      'stylegan2_deepspeed = stylegan2_deepspeed.cli:main',
    ],
  },
  version = __version__,
  install_requires=[
    'einops',
    'kornia>=0.5.4',
    'torch',
    'torchvision',
    'vector-quantize-pytorch>=0.1.0'
  ]
)