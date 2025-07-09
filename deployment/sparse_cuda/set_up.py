from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import pybind11
import torch
import os

# Get include directories
include_dirs = [
    pybind11.get_include(),
    torch.utils.cpp_extension.include_paths()[0],
    os.path.join(torch.utils.cpp_extension.include_paths()[0], 'torch/csrc/api/include'),
]

# Compiler flags
cxx_flags = [
    '-O3', 
    '-std=c++17',
    '-fPIC',
    '-Wall'
]

nvcc_flags = [
    '-O3', 
    '--use_fast_math',
    '-gencode=arch=compute_53,code=sm_53',  # Jetson Nano
    '-gencode=arch=compute_61,code=sm_61',  # GTX 1080
    '-gencode=arch=compute_75,code=sm_75',  # RTX 2080
    '-gencode=arch=compute_86,code=sm_86',  # RTX 3080
    '--expt-relaxed-constexpr',
    '-std=c++17'
]

# Define extension module
ext_modules = [
    CUDAExtension(
        name='spmm_cuda',
        sources=[
            'binding.cpp', 
            'csr_spmm_kernel.cu'
        ],
        include_dirs=include_dirs,
        extra_compile_args={
            'cxx': cxx_flags, 
            'nvcc': nvcc_flags
        },
        extra_link_args=['-lcudart', '-lcublas', '-lcusparse']
    )
]

# Setup configuration
setup(
    name='spmm_cuda',
    version='1.0.0',
    description='CUDA-accelerated sparse matrix multiplication for PyTorch',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/your-username/sparse_cuda',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
    python_requires='>=3.7',
    install_requires=[
        'torch>=1.7.0',
        'numpy>=1.19.0'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: C++',
        'Programming Language :: CUDA',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='cuda pytorch sparse matrix multiplication spmm',
    long_description='''
    A high-performance CUDA extension for PyTorch that provides optimized 
    sparse matrix multiplication operations. This package implements efficient 
    CSR (Compressed Sparse Row) format SpMM kernels for accelerated sparse 
    deep learning computations.
    
    Features:
    - Optimized CUDA kernels for sparse matrix multiplication
    - Support for multiple GPU architectures
    - Seamless PyTorch integration
    - Easy-to-use Python API
    ''',
    long_description_content_type='text/plain'
)
