# Author: Tom Furnival
# License: GPLv3

import numpy as np
from Cython.Build import cythonize
from setuptools import find_packages, setup
from setuptools.extension import Extension

extensions = [
    Extension(
        "pguresvt._pguresvt",
        sources=["pguresvt/_pguresvt.pyx"],
        include_dirs=["pguresvt/", "src/", np.get_include()],
        libraries=["lapack", "armadillo", "nlopt"],
        language="c++",
        extra_compile_args=[
            "-fPIC",
            "-O3",
            "-march=native",
            "-pthread",
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Wno-unused-parameter",
            "-Wno-unused-function",
            "-D NPY_NO_DEPRECATED_API",
        ],
    ),
]

exec(open("pguresvt/release_info.py").read())

setup(
    name="pguresvt",
    version=version,
    description=description,
    author=author,
    author_email=email,
    license=license,
    url=url,
    long_description=open("README.md").read(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    packages=find_packages(),
    package_data={"": ["LICENSE", "README.md"], "pguresvt": ["*.py", "tests/*"]},
    install_requires=["numpy"],
    ext_modules=cythonize(extensions),
)
