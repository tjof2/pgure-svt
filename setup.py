# Copyright 2015-2020 Tom Furnival
#
# This file is part of PGURE-SVT.
#
# PGURE-SVT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PGURE-SVT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PGURE-SVT.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from Cython.Build import cythonize
from setuptools import find_packages, setup
from setuptools.extension import Extension

extensions = [
    Extension(
        "pguresvt._pguresvt",
        sources=["pguresvt/_pguresvt.pyx"],
        include_dirs=["pguresvt/", "src/", np.get_include()],
        libraries=["openblas", "lapack", "armadillo", "nlopt"],
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
    package_data={"": ["LICENSE", "README.md"], "pguresvt": ["*.py"]},
    install_requires=["numpy"],
    setup_requires=["wheel", "auditwheel"],
    ext_modules=cythonize(extensions),
)
