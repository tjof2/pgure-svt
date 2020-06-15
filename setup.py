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

from setuptools import find_packages, setup

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
    package_data={"": ["LICENSE", "README.md"]},
    python_requires=">=3.6",
    install_requires=["numpy"],
    setup_requires=["wheel", "auditwheel"],
)
