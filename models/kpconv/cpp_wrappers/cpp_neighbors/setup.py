"""Build the optional radius-neighbour extension with the active Python."""
import os
from pathlib import Path

import numpy as np
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class BuildExt(build_ext):
    """Select flags for the compiler actually chosen by setuptools."""

    def build_extensions(self):
        if self.compiler.compiler_type == "msvc":
            flags = ["/O2", "/EHsc", "/std:c++14"]
        else:
            flags = ["-O3", "-std=c++11"]
        for extension in self.extensions:
            extension.extra_compile_args = flags
        super().build_extensions()


if __name__ == "__main__":
    # Resolve source paths and --inplace output independently of the shell cwd.
    os.chdir(Path(__file__).resolve().parent)
    setup(
        name="radius_neighbors",
        packages=[],
        ext_modules=[Extension(
            name="radius_neighbors",
            sources=["../cpp_utils/cloud/cloud.cpp", "neighbors/neighbors.cpp", "wrapper.cpp"],
            include_dirs=[np.get_include()],
            language="c++",
        )],
        cmdclass={"build_ext": BuildExt},
    )
