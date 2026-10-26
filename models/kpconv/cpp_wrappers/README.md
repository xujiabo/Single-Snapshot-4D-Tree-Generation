# Optional KPConv C++ extensions

Compilation is optional. The project works with NumPy voxel subsampling and SciPy radius-neighbour search when these extensions are unavailable. Compiling them accelerates geometry preprocessing; the learned KPConv network is unchanged. Voxel ordering and floating-point rounding may differ between backends.

Use the Python environment in which you run this project:

```sh
python -m pip install setuptools "numpy<2"
```

On Linux or macOS, install a C++ compiler and run this command from the repository root:

```sh
bash models/kpconv/cpp_wrappers/compile_wrappers.sh
```

On Windows, install Visual Studio Build Tools with the C++ workload, then run these commands in PowerShell from the repository root:

```powershell
& .\models\kpconv\cpp_wrappers\cpp_subsampling\build.bat
& .\models\kpconv\cpp_wrappers\cpp_neighbors\build.bat
```

The scripts resolve their own directories, so they can also be invoked by absolute path from any working directory. They use `python` from the active environment; set `PYTHON` to a Python executable path to override it. Rebuild after changing the Python environment or NumPy version. Generated binaries and build directories are excluded from version control.

Validation: both extensions were compiled and imported on Windows with CPython 3.8, NumPy 1.24.4, and Visual Studio 2019. A two-cloud fixture checked native and fallback preprocessing values. The Linux/macOS build script was reviewed but was not executed on those platforms. Full pipeline checks use the NumPy/SciPy fallback.
