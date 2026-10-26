# Third-party notices

## KPConv

`models/kpconv/` includes code adapted from Hugues Thomas's
[KPConv-PyTorch](https://github.com/HuguesTHOMAS/KPConv-PyTorch) implementation.
The original author headers remain in the source files. Its MIT license is
included in [licenses/KPConv-MIT.txt](licenses/KPConv-MIT.txt), copied from the
[upstream license](https://github.com/HuguesTHOMAS/KPConv-PyTorch/blob/master/LICENSE.txt).

Local adaptations include portable geometry preprocessing, device/path handling,
normalization blocks, and optional extension build scripts.

## nanoflann

The optional KPConv C++ neighbor implementation includes nanoflann in
`models/kpconv/cpp_wrappers/cpp_utils/nanoflann/nanoflann.hpp`.
Its original BSD license and copyright notice remain at the top of that header.
A copy is also included in [licenses/nanoflann-BSD.txt](licenses/nanoflann-BSD.txt).

These notices cover the identified third-party components; they do not assign a
new license to the project-specific code, data, or animations.
