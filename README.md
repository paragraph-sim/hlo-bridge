# HLO Bridge

This is a ParaGraph bridges that organizes support for HLO files dumped by [XLA](https://www.tensorflow.org/xla). 

## How to install
The C++ projects use [Bazel](https://bazel.build/ "Bazel Build") for building binaries. To install Bazel, follow the directions at [here](https://bazel.build/install "Bazel Install"). You need bazel-3.7.2 and a bit of patience. The bridge requires to compile good chunk of TensorFlow repo for XLA support.
Use hte following command to build and testt the project 
```
bazel test -c opt ...
```
To build the bridge without linter check, you can use 
```
bazel build bridge:hlo_bridge
```

## How to prepare HLO files
To dump files from the XLA-supported project, e.g. [TensorFlow](https://www.tensorflow.org/), [JAX](https://github.com/google/jax), or [PyTorch-XLA](https://github.com/pytorch/xla), use XLA flags
```
export XLA_FLAGS="--xla_dump_to=./hlo_files --xla_dump_hlo_as_text=true "
```
We provide some [JAX examples](https://github.com/paragraph-sim/hlo-examples/tree/main/jax) as a reference point.

## How to use bridge
Use `./bazel-bin/bridge/hlo_bridge --help` to see available options.
