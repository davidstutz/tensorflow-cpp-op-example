# Example of Tensorflow Operation in C++

This repository contains an example of a simple Tensorflow operation and its gradient both implemented in C++, as described in [this article](http://davidstutz.de/implementing-tensorflow-operations-in-c-including-gradients/).

## Building

The operation is built using [CMake](https://cmake.org/) and requires an appropriate version of Tensorflow to be installed. In order to get the necessary include directories containing the Tensorflow header files, the following trick is used (also see the [Tensorflow documentation](https://www.tensorflow.org/how_tos/adding_an_op/)):

    import tensorflow
    print(tensorflow.sysconfig.get_include())

In the `CMakeLists.txt` this is used as follows:

    execute_process(COMMAND python3 -c "import tensorflow; print(tensorflow.sysconfig.get_include())" OUTPUT_VARIABLE Tensorflow_INCLUDE_DIRS)

The remaining contents are pretty standard. Building is now done using:

    $ mkdir build
    $ cd build
    $ cmake ..
    $ make
    Scanning dependencies of target inner_product
    [ 50%] Building CXX object CMakeFiles/inner_product.dir/inner_product.cc.o
    Linking CXX shared library libinner_product.so
    [ 50%] Built target inner_product
    Scanning dependencies of target inner_product_grad
    [100%] Building CXX object CMakeFiles/inner_product_grad.dir/inner_product_grad.cc.o
    Linking CXX shared library libinner_product_grad.so
    [100%] Built target inner_product_grad

`libinner_product.so` and `libinner_product_grad.so` can be found in `build` and need to be included in order to load the module in Python:

    import tensorflow as tf
    inner_product_module = tf.load_op_library('build/libinner_product.so')

See `inner_product_tests.py` for usage examples.

## License

Copyright (c) 2016 David Stutz

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
