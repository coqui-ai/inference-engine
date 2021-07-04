# Coqui Inference Engine

**Coqui Inference Engine** is a library for efficiently deploying speech models.

This project is at an early proof-of-concept stage. Collaboration on design and implementation is very welcome. Join our Gitter channel by clicking the badge above!

This project is the successor to the STT "native client", containing the core inference logic for deploying Coqui STT models (and eventually Coqui TTS and other models too).

Coqui Inference Engine aims to be:

- Fast: streaming inference with low latency on small devices (phones, IoT)
- Easy to use: simple, stable, well-documented API
- Available: easy to expose to different programming languages, available in standard package managers
- Extensible: able to handle different model types, architectures, and formats

[![Covenant](https://camo.githubusercontent.com/7d620efaa3eac1c5b060ece5d6aacfcc8b81a74a04d05cd0398689c01c4463bb/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f436f6e7472696275746f72253230436f76656e616e742d76322e3025323061646f707465642d6666363962342e737667)](CODE_OF_CONDUCT.md)
[![Gitter](https://badges.gitter.im/coqui-ai/inference-engine.svg)](https://gitter.im/coqui-ai/inference-engine?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

📰 [**Subscribe to the 🐸Coqui.ai Newsletter**](https://coqui.ai/?subscription=true)

## Build instructions

For the build you'll need to install [CMake >= 3.10](https://cmake.org/).

Currently you'll also have to build [onnxruntime](https://onnxruntime.ai) yourself and place the built files manually before building the inference engine, following the steps below:

```bash
$ # Clone the Coqui Inference Engine repo
$ git clone https://github.com/coqui-ai/inference-engine
$ # Clone onnxruntime repo
$ git clone --recursive https://github.com/microsoft/onnxruntime/
$ cd onnxruntime
$ # Build it
$ ./build.sh --config Debug --parallel
$ # Copy built files for inference engine build
$ cp -R build/*/*/libonnxruntime* ../inference-engine/onnxruntime/lib
```

Now, we're ready to build the inference engine:

```bash
$ cd ../inference-engine
$ # Create build dir
$ mkdir build
$ cd build
$ # Prepare build
$ cmake -DCMAKE_BUILD_TYPE=Debug ..
$ # Build
$ make -j
```

You should now be able to run the test client by running `./main`.

```bash
$ ./main --model ../output_graph.onnx --audio ../test-audio.wav
```

## Exporting a test model from Coqui STT

You can use the `experimental-inference-engine-export` branch of [Coqui STT](https://github.com/coqui-ai/STT) to export an STT checkpoint in the format expected by the inference engine.

```bash
$ git clone --branch experimental-inference-engine-export https://github.com/coqui-ai/STT
$ cd STT
$ python -m pip install -e .
$ cd native_client/ctcdecode
$ make bindings
$ python -m pip install --force-reinstall dist/*.whl
```

After the steps above, you can then follow the [documentation for exporting a model](https://stt.readthedocs.io/en/latest/EXPORTING_MODELS.html#exporting-checkpoints), and include the `--export_onnx true` flag. You should then get an `output_graph.onnx` file exported which can be read by the inference engine.
