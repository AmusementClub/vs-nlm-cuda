# vs-nlm-cuda
Non-local means denoise filter in CUDA, drop-in replacement of the KNLMeansCL for VapourSynth

## Usage
Prototype:

`core.nlm_cuda.NLMeans(clip clip[, int d = 1, int a = 2, int s = 4, float h = 1.2, string channels = "AUTO", int wmode = 0, float wref = 1.0, clip rclip = None, int device_id = 0, int num_streams = 1])`

## Compilation
```bash
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release \
-D CMAKE_CUDA_FLAGS="--use_fast_math" \
-D CMAKE_CUDA_ARCHITECTURES="50;61-real;70-virtual;75-real;86-real;89-real"

cmake --build build

cmake --install build
```

