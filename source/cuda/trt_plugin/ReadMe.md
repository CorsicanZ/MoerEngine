# FLNR TensorRT Plugins
高性能TensorRT插件集合，用于加速FLNR4视频去噪模型推理

## 插件列表
- **LinalgSolvePlugin**: 线性求解器（替代 `torch.linalg.solve`）
- **GaussianBlurPlugin**: 高斯模糊（替代可分离卷积）

## 文件结构

```
trt_plugin/
├── ReadMe.md                      # 本文件
├── CMakeLists.txt                 # 主CMake构建配置（构建两个插件）
├── common.py                      # Python 通用函数
├── load_plugin_lib.py             # 动态加载插件库
├── plugin_op.py                   # PyTorch自定义算子（含ONNX symbolic）
├── test_linalg.py                 # LinalgSolve插件测试脚本
├── test_gaussian.py               # GaussianBlur插件测试脚本
├── include/                       # 共享头文件目录
│   ├── common.h                   # C++ 通用头文件（错误报告、宏定义、工具函数）
│   └── logger.h                   # TensorRT 日志系统
├── linalg_solve/                  # LinalgSolve插件源代码
│   ├── linalgSolvePlugin.h        # 插件头文件
│   ├── linalgSolvePlugin.cpp      # 插件实现（支持 FP16/FP32）
│   ├── linalgSolveKernels.h       # CUDA内核头文件
│   └── linalgSolveKernels.cu      # CUDA内核实现（cuBLAS批量求解）
├── gaussian_blur/                 # GaussianBlur插件源代码
│   ├── GaussianBlurPlugin.h       # 插件头文件
│   ├── GaussianBlurPlugin.cpp     # 插件实现（支持 FP16/FP32）
│   ├── GaussianBlurKernels.h      # CUDA内核头文件
│   └── GaussianBlurKernels.cu     # CUDA内核实现（可分离高斯模糊，共享内存优化）
└── archive/                       # 旧版本备份
```

## 插件功能

### LinalgSolvePlugin
- **功能**: 替代 `torch.linalg.solve`，使用 cuBLAS 批量求解线性系统
- **对应代码**: `flnr.py` 的 `_solve_linear_system` 方法
- **输入**:
  - `XTX`: [B, (Q+1)², H_ds, W_ds] - 外积矩阵（已模糊）
  - `XTY`: [B, (Q+1)*C, H_ds, W_ds] - 交叉乘积（已模糊）
- **输出**:
  - `coeffs`: [B, (Q+1)*C, H_ds, W_ds] - 回归系数
- **参数**:
  - `q_plus_one` (int): 矩阵维度 Q+1
  - `epsilon` (float): Tikhonov 正则化参数
  - `eta` (float): 额外正则化参数
- **特性**:
  - 自动 FP16↔FP32 类型转换
  - 使用 cuBLAS 批量求解器 (cublasSgetrfBatched + cublasSgetrsBatched)
  - 支持大规模并行线性系统求解

### GaussianBlurPlugin
- **功能**: 高性能可分离高斯模糊，替代PyTorch Conv2D实现
- **对应代码**: `flnr.py` 的 `_separable_gaussian_blur` 方法
- **输入**:
  - `input`: [B, C, H, W] - 输入张量（NCHW格式）
- **输出**:
  - `output`: [B, C, H, W] - 模糊后的张量
- **参数**:
  - `sigma` (float): 高斯分布标准差
- **特性**:
  - 可分离卷积（垂直H方向 + 水平W方向两遍）
  - **Zero padding**（匹配 PyTorch F.conv2d 默认行为，与 flnr.py 一致）
  - 共享内存优化，减少全局内存访问
  - 编译时模板特化（支持1-16通道）
  - 支持 FP16 和 FP32
- **重要**: 使用 zero padding 而非 reflect padding，与 flnr.py 的实现保持一致

## 编译方式

### 前置要求
- CUDA 12.6 或更高版本
- cuBLAS 12.6 或更高版本
- TensorRT 8.6 或更高版本
- CMake 3.18 或更高版本
- MSVC 2019/2022 或 GCC 9.4+
- ONNX 导出依赖：`pip install onnx onnx-graphsurgeon`

### Windows 编译

#### CMake
```bash
# 进入插件目录
cd scripts/models/regression/trt_plugin

# 创建构建目录
mkdir build && cd build

# 配置项目
cmake .. -G "Visual Studio 17 2022" -A x64 ^
    -DCUDA_TOOLKIT_ROOT_DIR="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6" ^
    -DTENSORRT_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT"

# 编译（会生成两个插件）
cmake --build . --config Release

# 或使用 VS Developer Command Prompt:
# msbuild FLNRPlugins.sln /p:Configuration=Release /p:Platform=x64
```


### Linux 编译(4090服务器)

```bash
# 进入插件目录
cd scripts/models/regression/trt_plugin

# 创建构建目录
mkdir build && cd build

# 配置项目
cmake .. \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
    -DTENSORRT_ROOT=/usr/src/tensorrt

# 编译（会生成两个插件）
make -j$(nproc)

# 输出文件
ls -la *.so
```

## 输出文件

编译后会生成两个插件库：

- **Windows**:
  - `build/Release/linalgSolvePlugin.dll`
  - `build/Release/gaussianBlurPlugin.dll`
- **Linux**:
  - `build/liblinalgSolvePlugin.so`
  - `build/libgaussianBlurPlugin.so`

## 使用方法

### 1. 在 Python 中加载插件
```python
# 方法1: 使用提供的加载器
from scripts.models.regression.trt_plugin.load_plugin_lib import load_plugin_lib
plugins_loaded = load_plugin_lib()  # 返回 ['LinalgSolve', 'GaussianBlur']
print(f"Loaded plugins: {plugins_loaded}")

# 方法2: 手动加载
import ctypes
linalg_plugin = "scripts/models/regression/trt_plugin/build/Release/linalgSolvePlugin.dll"
gaussian_plugin = "scripts/models/regression/trt_plugin/build/Release/gaussianBlurPlugin.dll"
ctypes.CDLL(linalg_plugin)
ctypes.CDLL(gaussian_plugin)

# 现在 TensorRT 可以识别自定义插件
import tensorrt as trt
logger = trt.Logger(trt.Logger.WARNING)
```

### 2. 测试插件
```bash
# 测试 LinalgSolve 插件
cd scripts/models/regression/trt_plugin
python test_linalg.py

# 测试 GaussianBlur 插件
python test_gaussian.py
```

### 3. 在 ONNX 导出中使用

参考 `create_onnx.py` 和 `build_trt_engine.py`：

```bash
# 1. 导出带插件的 ONNX 模型
python -m scripts.models.regression.create_onnx \
    --checkpoint model.pth \
    --onnx model_plugin.onnx \
    --height 540 --width 960

# 2. 构建 TensorRT 引擎
python -m scripts.models.regression.build_trt_engine \
    --onnx model_plugin.onnx \
    --engine model.engine \
    --plugin trt_plugin/build/Release/linalgSolvePlugin.dll \
    --fp16
```

### 4. 性能优化效果

- **LinalgSolvePlugin**: ~16.5ms (优化后，替代PyTorch linalg.solve)
- **GaussianBlurPlugin**: 预期大幅优化（原ONNX Expand操作: 538ms）
- **总体目标**: 从1020ms降至~10ms (RTX 3090, 60 GFlops)