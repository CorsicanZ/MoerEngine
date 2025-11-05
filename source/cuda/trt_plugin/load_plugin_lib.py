#
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import ctypes

WORKING_DIR = os.environ.get("TRT_WORKING_DIR") or os.path.dirname(
    os.path.realpath(__file__)
)
IS_WINDOWS = os.name == "nt"

# LinalgSolve plugin paths
# Both plugins are built in the main build directory (trt_plugin/build/)
if IS_WINDOWS:
    LINALG_PLUGIN_LIBRARY_NAME = "linalgSolvePlugin.dll"
    LINALG_PLUGIN_LIBRARY = [
        os.path.join(WORKING_DIR, "build", "Debug", LINALG_PLUGIN_LIBRARY_NAME),
        os.path.join(WORKING_DIR, "build", "Release", LINALG_PLUGIN_LIBRARY_NAME),
    ]
else:
    LINALG_PLUGIN_LIBRARY_NAME = "liblinalgSolvePlugin.so"
    LINALG_PLUGIN_LIBRARY = [
        os.path.join(WORKING_DIR, "build", LINALG_PLUGIN_LIBRARY_NAME)
    ]

# GaussianBlur plugin paths
if IS_WINDOWS:
    GAUSSIAN_PLUGIN_LIBRARY_NAME = "gaussianBlurPlugin.dll"
    GAUSSIAN_PLUGIN_LIBRARY = [
        os.path.join(WORKING_DIR, "build", "Debug", GAUSSIAN_PLUGIN_LIBRARY_NAME),
        os.path.join(WORKING_DIR, "build", "Release", GAUSSIAN_PLUGIN_LIBRARY_NAME),
    ]
else:
    GAUSSIAN_PLUGIN_LIBRARY_NAME = "libgaussianBlurPlugin.so"
    GAUSSIAN_PLUGIN_LIBRARY = [
        os.path.join(WORKING_DIR, "build", GAUSSIAN_PLUGIN_LIBRARY_NAME)
    ]


def load_plugin_lib():
    """Load both LinalgSolve and GaussianBlur TensorRT plugins"""
    plugins_loaded = []

    # Try to load LinalgSolve plugin
    linalg_found = False
    for plugin_lib in LINALG_PLUGIN_LIBRARY:
        print(f"Checking LinalgSolve plugin: {plugin_lib}")
        if os.path.isfile(plugin_lib):
            try:
                print(f"  Found! Loading...")
                # Python specifies that winmode is 0 by default, but some implementations
                # incorrectly default to None instead. See:
                # https://docs.python.org/3.8/library/ctypes.html
                # https://github.com/python/cpython/blob/3.10/Lib/ctypes/__init__.py#L343
                ctypes.CDLL(plugin_lib, winmode=0)
                plugins_loaded.append("LinalgSolve")
                print(f"  ✓ LinalgSolve plugin loaded successfully")
                linalg_found = True
            except TypeError:
                # winmode only introduced in python 3.8
                ctypes.CDLL(plugin_lib)
                plugins_loaded.append("LinalgSolve")
                print(f"  ✓ LinalgSolve plugin loaded successfully (legacy mode)")
                linalg_found = True
            except Exception as e:
                print(f"  ✗ Failed to load: {e}")
            break
        else:
            print(f"  Not found")

    if not linalg_found:
        print(f"⚠ LinalgSolve plugin not found in any of the expected locations")

    # Try to load GaussianBlur plugin
    gaussian_found = False
    for plugin_lib in GAUSSIAN_PLUGIN_LIBRARY:
        print(f"Checking GaussianBlur plugin: {plugin_lib}")
        if os.path.isfile(plugin_lib):
            try:
                print(f"  Found! Loading...")
                ctypes.CDLL(plugin_lib, winmode=0)
                plugins_loaded.append("GaussianBlur")
                print(f"  ✓ GaussianBlur plugin loaded successfully")
                gaussian_found = True
            except TypeError:
                ctypes.CDLL(plugin_lib)
                plugins_loaded.append("GaussianBlur")
                print(f"  ✓ GaussianBlur plugin loaded successfully (legacy mode)")
                gaussian_found = True
            except Exception as e:
                print(f"  ✗ Failed to load: {e}")
            break
        else:
            print(f"  Not found")

    if not gaussian_found:
        print(f"⚠ GaussianBlur plugin not found in any of the expected locations")

    if not plugins_loaded:
        raise IOError(
            "\n{}\n{}\n{}\n".format(
                "Failed to load TensorRT plugins.",
                "Please build the LinalgSolve and/or GaussianBlur plugins.",
                "For more information, see the included README.md",
            )
        )

    return plugins_loaded