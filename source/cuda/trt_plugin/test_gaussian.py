import numpy as np
import os
import sys
import tensorrt as trt
import common

parent_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
sys.path.insert(1, parent_dir)
import torch
import torch.nn.functional as F

from load_plugin_lib import load_plugin_lib

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

def create_gaussian_kernel(kernel_size: int, sigma: float) -> np.ndarray:
    """Create 1D Gaussian kernel for separable convolution"""
    if kernel_size % 2 == 0:
        kernel_size += 1

    x = np.arange(kernel_size, dtype=np.float32) - kernel_size // 2
    gauss_1d = np.exp(-(x ** 2) / (2 * sigma ** 2))
    gauss_1d = gauss_1d / gauss_1d.sum()

    return gauss_1d

def gaussian_reference_impl(tensor, kernel_size=5, sigma=1.5):
    """
    Reference implementation matching flnr.py's _separable_gaussian_blur
    Input: tensor [B, C, H, W] (NCHW format, NumPy)
    Output: blurred [B, C, H, W]

    IMPORTANT: Uses ZERO padding (F.conv2d default) to match flnr.py implementation
    """
    # Convert to torch for easier processing
    tensor_torch = torch.from_numpy(tensor)

    padding = kernel_size // 2
    channels = tensor_torch.shape[1]

    # Create Gaussian kernel
    gauss_kernel_1d = create_gaussian_kernel(kernel_size, sigma)
    gauss_kernel_1d = torch.from_numpy(gauss_kernel_1d)

    # Expand kernel for conv2d
    kernel_h = gauss_kernel_1d.view(1, 1, kernel_size, 1).expand(channels, 1, -1, 1)
    kernel_v = gauss_kernel_1d.view(1, 1, 1, kernel_size).expand(channels, 1, 1, -1)

    # Apply separable convolution with ZERO padding (matching flnr.py)
    # First pass: vertical blur (H direction) with padding=(padding, 0)
    blurred = F.conv2d(tensor_torch, kernel_h, padding=(padding, 0), groups=channels)

    # Second pass: horizontal blur (W direction) with padding=(0, padding)
    blurred = F.conv2d(blurred, kernel_v, padding=(0, padding), groups=channels)

    return blurred.numpy()


def make_trt_network_and_engine(B, C, H, W, sigma):
    print("\n=== Building TensorRT Engine ===")
    print(f"Loading plugin library...")
    plugins_loaded = load_plugin_lib()
    print(f"Loaded plugins: {plugins_loaded}")

    registry = trt.get_plugin_registry()
    print(f"Registry has {len(registry.plugin_creator_list)} plugin creators")

    plugin_creator = registry.get_plugin_creator("GaussianBlurPlugin", "1")
    if plugin_creator is None:
        raise RuntimeError("Could not find GaussianBlurPlugin")

    print(f"Found plugin creator: {plugin_creator.name}")

    # Set plugin parameter (sigma)
    sigma_buffer = np.array([sigma], dtype=np.float32)
    sigma_attr = trt.PluginField("sigma", sigma_buffer, type=trt.PluginFieldType.FLOAT32)

    field_collection = trt.PluginFieldCollection([sigma_attr])
    plugin = plugin_creator.create_plugin(
        name="GaussianBlur", field_collection=field_collection
    )

    print(f"Created plugin instance with sigma={sigma}")

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(0)
    config = builder.create_builder_config()
    runtime = trt.Runtime(TRT_LOGGER)

    # Input shape: NCHW format
    input_shape = (B, C, H, W)

    print(f"Input shape: {input_shape}")

    input_tensor = network.add_input(name="input", dtype=trt.float32, shape=input_shape)

    gaussian = network.add_plugin_v2(inputs=[input_tensor], plugin=plugin)
    network.mark_output(gaussian.get_output(0))

    print(f"Building engine...")
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("Failed to build TensorRT engine")

    engine = runtime.deserialize_cuda_engine(plan)
    print(f"Engine built successfully")

    return engine


def custom_plugin_impl(input_data, engine):
    """Run inference with GaussianBlurPlugin"""
    print("\n=== Running Plugin Inference ===")

    # Print input data info before inference
    print(f"Input data info:")
    print(f"  Shape: {input_data.shape}")
    print(f"  Dtype: {input_data.dtype}")
    print(f"  Range: [{input_data.min():.6f}, {input_data.max():.6f}]")
    print(f"  Mean: {input_data.mean():.6f}, Std: {input_data.std():.6f}")
    print(f"  First 5 values: {input_data.flatten()[:5]}")

    inputs, outputs, bindings, stream = common.allocate_buffers(engine)

    print(f"\nBuffer allocation:")
    print(f"  Input buffer shape: {inputs[0].host.shape}")
    print(f"  Output buffer shape: {outputs[0].host.shape}")

    context = engine.create_execution_context()

    # Set input tensor (NCHW format, flattened)
    inputs[0].host[:] = input_data.astype(np.float32).flatten()

    print(f"\nInput buffer after copy:")
    print(f"  First 5 values: {inputs[0].host[:5]}")

    print("\nExecuting inference...")
    trt_outputs = common.do_inference(
        context,
        engine=engine,
        bindings=bindings,
        inputs=inputs,
        outputs=outputs,
        stream=stream,
    )

    print(f"Inference complete")
    print(f"Output info:")
    print(f"  Output shape: {trt_outputs[0].shape}")
    print(f"  Output range: [{trt_outputs[0].min():.6f}, {trt_outputs[0].max():.6f}]")
    print(f"  Output mean: {trt_outputs[0].mean():.6f}, std: {trt_outputs[0].std():.6f}")
    print(f"  First 5 values: {trt_outputs[0][:5]}")

    output = trt_outputs[0].copy()
    common.free_buffers(inputs, outputs, stream)
    return output


def main():
    print("=== GaussianBlur Plugin Test ===\n")

    # Test parameters - start with smaller channel count to debug
    # B, C, H, W = 2, 16, 68, 120  # Full test (may exceed shared memory)
    B, C, H, W = 2, 4, 68, 120  # Reduced channels for debugging
    kernel_size = 5
    sigma = 1.5

    print(f"Test config: B={B}, C={C}, H={H}, W={W}")
    print(f"Kernel size: {kernel_size}, Sigma: {sigma}")

    # Generate test data (NCHW format)
    np.random.seed(42)
    input_data = np.random.randn(B, C, H, W).astype(np.float32) * 0.1

    print(f"\nInput shape: {input_data.shape}")
    print(f"Input sample [0,0,0,:5]: {input_data[0,0,0,:5]}")
    print(f"Input range: [{input_data.min():.4f}, {input_data.max():.4f}]")

    # Run reference implementation
    print("\n=== Reference Implementation ===")
    res_ref = gaussian_reference_impl(input_data, kernel_size=kernel_size, sigma=sigma)
    print(f"Reference output shape: {res_ref.shape}")
    print(f"Reference output sample [0,0,0,:5]: {res_ref[0,0,0,:5]}")
    print(f"Reference output range: [{res_ref.min():.4f}, {res_ref.max():.4f}]")

    # Build TensorRT engine and run plugin
    try:
        engine = make_trt_network_and_engine(B, C, H, W, sigma)
        res_plugin = custom_plugin_impl(input_data, engine)

        # Reshape and compare
        res_plugin = res_plugin.reshape(res_ref.shape)
        print(f"\nPlugin output shape: {res_plugin.shape}")
        print(f"Plugin output sample [0,0,0,:5]: {res_plugin[0,0,0,:5]}")
        print(f"Plugin output range: [{res_plugin.min():.4f}, {res_plugin.max():.4f}]")

        # Check results
        max_diff = np.max(np.abs(res_ref - res_plugin))
        mean_diff = np.mean(np.abs(res_ref - res_plugin))
        rel_error = np.mean(np.abs(res_ref - res_plugin) / (np.abs(res_ref) + 1e-8))

        print(f"\n=== Comparison Results ===")
        print(f"Max absolute difference: {max_diff:.6f}")
        print(f"Mean absolute difference: {mean_diff:.6f}")
        print(f"Mean relative error: {rel_error:.6f}")

        # Show some detailed comparisons
        print(f"\nSample comparison [0,0,0,:5]:")
        print(f"Reference: {res_ref[0,0,0,:5]}")
        print(f"Plugin:    {res_plugin[0,0,0,:5]}")
        print(f"Diff:      {(res_ref - res_plugin)[0,0,0,:5]}")

        # Tolerance for float32 operations
        threshold = 1e-4  # More relaxed than linalg due to separable conv

        if max_diff > threshold:
            print(f"\n FAILED! Max difference {max_diff} exceeds threshold {threshold}")

            # Find location of max difference
            max_idx = np.unravel_index(np.argmax(np.abs(res_ref - res_plugin)), res_ref.shape)
            print(f"\nMax difference at index {max_idx}:")
            print(f"  Reference: {res_ref[max_idx]}")
            print(f"  Plugin:    {res_plugin[max_idx]}")
            print(f"  Diff:      {res_ref[max_idx] - res_plugin[max_idx]}")

            return False
        else:
            print(f"\n PASSED! All differences within threshold {threshold}")
            return True

    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
