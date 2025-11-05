import numpy as np
import os
import sys
import tensorrt as trt
import common

parent_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
sys.path.insert(1, parent_dir)
import torch

from load_plugin_lib import load_plugin_lib

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

def linalg_reference_impl(XTX, XTY, eps=1e-6):
    """
    Reference implementation matching TRT plugin's expected layout
    Input: XTX [B, H, W, (Q+1)^2], XTY [B, H, W, (Q+1)*C]
    Output: [B, H, W, (Q+1)*C]
    """
    B, H, W, Q_sq = XTX.shape
    Q_plus_1 = int(np.sqrt(Q_sq))
    C = XTY.shape[3] // Q_plus_1

    print(f"Reference: B={B}, H={H}, W={W}, Q+1={Q_plus_1}, C={C}")

    # Reshape for batch processing
    XTX_batch = XTX.reshape(-1, Q_plus_1, Q_plus_1)  # [B*H*W, Q+1, Q+1]
    XTY_batch = XTY.reshape(-1, Q_plus_1, C)          # [B*H*W, Q+1, C]

    # Add regularization
    eye = np.eye(Q_plus_1, dtype=XTX.dtype)
    XTX_reg = XTX_batch + eps * eye[np.newaxis, :, :]

    # Solve linear system
    A_batch = np.linalg.solve(XTX_reg, XTY_batch)

    # Reshape back
    A = A_batch.reshape(B, H, W, Q_plus_1 * C)

    return A


def make_trt_network_and_engine(B, H, W, Q_plus_1, C):
    print("\n=== Building TensorRT Engine ===")
    print(f"Loading plugin library...")
    load_plugin_lib()

    registry = trt.get_plugin_registry()
    print(f"Available plugins: {[registry.get_plugin_creator(registry.plugin_creator_list[i].name, '1') for i in range(len(registry.plugin_creator_list))]}")

    plugin_creator = registry.get_plugin_creator("LinalgSolvePlugin", "1")
    if plugin_creator is None:
        raise RuntimeError("Could not find LinalgSolvePlugin")

    print(f"Found plugin creator: {plugin_creator.name}")

    # Set plugin parameters
    epsilon = 0.01
    eta = 0.001

    q_buffer = np.array([Q_plus_1], dtype=np.int32)
    eps_buffer = np.array([epsilon], dtype=np.float32)
    eta_buffer = np.array([eta], dtype=np.float32)

    q_attr = trt.PluginField("q_plus_one", q_buffer, type=trt.PluginFieldType.INT32)
    eps_attr = trt.PluginField("epsilon", eps_buffer, type=trt.PluginFieldType.FLOAT32)
    eta_attr = trt.PluginField("eta", eta_buffer, type=trt.PluginFieldType.FLOAT32)

    field_collection = trt.PluginFieldCollection([q_attr, eps_attr, eta_attr])
    plugin = plugin_creator.create_plugin(
        name="LinalgSolve", field_collection=field_collection
    )

    print(f"Created plugin instance")

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(0)
    config = builder.create_builder_config()
    config.set_tactic_sources(
        config.get_tactic_sources() | 1 << int(trt.TacticSource.CUBLAS)
    )
    runtime = trt.Runtime(TRT_LOGGER)

    # Input shapes
    XTX_shape = (B, H, W, Q_plus_1 * Q_plus_1)
    XTY_shape = (B, H, W, Q_plus_1 * C)

    print(f"Input shapes: XTX={XTX_shape}, XTY={XTY_shape}")

    input_XTX = network.add_input(name="XTX", dtype=trt.float32, shape=XTX_shape)
    input_XTY = network.add_input(name="XTY", dtype=trt.float32, shape=XTY_shape)

    linalg = network.add_plugin_v2(inputs=[input_XTX, input_XTY], plugin=plugin)
    network.mark_output(linalg.get_output(0))

    print(f"Building engine...")
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("Failed to build TensorRT engine")

    engine = runtime.deserialize_cuda_engine(plan)
    print(f"Engine built successfully")

    return engine


def custom_plugin_impl(XTX, XTY, engine):
    """Run inference with LinalgSolvePlugin"""
    print("\n=== Running Plugin Inference ===")
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)

    print(f"Input 0 shape: {inputs[0].host.shape}")
    print(f"Input 1 shape: {inputs[1].host.shape}")
    print(f"Output 0 shape: {outputs[0].host.shape}")

    context = engine.create_execution_context()

    # Set input tensors
    inputs[0].host[:] = XTX.astype(np.float32).flatten()
    inputs[1].host[:] = XTY.astype(np.float32).flatten()

    print("Running inference...")
    trt_outputs = common.do_inference(
        context,
        engine=engine,
        bindings=bindings,
        inputs=inputs,
        outputs=outputs,
        stream=stream,
    )
    output = trt_outputs[0].copy()
    common.free_buffers(inputs, outputs, stream)
    print("Inference complete")
    return output


def main():
    print("=== LinalgSolve Plugin Test ===\n")

    # Test with 540p resolution (non-square H!=W)
    B, H, W, Q_plus_1, C = 2, 540, 960, 3, 3
    print(f"Test config: B={B}, H={H}, W={W}, Q+1={Q_plus_1}, C={C}")

    # Generate simple test data (make matrices positive definite)
    np.random.seed(42)

    # Create symmetric positive definite matrices for XTX
    batch_size = B * H * W
    Q_sq = Q_plus_1 * Q_plus_1
    XTX = np.zeros((B, H, W, Q_sq), dtype=np.float32)
    for i in range(batch_size):
        b, h, w = i // (H*W), (i // W) % H, i % W
        M = np.random.rand(Q_plus_1, Q_plus_1).astype(np.float32)
        SPD = M @ M.T + np.eye(Q_plus_1, dtype=np.float32) * 0.1  # Make SPD
        XTX[b, h, w, :] = SPD.flatten()

    XTY = np.random.rand(B, H, W, Q_plus_1 * C).astype(np.float32) * 0.1

    print(f"\nXTX shape: {XTX.shape}")
    print(f"XTY shape: {XTY.shape}")
    print(f"XTX sample:\n{XTX[0,0,0,:9].reshape(3,3)}")

    # Run reference implementation
    print("\n=== Reference Implementation ===")
    res_ref = linalg_reference_impl(XTX, XTY, eps=0.01)
    print(f"Reference output shape: {res_ref.shape}")
    print(f"Reference output sample: {res_ref[0,0,0,:3]}")

    # Build TensorRT engine and run plugin
    try:
        engine = make_trt_network_and_engine(B, H, W, Q_plus_1, C)
        res_plugin = custom_plugin_impl(XTX, XTY, engine)

        # Reshape and compare
        res_plugin = res_plugin.reshape(res_ref.shape)
        print(f"\nPlugin output shape: {res_plugin.shape}")
        print(f"Plugin output sample: {res_plugin[0,0,0,:3]}")

        # Check results
        max_diff = np.max(np.abs(res_ref - res_plugin))
        mean_diff = np.mean(np.abs(res_ref - res_plugin))
        print(f"\nMax difference: {max_diff}")
        print(f"Mean difference: {mean_diff}")

        if max_diff > 1e-3:
            print(f"FAILED! Max difference {max_diff} exceeds threshold")
            print(f"\nDifference map:\n{np.abs(res_ref - res_plugin)[0,0,0,:]}")
            return False
        else:
            print(f"PASSED!")
            return True
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)