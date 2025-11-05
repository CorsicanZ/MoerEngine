"""
PyTorch Custom Ops for TensorRT Plugins (Inference Only)

This module defines custom PyTorch operations for TensorRT plugins:
1. LinalgSolvePlugin: Fast linear system solving with cuBLAS
2. GaussianBlurPlugin: Optimized separable Gaussian blur

Each operation:
1. In PyTorch mode: Calls native PyTorch implementation (for validation)
2. In ONNX export: Generates a TensorRT plugin node

No backward() is needed since these are inference-only.
"""

import torch
import torch.nn.functional as F


class LinalgSolvePluginOp(torch.autograd.Function):
    """
    Custom autograd Function for LinalgSolvePlugin

    This replaces the entire _solve_linear_system function with a single
    TensorRT plugin node during ONNX export.

    Input Layout:
        XTX: [B, (Q+1)², H_ds, W_ds] - X^T*X gram matrices
        XTY: [B, (Q+1)*C, H_ds, W_ds] - X^T*Y cross-correlation

    Output Layout:
        A: [B, (Q+1)*C, H_ds, W_ds] - regression coefficients

    The plugin internally performs:
        1. Reshape XTX to [B*H*W, Q+1, Q+1]
        2. Reshape XTY to [B*H*W, Q+1, C]
        3. Add regularization: XTX_reg = XTX + eps*I
        4. Solve: A = XTX_reg^(-1) * XTY using cuBLAS
        5. Reshape back and return
    """

    @staticmethod
    def forward(ctx, XTX, XTY, q_plus_one, epsilon, eta):
        """
        Forward pass - used for PyTorch inference validation

        Args:
            ctx: Context for backward (unused in inference)
            XTX: [B, (Q+1)², H_ds, W_ds] gram matrices
            XTY: [B, (Q+1)*C, H_ds, W_ds] cross-correlation
            q_plus_one: Q+1, where Q is the number of guide channels
            epsilon: Regularization parameter (Tikhonov)
            eta: Additional parameter (reserved for future use)

        Returns:
            A: [B, (Q+1)*C, H_ds, W_ds] regression coefficients
        """
        B, _, H_ds, W_ds = XTX.shape
        Q_bias = q_plus_one
        C = XTY.shape[1] // Q_bias

        # Reshape XTX: [B, Q²,  H, W] -> [B, H, W, Q, Q]
        XTX_mat = XTX.view(B, Q_bias, Q_bias, H_ds, W_ds).permute(0, 3, 4, 1, 2)

        # Reshape XTY: [B, Q*C, H, W] -> [B, H, W, Q, C]
        XTY_mat = XTY.view(B, Q_bias, C, H_ds, W_ds).permute(0, 3, 4, 1, 2)

        # Flatten spatial dimensions for batch operations
        XTX_batch = XTX_mat.reshape(-1, Q_bias, Q_bias)  # [B*H*W, Q, Q]
        XTY_batch = XTY_mat.reshape(-1, Q_bias, C)       # [B*H*W, Q, C]

        # Add Tikhonov regularization
        eye = torch.eye(Q_bias, device=XTX.device, dtype=XTX.dtype).unsqueeze(0)
        XTX_reg = XTX_batch + epsilon * eye

        # Solve linear system: A = (X^T*X + εI)^(-1) * X^T*Y
        A_batch = torch.linalg.solve(XTX_reg, XTY_batch)

        # Reshape back to spatial layout
        A = A_batch.reshape(B, H_ds, W_ds, Q_bias, C).permute(0, 3, 4, 1, 2)  # [B, Q, C, H, W]

        # Flatten to match expected output layout
        A_flat = A.reshape(B, Q_bias * C, H_ds, W_ds)

        return A_flat

    @staticmethod
    def symbolic(g, XTX, XTY, q_plus_one, epsilon, eta):
        """
        ONNX symbolic function - called during ONNX export

        This generates a TensorRT plugin node in the ONNX graph.

        Args:
            g: ONNX graph builder
            XTX: Input tensor (symbolic)
            XTY: Input tensor (symbolic)
            q_plus_one: Attribute value
            epsilon: Attribute value
            eta: Attribute value

        Returns:
            ONNX node representing the LinalgSolvePlugin
        """
        return g.op(
            "LinalgSolvePlugin",  # Plugin name (must match LINALG_SOLVE_PLUGIN_NAME)
            XTX, XTY,             # Input tensors
            # Attributes (follow ONNX naming convention)
            q_plus_one_i=q_plus_one,  # _i suffix for int32
            epsilon_f=epsilon,         # _f suffix for float32
            eta_f=eta,                 # _f suffix for float32
            outputs=1                  # Number of outputs
        )


def linalg_solve_plugin(XTX, XTY, q_plus_one, epsilon, eta=1e-3):
    """
    Convenience function for LinalgSolvePlugin operation

    This function can be used as a drop-in replacement for _solve_linear_system.

    Usage in training/inference:
        A = linalg_solve_plugin(XTX, XTY, Q+1, eps, eta)

    During ONNX export, this will automatically be converted to a TensorRT plugin node.

    Args:
        XTX: [B, (Q+1)², H_ds, W_ds] gram matrices
        XTY: [B, (Q+1)*C, H_ds, W_ds] cross-correlation
        q_plus_one: Q+1 (kernel size + 1)
        epsilon: Regularization parameter
        eta: Additional parameter (default: 1e-3)

    Returns:
        A: [B, (Q+1)*C, H_ds, W_ds] regression coefficients
    """
    return LinalgSolvePluginOp.apply(XTX, XTY, q_plus_one, epsilon, eta)


# ============================================================================
# GaussianBlurPlugin Custom Op
# ============================================================================

class GaussianBlurPluginOp(torch.autograd.Function):
    """
    Custom autograd Function for GaussianBlurPlugin

    This replaces the separable Gaussian blur with a single
    TensorRT plugin node during ONNX export.

    Input Layout:
        input: [B, C, H, W] - input tensor (NCHW format)

    Output Layout:
        output: [B, C, H, W] - blurred tensor (same shape)

    The plugin performs optimized separable Gaussian blur using:
        1. Horizontal blur pass (with zero padding)
        2. Vertical blur pass (with zero padding)
        3. Shared memory optimization for efficiency

    Note: Uses zero padding to match flnr.py's _separable_gaussian_blur implementation
          and F.conv2d default behavior.
    """

    @staticmethod
    def forward(ctx, input, sigma):
        """
        Forward pass - used for PyTorch inference validation

        Args:
            ctx: Context for backward (unused in inference)
            input: [B, C, H, W] input tensor
            sigma: Gaussian sigma parameter

        Returns:
            output: [B, C, H, W] blurred tensor
        """
        # Simple separable Gaussian blur using PyTorch
        # This matches flnr.py's _separable_gaussian_blur implementation
        # Uses zero padding to match F.conv2d default behavior and CUDA implementation

        kernel_radius = 2  # FLNR uses kernel_size=5, so radius=2
        kernel_size = 2 * kernel_radius + 1

        # Create 1D Gaussian kernel - match flnr.py's calculation
        x = torch.arange(-kernel_radius, kernel_radius + 1, dtype=input.dtype, device=input.device)
        gaussian_kernel = torch.exp(-x**2 / (2 * sigma**2))
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

        # Expand kernel for conv2d: [C, 1, 1, kernel_size] for horizontal
        kernel_h = gaussian_kernel.view(1, 1, 1, kernel_size).expand(input.shape[1], 1, 1, kernel_size)
        kernel_v = gaussian_kernel.view(1, 1, kernel_size, 1).expand(input.shape[1], 1, kernel_size, 1)

        # Apply horizontal blur with zero padding (F.conv2d default behavior)
        h_blur = F.conv2d(input, kernel_h, padding=(kernel_radius, 0), groups=input.shape[1])

        # Apply vertical blur with zero padding (F.conv2d default behavior)
        output = F.conv2d(h_blur, kernel_v, padding=(0, kernel_radius), groups=input.shape[1])

        return output

    @staticmethod
    def symbolic(g, input, sigma):
        """
        ONNX symbolic function - called during ONNX export

        This generates a TensorRT plugin node in the ONNX graph.

        Args:
            g: ONNX graph builder
            input: Input tensor (symbolic)
            sigma: Gaussian sigma parameter

        Returns:
            ONNX node representing the GaussianBlurPlugin
        """
        return g.op(
            "GaussianBlurPlugin",  # Plugin name (must match GAUSSIAN_BLUR_PLUGIN_NAME)
            input,                 # Input tensor
            # Attributes (follow ONNX naming convention)
            sigma_f=sigma,         # _f suffix for float32
            outputs=1              # Number of outputs
        )


def gaussian_blur_plugin(input, sigma):
    """
    Convenience function for GaussianBlurPlugin operation

    This function can be used as a drop-in replacement for separable Gaussian blur.

    Usage in training/inference:
        output = gaussian_blur_plugin(input, sigma)

    During ONNX export, this will automatically be converted to a TensorRT plugin node.

    Args:
        input: [B, C, H, W] input tensor
        sigma: Gaussian sigma parameter

    Returns:
        output: [B, C, H, W] blurred tensor
    """
    return GaussianBlurPluginOp.apply(input, sigma)
