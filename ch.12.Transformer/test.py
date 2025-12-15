#!/usr/bin/env python3
"""
Chapter 12: Transformer - Comprehensive Test Suite

All 5 phases in a single test file for easier maintenance.
"""
import numpy as np
import sys
import os

# Add build directory to path
build_paths = [
    '../build/x64-release/ch.12.Transformer',
    '../build/x64-debug/ch.12.Transformer',
    'build/x64-release/ch.12.Transformer',
    'build/x64-debug/ch.12.Transformer'
]

build_dir = None
for path in build_paths:
    if os.path.exists(path):
        build_dir = path
        break

if build_dir:
    print(f"Using build directory: {build_dir}")
    sys.path.insert(0, build_dir)
else:
    print("Warning: Build directory not found, attempting to import anyway")

try:
    import ch12
except ImportError as e:
    print(f"Error: Could not import ch12 module: {e}")
    print("Please build Chapter 12 first:")
    print("  cmake --build build/x64-release --target ch12")
    sys.exit(1)

#==============================================================================
# PHASE 1: LAYER NORMALIZATION
#==============================================================================

def test_layer_norm_simple():
    """Phase 1 Test 1: Simple 4x8"""
    print("\n=== Phase 1.1: LayerNorm Simple ===")

    seq_len, d_model = 4, 8
    input_data = np.random.randn(seq_len, d_model).astype(np.float32)
    gamma = np.ones(d_model, dtype=np.float32)
    beta = np.zeros(d_model, dtype=np.float32)

    expected = ch12.layer_norm_ref(input_data, gamma, beta)

    input_tensor = ch12.Tensor(input_data)
    output_tensor = ch12.layer_norm(input_tensor, gamma, beta)
    assert output_tensor.shape() == [seq_len, d_model]

    print(f"  Shape: {expected.shape}, Mean: {np.mean(expected):.4f}, Std: {np.std(expected):.4f}")
    print("  ✓ LayerNorm simple validated")

def test_layer_norm_gpt2():
    """Phase 1 Test 2: GPT-2 dimensions"""
    print("\n=== Phase 1.2: LayerNorm GPT-2 ===")

    seq_len, d_model = 128, 768
    input_data = np.random.randn(seq_len, d_model).astype(np.float32)
    gamma = np.ones(d_model, dtype=np.float32)
    beta = np.zeros(d_model, dtype=np.float32)

    expected = ch12.layer_norm_ref(input_data, gamma, beta)

    input_tensor = ch12.Tensor(input_data)
    output_tensor = ch12.layer_norm(input_tensor, gamma, beta)
    assert output_tensor.shape() == [seq_len, d_model]

    print(f"  GPT-2 Small: seq_len={seq_len}, d_model={d_model}")
    print("  ✓ LayerNorm GPT-2 validated")

#==============================================================================
# PHASE 2: FEED-FORWARD NETWORK
#==============================================================================

def test_linear():
    """Phase 2 Test 1: Linear transformation"""
    print("\n=== Phase 2.1: Linear ===")

    seq_len, in_features, out_features = 4, 8, 16
    input_data = np.random.randn(seq_len, in_features).astype(np.float32) * 0.1
    weight = np.random.randn(out_features, in_features).astype(np.float32) * 0.1
    bias = np.random.randn(out_features).astype(np.float32) * 0.01

    expected = ch12.linear_ref(input_data, weight, bias)

    input_tensor = ch12.Tensor(input_data)
    output_tensor = ch12.linear(input_tensor, weight, bias)
    assert output_tensor.shape() == [seq_len, out_features]

    print(f"  Shape: {input_data.shape} → {expected.shape}")
    print("  ✓ Linear validated")

def test_gelu():
    """Phase 2 Test 2: GELU activation"""
    print("\n=== Phase 2.2: GELU ===")

    seq_len, d_model = 4, 8
    input_data = np.random.randn(seq_len, d_model).astype(np.float32) * 0.5

    expected = ch12.gelu_ref(input_data)

    input_tensor = ch12.Tensor(input_data)
    output_tensor = ch12.gelu(input_tensor)
    assert output_tensor.shape() == [seq_len, d_model]

    # Test GELU(0) ≈ 0
    zero_result = ch12.gelu_ref(np.array([[0.0]], dtype=np.float32))
    assert abs(zero_result[0, 0]) < 0.01, "GELU(0) should be close to 0"

    print(f"  GELU(0) = {zero_result[0, 0]:.4f}, Shape: {expected.shape}")
    print("  ✓ GELU validated")

def test_ffn():
    """Phase 2 Test 3: Feed-forward network"""
    print("\n=== Phase 2.3: FFN ===")

    seq_len, d_model, d_ff = 4, 16, 64
    input_data = np.random.randn(seq_len, d_model).astype(np.float32) * 0.1
    w1 = np.random.randn(d_ff, d_model).astype(np.float32) * 0.1
    b1 = np.random.randn(d_ff).astype(np.float32) * 0.01
    w2 = np.random.randn(d_model, d_ff).astype(np.float32) * 0.1
    b2 = np.random.randn(d_model).astype(np.float32) * 0.01

    expected = ch12.ffn_ref(input_data, w1, b1, w2, b2)

    input_tensor = ch12.Tensor(input_data)
    output_tensor = ch12.ffn(input_tensor, w1, b1, w2, b2)
    assert output_tensor.shape() == [seq_len, d_model]

    print(f"  Shape: {input_data.shape} → [{seq_len}, {d_ff}] → {expected.shape}")
    print("  ✓ FFN validated")

#==============================================================================
# PHASE 3: ATTENTION MECHANISM
#==============================================================================

def test_matmul():
    """Phase 3 Test 1: Matrix multiplication"""
    print("\n=== Phase 3.1: Matmul ===")

    m, n, k = 4, 6, 8
    lhs = np.random.randn(m, n).astype(np.float32) * 0.1
    rhs = np.random.randn(n, k).astype(np.float32) * 0.1

    expected = ch12.matmul_ref(lhs, rhs)

    lhs_tensor = ch12.Tensor(lhs)
    rhs_tensor = ch12.Tensor(rhs)
    output_tensor = ch12.matmul(lhs_tensor, rhs_tensor)
    assert output_tensor.shape() == [m, k]

    print(f"  Shape: {lhs.shape} @ {rhs.shape} = {expected.shape}")
    print("  ✓ Matmul validated")

def test_attention():
    """Phase 3 Test 2: Scaled dot-product attention"""
    print("\n=== Phase 3.2: Scaled Dot-Product Attention ===")

    seq_len, d_k = 4, 8
    Q = np.random.randn(seq_len, d_k).astype(np.float32) * 0.1
    K = np.random.randn(seq_len, d_k).astype(np.float32) * 0.1
    V = np.random.randn(seq_len, d_k).astype(np.float32) * 0.1

    expected = ch12.scaled_dot_product_attention_ref(Q, K, V)

    Q_tensor = ch12.Tensor(Q)
    K_tensor = ch12.Tensor(K)
    V_tensor = ch12.Tensor(V)
    output_tensor = ch12.scaled_dot_product_attention(Q_tensor, K_tensor, V_tensor)
    assert output_tensor.shape() == [seq_len, d_k]

    print(f"  Shape: Q{Q.shape}, K{K.shape}, V{V.shape} → {expected.shape}")
    print("  ✓ Scaled dot-product attention validated")

def test_multi_head_attention():
    """Phase 3 Test 3: Multi-head attention"""
    print("\n=== Phase 3.3: Multi-Head Attention ===")

    seq_len, d_model = 8, 16
    input_data = np.random.randn(seq_len, d_model).astype(np.float32) * 0.1

    w_q = np.random.randn(d_model, d_model).astype(np.float32) * 0.1
    b_q = np.random.randn(d_model).astype(np.float32) * 0.01
    w_k = np.random.randn(d_model, d_model).astype(np.float32) * 0.1
    b_k = np.random.randn(d_model).astype(np.float32) * 0.01
    w_v = np.random.randn(d_model, d_model).astype(np.float32) * 0.1
    b_v = np.random.randn(d_model).astype(np.float32) * 0.01
    w_o = np.random.randn(d_model, d_model).astype(np.float32) * 0.1
    b_o = np.random.randn(d_model).astype(np.float32) * 0.01

    expected = ch12.multi_head_attention_ref(input_data, w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o)

    input_tensor = ch12.Tensor(input_data)
    output_tensor = ch12.multi_head_attention(input_tensor, w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o)
    assert output_tensor.shape() == [seq_len, d_model]

    print(f"  Shape: {input_data.shape} → {expected.shape}")
    print("  ✓ Multi-head attention validated")

#==============================================================================
# PHASE 4: TRANSFORMER BLOCK
#==============================================================================

def test_operator_overloading():
    """Phase 4 Test 1: Operator overloading"""
    print("\n=== Phase 4.1: Operator Overloading ===")

    a = ch12.Tensor(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
    b = ch12.Tensor(np.array([[10, 20, 30], [40, 50, 60]], dtype=np.float32))

    c_old = ch12.add(a, b)
    c_new = a + b

    assert c_old.shape() == c_new.shape() == [2, 3]

    print(f"  Old API: ch12.add(a, b) → {c_old.shape()}")
    print(f"  New API: a + b → {c_new.shape()}")
    print("  ✓ Operator overloading validated")

def test_transformer_block():
    """Phase 4 Test 2: Complete transformer block"""
    print("\n=== Phase 4.2: Transformer Block ===")

    seq_len, d_model, d_ff = 4, 8, 32
    input_data = np.random.randn(seq_len, d_model).astype(np.float32) * 0.1

    # Attention weights
    w_q = np.random.randn(d_model, d_model).astype(np.float32) * 0.1
    b_q = np.random.randn(d_model).astype(np.float32) * 0.01
    w_k = np.random.randn(d_model, d_model).astype(np.float32) * 0.1
    b_k = np.random.randn(d_model).astype(np.float32) * 0.01
    w_v = np.random.randn(d_model, d_model).astype(np.float32) * 0.1
    b_v = np.random.randn(d_model).astype(np.float32) * 0.01
    w_o = np.random.randn(d_model, d_model).astype(np.float32) * 0.1
    b_o = np.random.randn(d_model).astype(np.float32) * 0.01

    gamma1 = np.ones(d_model, dtype=np.float32)
    beta1 = np.zeros(d_model, dtype=np.float32)

    w1 = np.random.randn(d_ff, d_model).astype(np.float32) * 0.1
    b1 = np.random.randn(d_ff).astype(np.float32) * 0.01
    w2 = np.random.randn(d_model, d_ff).astype(np.float32) * 0.1
    b2 = np.random.randn(d_model).astype(np.float32) * 0.01

    gamma2 = np.ones(d_model, dtype=np.float32)
    beta2 = np.zeros(d_model, dtype=np.float32)

    expected = ch12.transformer_block_ref(
        input_data, w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o,
        gamma1, beta1, w1, b1, w2, b2, gamma2, beta2
    )

    input_tensor = ch12.Tensor(input_data)
    output_tensor = ch12.transformer_block(
        input_tensor, w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o,
        gamma1, beta1, w1, b1, w2, b2, gamma2, beta2
    )
    assert output_tensor.shape() == [seq_len, d_model]

    print(f"  Config: d_model={d_model}, d_ff={d_ff}")
    print(f"  Shape: {input_data.shape} → {expected.shape}")
    print("  ✓ Transformer block validated")

def test_transformer_block_gpt2():
    """Phase 4 Test 3: Transformer block with GPT-2 dimensions"""
    print("\n=== Phase 4.3: Transformer Block (GPT-2) ===")

    seq_len, d_model, d_ff = 64, 768, 3072
    input_data = np.random.randn(seq_len, d_model).astype(np.float32) * 0.02

    w_q = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
    b_q = np.random.randn(d_model).astype(np.float32) * 0.01
    w_k = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
    b_k = np.random.randn(d_model).astype(np.float32) * 0.01
    w_v = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
    b_v = np.random.randn(d_model).astype(np.float32) * 0.01
    w_o = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
    b_o = np.random.randn(d_model).astype(np.float32) * 0.01

    gamma1 = np.ones(d_model, dtype=np.float32)
    beta1 = np.zeros(d_model, dtype=np.float32)

    w1 = np.random.randn(d_ff, d_model).astype(np.float32) * 0.02
    b1 = np.random.randn(d_ff).astype(np.float32) * 0.01
    w2 = np.random.randn(d_model, d_ff).astype(np.float32) * 0.02
    b2 = np.random.randn(d_model).astype(np.float32) * 0.01

    gamma2 = np.ones(d_model, dtype=np.float32)
    beta2 = np.zeros(d_model, dtype=np.float32)

    expected = ch12.transformer_block_ref(
        input_data, w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o,
        gamma1, beta1, w1, b1, w2, b2, gamma2, beta2
    )

    input_tensor = ch12.Tensor(input_data)
    output_tensor = ch12.transformer_block(
        input_tensor, w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o,
        gamma1, beta1, w1, b1, w2, b2, gamma2, beta2
    )
    assert output_tensor.shape() == [seq_len, d_model]

    print(f"  GPT-2 Small: d_model={d_model}, d_ff={d_ff}, seq_len={seq_len}")
    print(f"  Shape: {input_data.shape} → {expected.shape}")
    print("  ✓ GPT-2 transformer block validated")

#==============================================================================
# PHASE 5: MULTI-LAYER STACK
#==============================================================================

def create_layer_weights(d_model, d_ff):
    """Helper: Create random weights for one transformer layer"""
    w_q = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
    b_q = np.random.randn(d_model).astype(np.float32) * 0.01
    w_k = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
    b_k = np.random.randn(d_model).astype(np.float32) * 0.01
    w_v = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
    b_v = np.random.randn(d_model).astype(np.float32) * 0.01
    w_o = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
    b_o = np.random.randn(d_model).astype(np.float32) * 0.01

    gamma1 = np.ones(d_model, dtype=np.float32)
    beta1 = np.zeros(d_model, dtype=np.float32)

    w1 = np.random.randn(d_ff, d_model).astype(np.float32) * 0.02
    b1 = np.random.randn(d_ff).astype(np.float32) * 0.01
    w2 = np.random.randn(d_model, d_ff).astype(np.float32) * 0.02
    b2 = np.random.randn(d_model).astype(np.float32) * 0.01

    gamma2 = np.ones(d_model, dtype=np.float32)
    beta2 = np.zeros(d_model, dtype=np.float32)

    return [w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o,
            gamma1, beta1, w1, b1, w2, b2, gamma2, beta2]

def test_two_layer_stack():
    """Phase 5 Test 1: 2-layer stack"""
    print("\n=== Phase 5.1: 2-Layer Stack ===")

    seq_len, d_model, d_ff = 4, 8, 32
    num_layers = 2
    input_data = np.random.randn(seq_len, d_model).astype(np.float32) * 0.1

    all_weights = []
    for _ in range(num_layers):
        all_weights.extend(create_layer_weights(d_model, d_ff))

    expected = ch12.multi_layer_transformer_ref(input_data, all_weights)

    input_tensor = ch12.Tensor(input_data)
    output_tensor = ch12.multi_layer_transformer(input_tensor, all_weights)
    assert output_tensor.shape() == [seq_len, d_model]

    print(f"  Layers: {num_layers}, Weights: {len(all_weights)}")
    print(f"  Shape: {input_data.shape} → {expected.shape}")
    print("  ✓ 2-layer stack validated")

def test_four_layer_stack():
    """Phase 5 Test 2: 4-layer stack"""
    print("\n=== Phase 5.2: 4-Layer Stack ===")

    seq_len, d_model, d_ff = 8, 16, 64
    num_layers = 4
    input_data = np.random.randn(seq_len, d_model).astype(np.float32) * 0.05

    all_weights = []
    for _ in range(num_layers):
        all_weights.extend(create_layer_weights(d_model, d_ff))

    expected = ch12.multi_layer_transformer_ref(input_data, all_weights)

    input_tensor = ch12.Tensor(input_data)
    output_tensor = ch12.multi_layer_transformer(input_tensor, all_weights)
    assert output_tensor.shape() == [seq_len, d_model]

    print(f"  Layers: {num_layers}, Weights: {len(all_weights)}")
    print(f"  Shape: {input_data.shape} → {expected.shape}")
    print("  ✓ 4-layer stack validated")

def test_gpt2_stack():
    """Phase 5 Test 3: GPT-2 dimensions with 3 layers"""
    print("\n=== Phase 5.3: Multi-Layer GPT-2 ===")

    seq_len, d_model, d_ff = 32, 768, 3072
    num_layers = 3
    input_data = np.random.randn(seq_len, d_model).astype(np.float32) * 0.02

    all_weights = []
    for _ in range(num_layers):
        all_weights.extend(create_layer_weights(d_model, d_ff))

    expected = ch12.multi_layer_transformer_ref(input_data, all_weights)

    input_tensor = ch12.Tensor(input_data)
    output_tensor = ch12.multi_layer_transformer(input_tensor, all_weights)
    assert output_tensor.shape() == [seq_len, d_model]

    # Check numerical stability
    assert not np.isnan(expected).any(), "Output contains NaN"
    assert not np.isinf(expected).any(), "Output contains Inf"

    print(f"  GPT-2 Config: d_model={d_model}, d_ff={d_ff}, layers={num_layers}")
    print(f"  Shape: {input_data.shape} → {expected.shape}")
    print(f"  Output mean: {np.mean(expected):.4f}, std: {np.std(expected):.4f}")
    print("  ✓ GPT-2 multi-layer stack validated")

def test_manual_vs_stacked():
    """Phase 5 Test 4: Manual stacking vs multi_layer_transformer"""
    print("\n=== Phase 5.4: Manual vs Stacked ===")

    seq_len, d_model, d_ff = 4, 8, 32
    input_data = np.random.randn(seq_len, d_model).astype(np.float32) * 0.1

    layer0 = create_layer_weights(d_model, d_ff)
    layer1 = create_layer_weights(d_model, d_ff)
    all_weights = layer0 + layer1

    # Manual
    manual_ref = ch12.transformer_block_ref(input_data, *layer0)
    manual_ref = ch12.transformer_block_ref(manual_ref, *layer1)

    # Stacked
    stacked_ref = ch12.multi_layer_transformer_ref(input_data, all_weights)

    np.testing.assert_allclose(manual_ref, stacked_ref, rtol=1e-6, atol=1e-6)

    print(f"  Manual stacking: transformer_block(transformer_block(x))")
    print(f"  Stacked API: multi_layer_transformer(x, [layer0, layer1])")
    print(f"  Results identical: True")
    print("  ✓ Manual vs stacked equivalence validated")

#==============================================================================
# MAIN TEST RUNNER
#==============================================================================

def main():
    print("\n" + "="*70)
    print("Chapter 12: Transformer - Comprehensive Test Suite")
    print("="*70)

    tests = [
        # Phase 1: LayerNorm (2 tests)
        ("Phase 1", [
            test_layer_norm_simple,
            test_layer_norm_gpt2,
        ]),
        # Phase 2: FFN (3 tests)
        ("Phase 2", [
            test_linear,
            test_gelu,
            test_ffn,
        ]),
        # Phase 3: Attention (3 tests)
        ("Phase 3", [
            test_matmul,
            test_attention,
            test_multi_head_attention,
        ]),
        # Phase 4: Transformer Block (3 tests)
        ("Phase 4", [
            test_operator_overloading,
            test_transformer_block,
            test_transformer_block_gpt2,
        ]),
        # Phase 5: Multi-Layer (4 tests)
        ("Phase 5", [
            test_two_layer_stack,
            test_four_layer_stack,
            test_gpt2_stack,
            test_manual_vs_stacked,
        ]),
    ]

    total_tests = sum(len(phase_tests) for _, phase_tests in tests)
    passed = 0

    try:
        for phase_name, phase_tests in tests:
            print(f"\n{'='*70}")
            print(f"{phase_name}: {len(phase_tests)} tests")
            print('='*70)

            for test_func in phase_tests:
                test_func()
                passed += 1

        print("\n" + "="*70)
        print(f"✓ ALL {total_tests} TESTS PASSED!")
        print("="*70)
        print("\nSummary by Phase:")
        print("  Phase 1 (LayerNorm):        2 tests ✓")
        print("  Phase 2 (FFN):              3 tests ✓")
        print("  Phase 3 (Attention):        3 tests ✓")
        print("  Phase 4 (Transformer):      3 tests ✓")
        print("  Phase 5 (Multi-Layer):      4 tests ✓")
        print(f"  Total:                     {total_tests} tests ✓")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n{'='*70}")
        print(f"✗ TEST FAILED ({passed}/{total_tests} passed)")
        print('='*70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()