"""
Graph Builder API for Neural Network Custom Dialect

Provides a PyTorch-style deferred execution API:
  g = Graph()
  x = g.variable([4])
  y = g.variable([4])
  z = g.add(x, y)
  mlir_text = g.get_mlir(z, "add_func")
"""
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class TensorValue:
    """Represents a tensor value in the computation graph"""
    id: int
    shape: List[int]

    def __repr__(self):
        return f"%{self.id}"

@dataclass
class Operation:
    """Represents an operation in the graph"""
    op_type: str  # 'variable', 'add', 'mul', 'matmul', 'relu', 'softmax'
    operands: List[TensorValue]
    result: TensorValue
    attrs: Dict[str, Any]  # Additional attributes

class Graph:
    """
    Computation graph builder for neural network operations.

    This class tracks operations symbolically and generates MLIR text
    when get_mlir() is called.
    """

    def __init__(self):
        self.next_id = 0
        self.operations: List[Operation] = []
        self.variables: List[TensorValue] = []

    def _new_value(self, shape: List[int]) -> TensorValue:
        """Create a new SSA value"""
        val = TensorValue(self.next_id, shape)
        self.next_id += 1
        return val

    def variable(self, shape: List[int]) -> TensorValue:
        """
        Create a variable (input placeholder).

        Args:
            shape: Shape of the tensor, e.g., [4] for 1D, [2, 3] for 2D

        Returns:
            TensorValue representing this variable
        """
        result = self._new_value(shape)
        op = Operation('variable', [], result, {})
        self.operations.append(op)
        self.variables.append(result)
        return result

    def add(self, lhs: TensorValue, rhs: TensorValue) -> TensorValue:
        """Element-wise addition"""
        if lhs.shape != rhs.shape:
            raise ValueError(f"Shape mismatch: {lhs.shape} vs {rhs.shape}")
        result = self._new_value(lhs.shape)
        op = Operation('add', [lhs, rhs], result, {})
        self.operations.append(op)
        return result

    def mul(self, lhs: TensorValue, rhs: TensorValue) -> TensorValue:
        """Element-wise multiplication"""
        if lhs.shape != rhs.shape:
            raise ValueError(f"Shape mismatch: {lhs.shape} vs {rhs.shape}")
        result = self._new_value(lhs.shape)
        op = Operation('mul', [lhs, rhs], result, {})
        self.operations.append(op)
        return result

    def matmul(self, lhs: TensorValue, rhs: TensorValue) -> TensorValue:
        """
        Matrix multiplication.

        For lhs: [M, K] and rhs: [K, N]
        Returns: [M, N]
        """
        if len(lhs.shape) != 2 or len(rhs.shape) != 2:
            raise ValueError(f"matmul requires 2D tensors, got {lhs.shape} and {rhs.shape}")

        M, K1 = lhs.shape
        K2, N = rhs.shape

        if K1 != K2:
            raise ValueError(f"matmul dimension mismatch: {lhs.shape} @ {rhs.shape}")

        result = self._new_value([M, N])
        op = Operation('matmul', [lhs, rhs], result, {})
        self.operations.append(op)
        return result

    def relu(self, input: TensorValue) -> TensorValue:
        """ReLU activation: max(0, input)"""
        result = self._new_value(input.shape)
        op = Operation('relu', [input], result, {})
        self.operations.append(op)
        return result

    def softmax(self, input: TensorValue) -> TensorValue:
        """Softmax activation"""
        result = self._new_value(input.shape)
        op = Operation('softmax', [input], result, {})
        self.operations.append(op)
        return result

    def _format_shape(self, shape: List[int]) -> str:
        """Format shape as MLIR tensor type"""
        shape_str = 'x'.join(map(str, shape))
        return f"tensor<{shape_str}xf32>"

    def get_mlir(self, output: TensorValue, func_name: str = "main") -> str:
        """
        Generate MLIR text representation using the nn dialect.

        Args:
            output: The final output value
            func_name: Name for the function

        Returns:
            MLIR text as string
        """
        lines = []
        lines.append("module {")

        # Build function signature
        input_types = [self._format_shape(v.shape) for v in self.variables]
        output_type = self._format_shape(output.shape)

        func_sig = f"  func.func @{func_name}("
        for i, (var, type_str) in enumerate(zip(self.variables, input_types)):
            if i > 0:
                func_sig += ", "
            func_sig += f"{var} : {type_str}"
        func_sig += f") -> {output_type} {{"
        lines.append(func_sig)

        # Generate operations (skip variables, they're function arguments)
        for op in self.operations:
            if op.op_type == 'variable':
                continue

            indent = "    "
            if op.op_type == 'add':
                line = f"{indent}{op.result} = nn.add {op.operands[0]}, {op.operands[1]} : {self._format_shape(op.result.shape)}"
            elif op.op_type == 'mul':
                line = f"{indent}{op.result} = nn.mul {op.operands[0]}, {op.operands[1]} : {self._format_shape(op.result.shape)}"
            elif op.op_type == 'matmul':
                lhs_type = self._format_shape(op.operands[0].shape)
                rhs_type = self._format_shape(op.operands[1].shape)
                out_type = self._format_shape(op.result.shape)
                line = f"{indent}{op.result} = nn.matmul {op.operands[0]}, {op.operands[1]} : {lhs_type}, {rhs_type} -> {out_type}"
            elif op.op_type == 'relu':
                line = f"{indent}{op.result} = nn.relu {op.operands[0]} : {self._format_shape(op.result.shape)}"
            elif op.op_type == 'softmax':
                line = f"{indent}{op.result} = nn.softmax {op.operands[0]} : {self._format_shape(op.result.shape)}"
            else:
                raise ValueError(f"Unknown operation type: {op.op_type}")

            lines.append(line)

        # Return statement
        lines.append(f"    return {output} : {output_type}")
        lines.append("  }")
        lines.append("}")

        return '\n'.join(lines)