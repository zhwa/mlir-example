"""
Lowering patterns: nn dialect → standard MLIR dialects

This module converts high-level nn operations to standard MLIR:
  - nn.add → arith.addf (element-wise)
  - nn.mul → arith.mulf (element-wise)  
  - nn.matmul → linalg.matmul
  - nn.relu → linalg.generic + arith.maximumf
  - nn.softmax → three-pass algorithm (from Chapter 6)
"""
from graph_builder import Graph, TensorValue
from typing import List

class MLIRLowering:
    """Lower nn dialect operations to standard MLIR dialects"""

    def __init__(self):
        self.indent_level = 2  # Start inside function body

    def _indent(self) -> str:
        return "  " * self.indent_level

    def _tensor_to_memref(self, shape: List[int]) -> str:
        """Convert tensor type to memref type"""
        shape_str = 'x'.join(map(str, shape))
        return f"memref<{shape_str}xf32>"

    def _shape_str(self, shape: List[int]) -> str:
        """Format shape as string"""
        return 'x'.join(map(str, shape))

    def lower_add(self, result_id: int, lhs_id: int, rhs_id: int, shape: List[int]) -> List[str]:
        """Lower nn.add to linalg.generic with arith.addf"""
        lines = []
        ind = self._indent()
        shape_str = self._shape_str(shape)
        memref_type = self._tensor_to_memref(shape)

        # For 1D: linalg.generic with parallel iterator
        # For 2D: also parallel iterators
        rank = len(shape)

        if rank == 1:
            indexing = "affine_map<(d0) -> (d0)>"
            iterator = '["parallel"]'
        elif rank == 2:
            indexing = "affine_map<(d0, d1) -> (d0, d1)>"
            iterator = '["parallel", "parallel"]'
        else:
            raise ValueError(f"Unsupported rank: {rank}")

        lines.append(f"{ind}%{result_id} = memref.alloc() : {memref_type}")
        lines.append(f"{ind}linalg.generic {{")
        lines.append(f"{ind}  indexing_maps = [{indexing}, {indexing}, {indexing}],")
        lines.append(f"{ind}  iterator_types = {iterator}")
        lines.append(f"{ind}}} ins(%{lhs_id}, %{rhs_id} : {memref_type}, {memref_type})")
        lines.append(f"{ind}   outs(%{result_id} : {memref_type}) {{")
        lines.append(f"{ind}^bb0(%arg0: f32, %arg1: f32, %arg2: f32):")
        lines.append(f"{ind}  %sum = arith.addf %arg0, %arg1 : f32")
        lines.append(f"{ind}  linalg.yield %sum : f32")
        lines.append(f"{ind}}}")

        return lines

    def lower_mul(self, result_id: int, lhs_id: int, rhs_id: int, shape: List[int]) -> List[str]:
        """Lower nn.mul to linalg.generic with arith.mulf"""
        lines = []
        ind = self._indent()
        memref_type = self._tensor_to_memref(shape)

        rank = len(shape)
        if rank == 1:
            indexing = "affine_map<(d0) -> (d0)>"
            iterator = '["parallel"]'
        elif rank == 2:
            indexing = "affine_map<(d0, d1) -> (d0, d1)>"
            iterator = '["parallel", "parallel"]'
        else:
            raise ValueError(f"Unsupported rank: {rank}")

        lines.append(f"{ind}%{result_id} = memref.alloc() : {memref_type}")
        lines.append(f"{ind}linalg.generic {{")
        lines.append(f"{ind}  indexing_maps = [{indexing}, {indexing}, {indexing}],")
        lines.append(f"{ind}  iterator_types = {iterator}")
        lines.append(f"{ind}}} ins(%{lhs_id}, %{rhs_id} : {memref_type}, {memref_type})")
        lines.append(f"{ind}   outs(%{result_id} : {memref_type}) {{")
        lines.append(f"{ind}^bb0(%arg0: f32, %arg1: f32, %arg2: f32):")
        lines.append(f"{ind}  %prod = arith.mulf %arg0, %arg1 : f32")
        lines.append(f"{ind}  linalg.yield %prod : f32")
        lines.append(f"{ind}}}")

        return lines

    def lower_matmul(self, result_id: int, lhs_id: int, rhs_id: int, 
                     lhs_shape: List[int], rhs_shape: List[int], result_shape: List[int]) -> List[str]:
        """Lower nn.matmul to linalg.matmul"""
        lines = []
        ind = self._indent()

        lhs_type = self._tensor_to_memref(lhs_shape)
        rhs_type = self._tensor_to_memref(rhs_shape)
        result_type = self._tensor_to_memref(result_shape)

        lines.append(f"{ind}%{result_id} = memref.alloc() : {result_type}")
        lines.append(f"{ind}linalg.fill ins(%cst_zero : f32) outs(%{result_id} : {result_type})")
        lines.append(f"{ind}linalg.matmul ins(%{lhs_id}, %{rhs_id} : {lhs_type}, {rhs_type})")
        lines.append(f"{ind}                outs(%{result_id} : {result_type})")

        return lines

    def lower_relu(self, result_id: int, input_id: int, shape: List[int]) -> List[str]:
        """Lower nn.relu to linalg.generic + arith.maximumf"""
        lines = []
        ind = self._indent()
        memref_type = self._tensor_to_memref(shape)

        # For now, use linalg.generic for simplicity
        rank = len(shape)
        if rank == 1:
            indexing = "affine_map<(d0) -> (d0)>"
            iterator = '["parallel"]'
        elif rank == 2:
            indexing = "affine_map<(d0, d1) -> (d0, d1)>"
            iterator = '["parallel", "parallel"]'
        else:
            raise ValueError(f"Unsupported rank: {rank}")

        lines.append(f"{ind}%{result_id} = memref.alloc() : {memref_type}")
        lines.append(f"{ind}linalg.generic {{")
        lines.append(f"{ind}  indexing_maps = [{indexing}, {indexing}],")
        lines.append(f"{ind}  iterator_types = {iterator}")
        lines.append(f"{ind}}} ins(%{input_id} : {memref_type})")
        lines.append(f"{ind}   outs(%{result_id} : {memref_type}) {{")
        lines.append(f"{ind}^bb0(%arg0: f32, %arg1: f32):")
        lines.append(f"{ind}  %relu_val = arith.maximumf %arg0, %cst_zero : f32")
        lines.append(f"{ind}  linalg.yield %relu_val : f32")
        lines.append(f"{ind}}}")

        return lines

    def lower_softmax(self, result_id: int, input_id: int, shape: List[int]) -> List[str]:
        """Lower nn.softmax using three-pass algorithm (from Chapter 6)"""
        lines = []
        ind = self._indent()

        # Only support 1D for now
        if len(shape) != 1:
            raise ValueError("Softmax currently only supports 1D tensors")

        size = shape[0]
        memref_type = f"memref<{size}xf32>"
        scalar_memref_type = "memref<f32>"

        # Allocate output
        lines.append(f"{ind}%{result_id} = memref.alloc() : {memref_type}")

        # 1. Find Max
        lines.append(f"{ind}%max_alloc = memref.alloc() : {scalar_memref_type}")
        lines.append(f"{ind}%neg_inf = arith.constant -3.40282347E+38 : f32")
        lines.append(f"{ind}memref.store %neg_inf, %max_alloc[] : {scalar_memref_type}")
        
        lines.append(f"{ind}linalg.generic {{")
        lines.append(f"{ind}  indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],")
        lines.append(f"{ind}  iterator_types = [\"reduction\"]")
        lines.append(f"{ind}}} ins(%{input_id} : {memref_type}) outs(%max_alloc : {scalar_memref_type}) {{")
        lines.append(f"{ind}^bb0(%in: f32, %acc: f32):")
        lines.append(f"{ind}  %max = arith.maximumf %in, %acc : f32")
        lines.append(f"{ind}  linalg.yield %max : f32")
        lines.append(f"{ind}}}")

        # 2. Compute Exp(x - max)
        lines.append(f"{ind}%exp_alloc = memref.alloc() : {memref_type}")
        
        lines.append(f"{ind}linalg.generic {{")
        lines.append(f"{ind}  indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>],")
        lines.append(f"{ind}  iterator_types = [\"parallel\"]")
        lines.append(f"{ind}}} ins(%{input_id}, %max_alloc : {memref_type}, {scalar_memref_type})")
        lines.append(f"{ind}  outs(%exp_alloc : {memref_type}) {{")
        lines.append(f"{ind}^bb0(%in: f32, %max_val: f32, %dest: f32):")
        lines.append(f"{ind}  %diff = arith.subf %in, %max_val : f32")
        lines.append(f"{ind}  %exp = math.exp %diff : f32")
        lines.append(f"{ind}  linalg.yield %exp : f32")
        lines.append(f"{ind}}}")

        # 3. Sum Exponentials
        lines.append(f"{ind}%sum_alloc = memref.alloc() : {scalar_memref_type}")
        lines.append(f"{ind}%c0 = arith.constant 0.0 : f32")
        lines.append(f"{ind}memref.store %c0, %sum_alloc[] : {scalar_memref_type}")
        
        lines.append(f"{ind}linalg.generic {{")
        lines.append(f"{ind}  indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],")
        lines.append(f"{ind}  iterator_types = [\"reduction\"]")
        lines.append(f"{ind}}} ins(%exp_alloc : {memref_type}) outs(%sum_alloc : {scalar_memref_type}) {{")
        lines.append(f"{ind}^bb0(%in: f32, %acc: f32):")
        lines.append(f"{ind}  %sum = arith.addf %in, %acc : f32")
        lines.append(f"{ind}  linalg.yield %sum : f32")
        lines.append(f"{ind}}}")

        # 4. Normalize
        lines.append(f"{ind}linalg.generic {{")
        lines.append(f"{ind}  indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>],")
        lines.append(f"{ind}  iterator_types = [\"parallel\"]")
        lines.append(f"{ind}}} ins(%exp_alloc, %sum_alloc : {memref_type}, {scalar_memref_type})")
        lines.append(f"{ind}  outs(%{result_id} : {memref_type}) {{")
        lines.append(f"{ind}^bb0(%exp_val: f32, %sum_val: f32, %dest: f32):")
        lines.append(f"{ind}  %norm = arith.divf %exp_val, %sum_val : f32")
        lines.append(f"{ind}  linalg.yield %norm : f32")
        lines.append(f"{ind}}}")

        return lines

    def lower_graph(self, graph: Graph, output: TensorValue, func_name: str = "main") -> str:
        """
        Lower entire computation graph to standard MLIR.

        Returns:
            MLIR text with only standard dialects (func, memref, linalg, arith, scf, math)
        """
        lines = []
        lines.append("module {")

        # Build function signature - output as parameter, not return value
        input_types = [self._tensor_to_memref(v.shape) for v in graph.variables]
        output_type = self._tensor_to_memref(output.shape)

        func_sig = f"  func.func @{func_name}("
        for i, (var, type_str) in enumerate(zip(graph.variables, input_types)):
            func_sig += f"%{var.id} : {type_str}, "
        func_sig += f"%out : {output_type}) {{"
        lines.append(func_sig)

        # Constants
        lines.append(f"    %cst_zero = arith.constant 0.000000e+00 : f32")

        # Lower each operation
        for op in graph.operations:
            if op.op_type == 'variable':
                continue  # Skip, already function arguments

            if op.op_type == 'add':
                op_lines = self.lower_add(op.result.id, op.operands[0].id, op.operands[1].id, op.result.shape)
            elif op.op_type == 'mul':
                op_lines = self.lower_mul(op.result.id, op.operands[0].id, op.operands[1].id, op.result.shape)
            elif op.op_type == 'matmul':
                op_lines = self.lower_matmul(
                    op.result.id, op.operands[0].id, op.operands[1].id,
                    op.operands[0].shape, op.operands[1].shape, op.result.shape
                )
            elif op.op_type == 'relu':
                op_lines = self.lower_relu(op.result.id, op.operands[0].id, op.result.shape)
            elif op.op_type == 'softmax':
                op_lines = self.lower_softmax(op.result.id, op.operands[0].id, op.result.shape)
            else:
                raise ValueError(f"Unknown operation type: {op.op_type}")

            lines.extend(op_lines)

        # Copy result to output parameter and return
        output_type = self._tensor_to_memref(output.shape)
        lines.append(f"    memref.copy %{output.id}, %out : {output_type} to {output_type}")
        lines.append(f"    return")
        lines.append("  }")
        lines.append("}")
        return '\n'.join(lines)