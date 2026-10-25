import { parseExpression } from "./expression.js";
import { numpy as np, type Tensor } from "./jax.js";
import { withTemps } from "./ownership.js";
import { type AxisLengths, planDot, planUnary, type UnaryOperation } from "./plan.js";

function unary(
    operation: UnaryOperation,
    description: string,
    tensor: Tensor,
    lengths: AxisLengths,
): Tensor {
    return withTemps([tensor], (own) => {
        const expression = parseExpression(description);
        const plan = planUnary(operation, expression, tensor.shape, lengths);
        let result = own(tensor.ref.reshape(plan.inputShape));

        if (plan.reductionAxes.length > 0) {
            const reduce = operation === "sum" ? np.sum : np.mean;
            result = own(reduce(result.ref, plan.reductionAxes));
        }

        if (plan.permutation.some((index, position) => index !== position)) {
            result = own(result.ref.transpose(plan.permutation));
        }

        return result.ref.reshape(plan.outputShape);
    });
}

/** Consume a tensor and rearrange its named axes. */
export function id(description: string, tensor: Tensor, lengths: AxisLengths = {}): Tensor {
    return unary("id", description, tensor, lengths);
}

/** Consume a tensor and sum marked axes, or axes omitted from the output. */
export function sum(description: string, tensor: Tensor, lengths: AxisLengths = {}): Tensor {
    return unary("sum", description, tensor, lengths);
}

/** Consume a tensor and average marked axes, or axes omitted from the output. */
export function mean(description: string, tensor: Tensor, lengths: AxisLengths = {}): Tensor {
    return unary("mean", description, tensor, lengths);
}

/** Consume two tensors and contract named axes omitted from the output. */
export function dot(
    description: string,
    left: Tensor,
    right: Tensor,
    lengths: AxisLengths = {},
): Tensor {
    return withTemps([left, right], (own) => {
        const expression = parseExpression(description);
        const plan = planDot(expression, left.shape, right.shape, lengths);
        const leftExpanded = own(left.ref.reshape(plan.leftShape));
        const rightExpanded = own(right.ref.reshape(plan.rightShape));
        const result = own(np.einsum(plan.subscripts, leftExpanded.ref, rightExpanded.ref));
        return result.ref.reshape(plan.outputShape);
    });
}
