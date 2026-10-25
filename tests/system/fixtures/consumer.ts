import { defaultDevice, grad, init, jit, numpy as np, vmap } from "@jax-js/jax";
import { dot, id, mean, sum } from "einx-js";

await init("wasm");
defaultDevice("wasm");

function check(condition: boolean, message: string): void {
    if (!condition) {
        throw new Error(message);
    }
}

function values(actual: np.Array, expected: readonly number[]): void {
    check(actual instanceof np.Array, "Result must use the consumer's JAX Array class");
    const observed = actual.dataSync();
    check(observed.length === expected.length, "Result length differs");

    for (const [index, value] of observed.entries()) {
        check(
            Math.abs(value - expected[index]) <= 0.0002 * Math.max(1, Math.abs(expected[index])),
            "Result differs",
        );
    }
}

const input = np.array(
    [
        [1, 2, 3],
        [4, 5, 6],
    ],
    { dtype: np.float32 },
);
const weights = np.array([2, 3, 4], { dtype: np.float32 });
const loss = (tensor: np.Array) =>
    mean("[row]", dot("row feature, feature -> row", tensor, weights.ref));
const compiled = jit(loss);
const derivative = jit(grad(loss));
const batched = jit(vmap((tensor: np.Array) => sum("[feature]", tensor)));

try {
    values(dot("row feature, feature -> row", input.ref, weights.ref), [20, 47]);
    values(id("row feature -> feature row", input.ref), [1, 4, 2, 5, 3, 6]);
    values(sum("row [feature]", input.ref), [6, 15]);
    values(compiled(input.ref), [33.5]);
    values(derivative(input.ref), [1, 1.5, 2, 1, 1.5, 2]);
    values(batched(input.ref), [6, 15]);
    check(input.refCount === 1, `Operations retained input references: ${input.refCount}`);
} finally {
    compiled.dispose();
    derivative.dispose();
    batched.dispose();
    const references = weights.refCount;
    input.dispose();
    weights.dispose();
    check(references === 1, `Compiled functions retained constants after disposal: ${references}`);
}

console.log("einx-js consumer passed");
