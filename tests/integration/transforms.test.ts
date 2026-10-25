import assert from "node:assert/strict";
import test from "node:test";
import { defaultDevice, grad, init, jit, type numpy as np, vmap } from "@jax-js/jax";
import * as einx from "../../src/index.ts";
import { trace, vector } from "../helpers.ts";

await init("wasm");
defaultDevice("wasm");

test("JIT, gradients and vmap compose with native array operations", () => {
    const weights = vector([1, 2, 3]);
    const operation = (input: np.Array) =>
        einx.mean("[batch]", einx.dot("batch [feature], [feature] -> batch", input, weights.ref));
    const compiled = jit(operation);
    const derivative = jit(grad(operation));
    const batched = jit(vmap((input: np.Array) => einx.sum("[feature]", input)));
    const rearranged = jit((input: np.Array) =>
        einx.id("batch (time channel) -> time batch channel", input, { channel: 2 }),
    );
    const input = vector([1, 2, 3, 4, 5, 6]).reshape([2, 3]);

    try {
        trace(compiled(input.ref), [23]);
        trace(derivative(input.ref), [0.5, 1, 1.5, 0.5, 1, 1.5]);
        trace(batched(input.ref), [6, 15]);
        trace(rearranged(input.ref.reshape([1, 6])), [1, 2, 3, 4, 5, 6]);
        assert.equal(input.refCount, 1);
    } finally {
        compiled.dispose();
        derivative.dispose();
        batched.dispose();
        rearranged.dispose();
        input.dispose();
        weights.dispose();
    }
});
