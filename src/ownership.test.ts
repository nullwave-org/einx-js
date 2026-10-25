import assert from "node:assert/strict";
import test from "node:test";
import { defaultDevice, init, numpy as np } from "@jax-js/jax";
import { trace } from "../tests/helpers.ts";
import { withTemps } from "./ownership.ts";

await init("wasm");
defaultDevice("wasm");

test("temporary ownership releases inputs while returning a separately owned result", () => {
    const input = np.ones([3]);

    try {
        const result = withTemps([input.ref], (own) => {
            const total = own(np.sum(input.ref));
            return total.ref;
        });

        assert.equal(input.refCount, 1);
        trace(result, [3]);
    } finally {
        input.dispose();
    }
});

test("temporary ownership releases all tracked references when the callback throws", () => {
    const input = np.ones([3]);
    const temporary = np.ones([2]);

    try {
        assert.throws(
            () =>
                withTemps([input.ref], (own) => {
                    own(temporary.ref);
                    throw new Error("operation failed");
                }),
            /operation failed/,
        );

        assert.equal(input.refCount, 1);
        assert.equal(temporary.refCount, 1);
    } finally {
        input.dispose();
        temporary.dispose();
    }
});
