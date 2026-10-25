import assert from "node:assert/strict";
import test from "node:test";
import { defaultDevice, grad, init, jit, type numpy as np } from "@jax-js/jax";
import { dot, id, mean, sum } from "../../src/index.ts";
import { trace, vector } from "../helpers.ts";

await init("wasm");
defaultDevice("wasm");

function random(seed: number): () => number {
    let state = seed;

    return () => {
        state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
        return state / 4294967296;
    };
}

test("seeded rearrangements, reductions and contractions match independent loops", () => {
    const next = random(0x137a45b9);

    for (let trial = 0; trial < Number(process.env.EINX_TEST_CASES ?? 32); trial++) {
        const rows = 1 + Math.floor(next() * 4);
        const inner = 1 + Math.floor(next() * 4);
        const columns = 1 + Math.floor(next() * 4);
        const leftValues = Array.from(
            { length: rows * inner },
            () => Math.floor(next() * 33 - 16) / 8,
        );
        const rightValues = Array.from(
            { length: inner * columns },
            () => Math.floor(next() * 33 - 16) / 8,
        );
        const left = vector(leftValues).reshape([rows, inner]);
        const right = vector(rightValues).reshape([inner, columns]);
        const reordered: number[] = [];
        const totals: number[] = [];
        const multiplied: number[] = [];

        for (let feature = 0; feature < inner; feature++) {
            for (let row = 0; row < rows; row++) {
                reordered.push(leftValues[row * inner + feature]);
            }
        }

        for (let row = 0; row < rows; row++) {
            let total = 0;

            for (let feature = 0; feature < inner; feature++) {
                total += leftValues[row * inner + feature];
            }

            totals.push(total);

            for (let column = 0; column < columns; column++) {
                let product = 0;

                for (let feature = 0; feature < inner; feature++) {
                    product +=
                        leftValues[row * inner + feature] * rightValues[feature * columns + column];
                }

                multiplied.push(product);
            }
        }

        try {
            trace(
                id("(row feature) -> feature row", left.ref.reshape([rows * inner]), {
                    feature: inner,
                }),
                reordered,
            );
            trace(sum("row feature -> row", left.ref), totals);
            trace(
                mean("row [feature]", left.ref),
                totals.map((total) => total / inner),
            );
            trace(
                dot("row feature, feature column -> (row column)", left.ref, right.ref),
                multiplied,
            );
            assert.equal(left.refCount, 1);
            assert.equal(right.refCount, 1);
        } finally {
            left.dispose();
            right.dispose();
        }
    }
});

test("repeated JIT gradients and failing operations release borrowed references", () => {
    const input = vector([1, 2, 3]);
    const loss = jit(grad((tensor: np.Array) => mean("[feature]", tensor)));

    try {
        for (let trial = 0; trial < Number(process.env.EINX_LIFETIME_CASES ?? 32); trial++) {
            trace(loss(input.ref), [1 / 3, 1 / 3, 1 / 3]);
            assert.throws(() => id("(row feature) -> row feature", input.ref), /Ambiguous/);
            assert.equal(input.refCount, 1);
        }
    } finally {
        input.dispose();
        loss.dispose();
    }
});
