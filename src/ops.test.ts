/* Upstream-derived test cases only:
 * MIT License
 *
 * Copyright (c) 2023- Florian Fervers
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { defaultDevice, init, numpy as np } from "@jax-js/jax";
import { trace, vector } from "../tests/helpers.ts";
import * as einx from "./ops.ts";

await init("wasm");
defaultDevice("wasm");

test("named axes permute, flatten, split and preserve scalar and unit dimensions", () => {
    const values = Array.from({ length: 12 }, (_, index) => index);
    const reordered = einx.id(
        "batch time channel -> time (batch channel)",
        vector(values).reshape([2, 2, 3]),
    );
    assert.deepEqual(reordered.shape, [2, 6]);
    trace(reordered, [0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11]);
    trace(
        einx.id("(time channel) -> channel time", vector([1, 2, 3, 4, 5, 6]), { channel: 2 }),
        [1, 3, 5, 2, 4, 6],
    );
    const unit = einx.id(" -> ()", np.array(7));
    assert.deepEqual(unit.shape, [1]);
    trace(einx.id("() -> ", unit), [7]);
    trace(einx.id("axis", vector([1, 2])), [1, 2]);
});

test("grouped axes reject ambiguous zero factorizations and accept explicit lengths", () => {
    assert.throws(
        () => einx.id("(row column) -> row column", np.zeros([0]), { row: 0 }),
        /Ambiguous/,
    );
    const output = einx.id("(row column) -> row column", np.zeros([0]), { row: 0, column: 3 });
    assert.deepEqual(output.shape, [0, 3]);
    trace(output, []);
});

/** Selected cases from einx/test/test_shapes.py, test_shape_id/reduce (5d01266, MIT). */
test("test_shape_id: upstream flattening, whitespace and unit-axis cases", () => {
    for (const [description, shape, parameters, expected] of [
        ["a b c -> (a b) c 1", [10, 20, 1], {}, [200, 1, 1]],
        ["a b c -> (a b) c 1 1 1", [10, 20, 1], {}, [200, 1, 1, 1, 1]],
        ["a b c", [10, 20, 1], {}, [10, 20, 1]],
        [" ( a   b  ) ->   b   a   ", [200], { a: 10 }, [20, 10]],
        ["(a) (b) -> b a", [20, 10], {}, [10, 20]],
        ["(a (b (c d))) -> a b c d", [120], { a: 2, b: 3, c: 4 }, [2, 3, 4, 5]],
        ["1 1 ->", [1, 1], {}, []],
    ] as const) {
        const output = einx.id(description, np.zeros([...shape]), parameters);

        try {
            assert.deepEqual(output.shape, expected);
        } finally {
            output.dispose();
        }
    }
});

test("test_shape_reduce: implicit markers and flattened reductions preserve unit axes", () => {
    for (const description of ["a b -> 1 a", "a [b] -> a 1", "a ([b])"]) {
        const output = einx.mean(description, np.ones([2, 3]));
        assert.deepEqual(output.shape, description === "a b -> 1 a" ? [1, 2] : [2, 1]);
        trace(output, [1, 1]);
    }
});

test("bracketed reductions support groups, output order and empty dimensions", () => {
    const input = vector([1, 2, 3, 4, 5, 6]).reshape([2, 3]);

    try {
        trace(einx.sum("batch [time]", input.ref), [6, 15]);
        trace(einx.mean("[batch time]", input.ref), [3.5]);
        trace(einx.sum("batch time -> time batch", input.ref), [1, 4, 2, 5, 3, 6]);
        trace(einx.sum("(batch [time])", input.ref.reshape([6]), { time: 3 }), [6, 15]);
        trace(einx.sum("batch [time]", np.zeros([2, 0])), [0, 0]);
        trace(einx.id("(batch channel) -> batch channel", np.zeros([0]), { channel: 2 }), []);
        assert.equal(input.refCount, 1);
    } finally {
        input.dispose();
    }
});

test("binary contractions agree with independent matrix and batch references", () => {
    const left = vector([1, 2, 3, 4, 5, 6]).reshape([2, 3]);
    const right = vector([1, 2, 3, 4, 5, 6]).reshape([3, 2]);

    try {
        trace(
            einx.dot("row [inner], [inner] column -> column row", left.ref, right.ref),
            [22, 49, 28, 64],
        );
        trace(
            einx.dot("row [inner], [inner] column -> (row column)", left.ref, right.ref),
            [22, 28, 49, 64],
        );
        assert.equal(left.refCount, 1);
        assert.equal(right.refCount, 1);
    } finally {
        left.dispose();
        right.dispose();
    }
});

test("axis constraints propagate between operands before resolving groups", () => {
    trace(
        einx.dot(
            "(row [inner]), [inner] column -> row column",
            vector([1, 2, 3, 4, 5, 6]),
            vector([1, 2, 3, 4, 5, 6]).reshape([3, 2]),
        ),
        [22, 28, 49, 64],
    );
    trace(
        einx.dot(
            "batch [inner], batch [inner] -> batch",
            vector([1, 2, 3, 4]).reshape([2, 2]),
            vector([2, 3, 4, 5]).reshape([2, 2]),
        ),
        [8, 32],
    );
    trace(einx.dot("left, right -> left right", vector([1, 2]), vector([3, 4])), [3, 4, 6, 8]);
});

/** Selected dot cases from einx/test/test_values.py, test_values (5d01266, MIT). */
test("test_values: upstream unmarked dot notation matches marked contractions", () => {
    trace(einx.dot("a b, b -> a", np.ones([10, 10]), np.ones([10])), Array(10).fill(10));
    trace(einx.dot("a, b -> a b", np.ones([2]), np.ones([3])), Array(6).fill(1));
    trace(einx.dot("a, -> a", vector([2, 3]), np.array(2)), [4, 6]);
    trace(einx.sum("a b -> a", vector([1, 2, 3, 4, 5, 6]).reshape([2, 3])), [6, 15]);
    trace(einx.mean("a b -> a", vector([1, 2, 3, 4, 5, 6]).reshape([2, 3])), [2, 5]);
});

test("test_shape_dot: contracted axes occur in both operands", () => {
    assert.throws(
        () => einx.dot("a b, c d -> a c", np.ones([2, 3]), np.ones([2, 3])),
        /both operands/,
    );
    assert.throws(
        () => einx.dot("a [b], a [b] -> 1", np.ones([2, 3]), np.ones([2, 3])),
        /unmarked axis/,
    );
});

test("invalid expressions and shapes release every supplied owned reference", () => {
    for (const expression of [
        "axis ... -> axis",
        "axis axis -> axis",
        "[axis",
        "axis -> [axis]",
        "axis -> axis + other",
        "axis -> axis -> axis",
        "[]",
        "axis)",
    ]) {
        const input = vector([1, 2]);
        assert.throws(() => einx.id(expression, input.ref), SyntaxError);
        assert.equal(input.refCount, 1);
        input.dispose();
    }

    for (const [expression, lengths] of [
        ["(row column) -> row column", {}],
        ["(row column) -> row column", { column: 4 }],
        ["axis -> axis", { unused: 2 }],
        ["axis -> axis", { axis: -1 }],
        ["axis -> axis", { axis: Number.POSITIVE_INFINITY }],
        ["row column -> row column", {}],
        ["axis -> other", {}],
    ] as const) {
        const input = vector([1, 2, 3, 4, 5, 6]);
        assert.throws(() => einx.id(expression, input.ref, lengths), RangeError);
        assert.equal(input.refCount, 1);
        input.dispose();
    }

    const left = np.ones([2, 3]);
    const right = np.ones([4, 2]);

    try {
        assert.throws(
            () => einx.dot("row [inner], [inner] column -> row column", left.ref, right.ref),
            /lengths/,
        );
        assert.throws(
            () => einx.dot("row [inner], inner column -> row column", left.ref, right.ref),
            /consistently/,
        );
        assert.throws(
            () => einx.dot("row [inner], [inner] column", left.ref, right.ref),
            /explicit output/,
        );
        assert.equal(left.refCount, 1);
        assert.equal(right.refCount, 1);
    } finally {
        left.dispose();
        right.dispose();
    }
});
