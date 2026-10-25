import assert from "node:assert/strict";
import test from "node:test";
import { parseExpression } from "./expression.ts";
import { type AxisLengths, planDot, planUnary } from "./plan.ts";

test("rearrangement and reduction plans contain only resolved execution metadata", () => {
    assert.deepEqual(
        planUnary("id", parseExpression("(row column) -> column row 1"), [6], { column: 3 }),
        { inputShape: [2, 3], reductionAxes: [], permutation: [1, 0], outputShape: [3, 2, 1] },
    );
    assert.deepEqual(
        planUnary("sum", parseExpression("batch [time] channel -> channel batch"), [2, 3, 4]),
        { inputShape: [2, 3, 4], reductionAxes: [1], permutation: [1, 0], outputShape: [4, 2] },
    );
    assert.deepEqual(planUnary("mean", parseExpression("batch ([time])"), [2, 3]), {
        inputShape: [2, 3],
        reductionAxes: [1],
        permutation: [0],
        outputShape: [2, 1],
    });
});

test("planning reuses parsed expressions without changing explicit or implicit markers", () => {
    const expression = parseExpression("row column -> row");
    const original = structuredClone(expression);
    const first = planUnary("sum", expression, [2, 3]);
    const second = planUnary("mean", expression, [4, 5]);
    assert.deepEqual(first.reductionAxes, [1]);
    assert.deepEqual(second.outputShape, [4]);
    assert.deepEqual(expression, original);
    assert.throws(() => planUnary("id", expression, [2, 3]), /unmarked axis/);
});

test("contraction plans resolve grouped axes across operands and preserve output order", () => {
    const plan = planDot(parseExpression("(row inner), inner column -> column row"), [6], [3, 4]);
    assert.deepEqual(plan, {
        leftShape: [2, 3],
        rightShape: [3, 4],
        subscripts: "ab,bc->ca",
        outputShape: [4, 2],
    });
    assert.deepEqual(planDot(parseExpression("row, -> row"), [2], []).subscripts, "a,->a");
});

test("invalid constraints fail before tensor execution", () => {
    const expression = parseExpression("(row column) -> row column");

    const invalid: AxisLengths[] = [
        { column: 4 },
        { column: -1 },
        { column: Infinity },
        { unused: 2 },
    ];

    for (const lengths of invalid) {
        assert.throws(() => planUnary("id", expression, [6], lengths), RangeError);
    }

    assert.throws(() => planUnary("id", expression, [6]), /Ambiguous/);
    assert.throws(() => planUnary("sum", parseExpression("row column"), [6]), /rank/);
    assert.throws(() => planUnary("sum", parseExpression("row, column"), [6]), /requires 1/);
    assert.throws(() => planUnary("id", parseExpression("[row]"), [6]), /marked axes/);
    assert.throws(
        () => planDot(parseExpression("row inner, other column -> row column"), [2, 3], [3, 4]),
        /both operands/,
    );
    assert.throws(
        () => planDot(parseExpression("row inner, inner column -> row column"), [2, 3], [4, 2]),
        /lengths/,
    );
});
