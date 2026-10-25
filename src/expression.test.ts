import assert from "node:assert/strict";
import test from "node:test";
import { axesOf, parseExpression } from "./expression.ts";

test("parsing preserves groups, markers and source locations without JAX", () => {
    const expression = parseExpression("batch (time [channel]) -> time (batch channel)");
    assert.equal(expression.inputs[0].length, 2);
    assert.equal(expression.inputs[0][1].grouped, true);
    assert.deepEqual(axesOf(expression.inputs[0]), [
        { name: "batch", marked: false },
        { name: "time", marked: false },
        { name: "channel", marked: true },
    ]);
    assert.throws(() => parseExpression("batch ? -> batch"), /token "\?" at column 7/);
    assert.throws(() => parseExpression("batch (time -> batch"), /Missing closing/);
});

test("generated malformed notation consistently reports syntax errors", () => {
    const count = Number(process.env.EINX_TEST_CASES ?? 128);
    const invalid = ["?", "+", "...", "=", "2", "(", "["];

    for (let trial = 0; trial < count; trial++) {
        const name = `axis_${trial}`;
        const suffix = invalid[trial % invalid.length];
        assert.throws(() => parseExpression(`${name} ${suffix} -> ${name}`), SyntaxError);
    }
});
