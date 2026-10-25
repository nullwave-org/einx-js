import assert from "node:assert/strict";
import { numpy as np } from "@jax-js/jax";

export function vector(values: readonly number[]): np.Array {
    return np.array(Array.from(values), { dtype: np.float32 });
}

export function trace(actual: np.Array, expected: readonly number[]): void {
    const values = actual.dataSync();
    assert.equal(values.length, expected.length);

    for (let index = 0; index < values.length; index++) {
        const tolerance =
            0.000002 + 0.0002 * Math.max(Math.abs(values[index]), Math.abs(expected[index]));
        assert.ok(
            Number.isFinite(values[index]) &&
                Math.abs(values[index] - expected[index]) <= tolerance,
            `${values[index]} != ${expected[index]}`,
        );
    }
}
