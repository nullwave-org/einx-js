import type { Tensor } from "./jax.js";

type Own = (value: Tensor) => Tensor;

/** Consume inputs and temporary references; the result must own a separate reference. */
export function withTemps<Result>(
    inputs: readonly Tensor[],
    operation: (own: Own) => Result,
): Result {
    const temporaries = [...inputs];

    const own: Own = (value) => {
        temporaries.push(value);
        return value;
    };

    try {
        return operation(own);
    } finally {
        for (const value of temporaries) {
            value.dispose();
        }
    }
}
