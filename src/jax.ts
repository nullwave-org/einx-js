import type * as Jax from "@jax-js/jax";

export type Tensor = Jax.numpy.Array;

async function loadJax(): Promise<typeof Jax> {
    try {
        return await import("@jax-js/jax");
    } catch (error) {
        if (
            error instanceof Error &&
            "code" in error &&
            error.code === "ERR_MODULE_NOT_FOUND" &&
            error.message.includes("'@jax-js/jax'")
        ) {
            throw new Error(
                "einx-js requires @jax-js/jax ^0.1.27. Install it with: pnpm add @jax-js/jax@^0.1.27",
                { cause: error },
            );
        }

        throw error;
    }
}

export const { numpy } = await loadJax();
