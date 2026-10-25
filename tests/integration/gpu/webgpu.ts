import { defaultDevice, getWebGPUDevice, init, jit, numpy as np } from "@jax-js/jax";
import * as einx from "einx-js";

export async function initialize() {
    const devices = await init("webgpu");

    if (!devices.includes("webgpu")) {
        throw new Error("Hardware WebGPU is required; no GPU device initialized");
    }

    defaultDevice("webgpu");

    const info = Reflect.get(getWebGPUDevice(), "adapterInfo") as
        | {
              vendor: string;
              architecture: string;
              device: string;
              description: string;
              isFallbackAdapter: boolean;
          }
        | undefined;

    if (info?.isFallbackAdapter !== false) {
        throw new Error("Hardware WebGPU is required; fallback or unidentified adapter");
    }

    const { vendor, architecture, device, description, isFallbackAdapter } = info;

    if (
        /swiftshader|llvmpipe|lavapipe|software/i.test(`${vendor} ${architecture} ${description}`)
    ) {
        throw new Error("Hardware WebGPU is required; software renderer detected");
    }

    return { vendor, architecture, device, description, isFallbackAdapter };
}

async function read(value: np.Array) {
    if (value.device !== "webgpu") {
        value.dispose();
        throw new Error("GPU check produced a result on another device");
    }

    return await value.data();
}

function close(actual: number, expected: number, relative = 0.0002, absolute = 0.000002): void {
    const tolerance = absolute + relative * Math.max(Math.abs(actual), Math.abs(expected));

    if (!Number.isFinite(actual) || Math.abs(actual - expected) > tolerance) {
        throw new Error(`${actual} != ${expected}; tolerance ${tolerance}`);
    }
}

async function trace(actual: np.Array, expected: readonly number[]): Promise<void> {
    const values = await read(actual);

    if (values.length !== expected.length) {
        throw new Error(`Result length ${values.length} != ${expected.length}`);
    }

    for (let index = 0; index < values.length; index++) {
        close(values[index], expected[index]);
    }
}

export const checks: Record<string, () => Promise<void>> = {};

checks["named-axis contraction, rearrangement and reduction under JIT"] = async () => {
    const operation = jit((input: np.Array) => {
        const transposed = einx.id("row column -> column row", input.ref);
        const product = einx.dot("row [inner], [inner] column -> row column", input, transposed);
        return einx.sum("row [column]", product);
    });
    const input = np.array(
        [
            [1, 2],
            [3, 4],
        ],
        { dtype: np.float32 },
    );

    try {
        await trace(operation(input.ref), [16, 36]);

        if (input.refCount !== 1) {
            throw new Error("Named-axis operations retained the input");
        }
    } finally {
        input.dispose();
        operation.dispose();
    }
};
