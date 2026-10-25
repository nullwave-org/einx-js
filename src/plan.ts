import { axesOf, type Expression, type Pattern } from "./expression.js";

export type AxisLengths = Readonly<Record<string, number>>;
export type UnaryOperation = "id" | "sum" | "mean";
type Operation = UnaryOperation | "dot";

export type UnaryPlan = {
    readonly inputShape: number[];
    readonly reductionAxes: number[];
    readonly permutation: number[];
    readonly outputShape: number[];
};

export type DotPlan = {
    readonly leftShape: number[];
    readonly rightShape: number[];
    readonly subscripts: string;
    readonly outputShape: number[];
};

function product(values: readonly number[]): number {
    const result = values.reduce((total, value) => total * value, 1);

    if (!Number.isSafeInteger(result)) {
        throw new RangeError("Axis size product exceeds the safe integer range");
    }

    return result;
}

function resolveSizes(
    inputs: readonly Pattern[],
    shapes: readonly (readonly number[])[],
    lengths: AxisLengths,
): Map<string, number> {
    const names = new Set(inputs.flatMap(axesOf).map((axis) => axis.name));
    const sizes = new Map<string, number>();

    function define(name: string, size: number): void {
        if (!Number.isSafeInteger(size) || size < 0) {
            throw new RangeError(`Invalid length for axis ${name}: ${size}`);
        }

        if (sizes.has(name) && sizes.get(name) !== size) {
            throw new RangeError(`Conflicting lengths for axis ${name}`);
        }

        sizes.set(name, size);
    }

    for (const [name, size] of Object.entries(lengths)) {
        if (!names.has(name)) {
            throw new RangeError(`Unknown axis length ${name}`);
        }

        define(name, size);
    }

    let pending = inputs.flatMap((input, operand) => {
        const shape = shapes[operand];

        if (input.length !== shape.length) {
            throw new RangeError(`Operand ${operand} rank does not match its axis expression`);
        }

        return input.map((dimension, index) => ({
            axes: dimension.axes,
            size: shape[index],
        }));
    });

    while (pending.length > 0) {
        const remaining: typeof pending = [];

        for (const constraint of pending) {
            const unknown = constraint.axes.filter((axis) => !sizes.has(axis.name));
            const known = product(constraint.axes.map((axis) => sizes.get(axis.name) ?? 1));

            if (unknown.length === 0) {
                if (known !== constraint.size) {
                    throw new RangeError("Grouped axis lengths do not match the tensor shape");
                }
            } else if (unknown.length === 1 && known !== 0) {
                define(unknown[0].name, constraint.size / known);
            } else {
                if (known === 0 && constraint.size !== 0) {
                    throw new RangeError("Zero axis length does not match the tensor shape");
                }

                remaining.push(constraint);
            }
        }

        if (remaining.length === pending.length) {
            throw new RangeError("Ambiguous grouped axes; provide explicit axis lengths");
        }

        pending = remaining;
    }

    return sizes;
}

function implicitOutput(input: Pattern): Pattern {
    const output = [];

    for (const dimension of input) {
        const axes = dimension.axes.filter((axis) => !axis.marked);

        // A parenthesized reduction retains a physical unit dimension.
        if (axes.length > 0 || dimension.grouped) {
            output.push({ ...dimension, axes });
        }
    }

    return output;
}

function resolve(
    operation: Operation,
    expression: Expression,
    shapes: readonly (readonly number[])[],
    lengths: AxisLengths,
) {
    const { inputs } = expression;
    const count = operation === "dot" ? 2 : 1;

    if (inputs.length !== count || shapes.length !== count) {
        throw new SyntaxError(`${operation} requires ${count} operand expressions and shapes`);
    }

    if (operation === "dot" && expression.output === undefined) {
        throw new SyntaxError("dot requires an explicit output expression");
    }

    const output = expression.output ?? implicitOutput(inputs[0]);
    const inputAxes = inputs.flatMap(axesOf);
    const outputNames = axesOf(output).map((axis) => axis.name);
    const outputSet = new Set(outputNames);
    const hasMarkers = inputAxes.some((axis) => axis.marked);

    if (operation === "id" && hasMarkers) {
        throw new SyntaxError("id does not accept marked axes");
    }

    if (axesOf(output).some((axis) => axis.marked)) {
        throw new SyntaxError("Output axes must be unmarked");
    }

    const roles = new Map<string, boolean>();

    for (const axis of inputAxes) {
        const reduced = hasMarkers ? axis.marked : operation !== "id" && !outputSet.has(axis.name);

        if (roles.has(axis.name) && roles.get(axis.name) !== reduced) {
            throw new SyntaxError(`Axis ${axis.name} must be marked consistently`);
        }

        roles.set(axis.name, reduced);
    }

    const kept = [...roles].filter(([, reduced]) => !reduced).map(([name]) => name);

    if (outputNames.length !== kept.length || kept.some((name) => !outputSet.has(name))) {
        throw new RangeError("Output must contain every unmarked axis exactly once");
    }

    const reduced = new Set([...roles].filter(([, reduced]) => reduced).map(([name]) => name));

    if (operation === "dot") {
        const operandNames = inputs.map((input) => new Set(axesOf(input).map((axis) => axis.name)));

        for (const name of reduced) {
            if (operandNames.some((names) => !names.has(name))) {
                throw new RangeError("Contracted axes must occur in both operands");
            }
        }
    }

    const sizes = resolveSizes(inputs, shapes, lengths);

    function sizeOf(name: string): number {
        const size = sizes.get(name);

        if (size === undefined) {
            throw new RangeError(`Cannot resolve length for axis ${name}`);
        }

        return size;
    }

    const inputShapes = inputs.map((input) => axesOf(input).map((axis) => sizeOf(axis.name)));
    const outputShape = output.map((dimension) =>
        product(dimension.axes.map((axis) => sizeOf(axis.name))),
    );
    return { inputShapes, outputShape, outputNames, reduced };
}

/** Resolve all shape and axis decisions before executing a unary operation. */
export function planUnary(
    operation: UnaryOperation,
    expression: Expression,
    shape: readonly number[],
    lengths: AxisLengths = {},
): UnaryPlan {
    const resolved = resolve(operation, expression, [shape], lengths);
    const inputNames = axesOf(expression.inputs[0]).map((axis) => axis.name);
    const reductionAxes: number[] = [];
    const kept: string[] = [];

    for (const [index, name] of inputNames.entries()) {
        if (resolved.reduced.has(name)) {
            reductionAxes.push(index);
        } else {
            kept.push(name);
        }
    }

    return {
        inputShape: resolved.inputShapes[0],
        reductionAxes,
        permutation: resolved.outputNames.map((name) => kept.indexOf(name)),
        outputShape: resolved.outputShape,
    };
}

/** Translate named binary contraction axes to JAX's einsum notation. */
export function planDot(
    expression: Expression,
    leftShape: readonly number[],
    rightShape: readonly number[],
    lengths: AxisLengths = {},
): DotPlan {
    const resolved = resolve("dot", expression, [leftShape, rightShape], lengths);
    const names = new Set(expression.inputs.flatMap(axesOf).map((axis) => axis.name));
    const alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

    if (names.size > alphabet.length) {
        throw new RangeError("dot supports at most 52 distinct named axes");
    }

    const symbols = new Map([...names].map((name, index) => [name, alphabet[index]]));
    const operands = expression.inputs.map((input) =>
        axesOf(input)
            .map((axis) => symbols.get(axis.name))
            .join(""),
    );
    const output = resolved.outputNames.map((name) => symbols.get(name)).join("");
    return {
        leftShape: resolved.inputShapes[0],
        rightShape: resolved.inputShapes[1],
        subscripts: `${operands.join(",")}->${output}`,
        outputShape: resolved.outputShape,
    };
}
