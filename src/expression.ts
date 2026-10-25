export type Axis = { readonly name: string; readonly marked: boolean };
export type Dimension = { readonly axes: readonly Axis[]; readonly grouped: boolean };
export type Pattern = readonly Dimension[];
export type Expression = { readonly inputs: readonly Pattern[]; readonly output?: Pattern };

export function axesOf(pattern: Pattern): Axis[] {
    return pattern.flatMap((dimension) => dimension.axes);
}

function parsePattern(source: string): Pattern {
    const tokens = [...source.matchAll(/[A-Za-z_]\w*|[^\s]/g)];
    let position = 0;

    function syntax(message: string): SyntaxError {
        const token = tokens[Math.max(0, position - 1)];
        return new SyntaxError(
            `${message} at column ${(token?.index ?? source.length) + 1} in ${JSON.stringify(source)}`,
        );
    }

    function sequence(closing?: string, marked = false): Dimension[] {
        const dimensions: Dimension[] = [];

        while (position < tokens.length) {
            const token = tokens[position++][0];

            if (token === closing) {
                return dimensions;
            }

            if (token === "[") {
                if (marked) {
                    throw syntax("Nested axis brackets are unsupported");
                }

                const contents = sequence("]", true);

                if (axesOf(contents).length === 0) {
                    throw syntax("Axis brackets must contain named axes");
                }

                dimensions.push(...contents);
            } else if (token === "(") {
                const contents = sequence(")", marked);
                dimensions.push({ axes: axesOf(contents), grouped: true });
            } else if (token === "1") {
                dimensions.push({ axes: [], grouped: true });
            } else if (/^[A-Za-z_]\w*$/.test(token)) {
                dimensions.push({ axes: [{ name: token, marked }], grouped: false });
            } else {
                throw syntax(`Unsupported axis token ${JSON.stringify(token)}`);
            }
        }

        if (closing !== undefined) {
            throw syntax(`Missing closing ${closing}`);
        }

        return dimensions;
    }

    const pattern = sequence();
    const names = axesOf(pattern).map((axis) => axis.name);

    if (new Set(names).size !== names.length) {
        throw syntax("Repeated axes within one operand are unsupported");
    }

    return pattern;
}

/** Parse notation without inspecting tensors or resolving operation semantics. */
export function parseExpression(source: string): Expression {
    const sides = source.split("->");

    if (sides.length > 2) {
        throw new SyntaxError("Expected at most one output arrow");
    }

    const inputs = sides[0].split(",").map(parsePattern);
    const output = sides.length === 2 ? parsePattern(sides[1]) : undefined;
    return { inputs, output };
}
