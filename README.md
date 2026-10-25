# einx-js

This library provides named-axis tensor operations for jax-js
similar to [einx](https://github.com/fferflo/einx).

Currently implemented are `id`, `sum`, `mean` and binary `dot`.

### Installation

```sh
pnpm add /path/to/einx-js-0.1.0.tgz @jax-js/jax@^0.1.27
```

### Usage

```ts
import { init, numpy as np } from "@jax-js/jax";
import * as einx from "einx-js";

await init("wasm");
const output = einx.dot("batch feature, feature -> batch", np.ones([2, 3]), np.ones([3]));
console.log(output.dataSync()); // [3, 3]
```

Operations consume input references; pass `tensor.ref` to reuse a tensor.
Supports named axes, groups, brackets, unit dimensions and explicit axis lengths.
Ellipses, diagonals, concatenation, broadcasting and indexing are unsupported.
TypeScript needs `ESNext` library declarations for JAX.
