import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import {
    access,
    copyFile,
    mkdir,
    mkdtemp,
    readFile,
    realpath,
    rm,
    writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { delimiter, join, resolve, sep } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";

const execute = promisify(execFile);
const repository = resolve(fileURLToPath(new URL("../../", import.meta.url)));
const manifest = JSON.parse(await readFile(new URL("../../package.json", import.meta.url), "utf8"));
const archive = fileURLToPath(
    new URL(`../../target/package/einx-js-${manifest.version}.tgz`, import.meta.url),
);
const environment: NodeJS.ProcessEnv = { ...process.env, CI: "true" };
delete environment.NODE_PATH;

// Keep external package managers installed under node_modules/.bin.
environment.PATH = environment.PATH?.split(delimiter)
    .filter((entry) => {
        const path = resolve(entry);
        return path !== repository && !path.startsWith(repository + sep);
    })
    .join(delimiter);

for (const manager of ["npm", "pnpm"] as const) {
    test(`${manager}: packed package, shared JAX, peer resolution and dependency failures`, {
        timeout: 240_000,
    }, async (context) => {
        const directory = await mkdtemp(join(tmpdir(), `einx-js-${manager}-consumer-`));
        context.after(() => rm(directory, { recursive: true, force: true }));
        await access(archive);

        async function project(name: string, dependencies: Record<string, string>) {
            const location = join(directory, name);
            await mkdir(location);
            await writeFile(
                join(location, "package.json"),
                JSON.stringify({
                    private: true,
                    type: "module",
                    dependencies: { "einx-js": `file:${archive}`, ...dependencies },
                }),
            );
            await writeFile(
                join(location, "pnpm-workspace.yaml"),
                "autoInstallPeers: true\nstrictPeerDependencies: true\n",
            );
            return location;
        }

        async function run(location: string, command: string, args: string[]) {
            return await execute(command, args, {
                cwd: location,
                env: environment,
                signal: context.signal,
                timeout: 60_000,
                encoding: "utf8",
            });
        }

        async function install(location: string, flags: string[] = []) {
            const args =
                manager === "npm"
                    ? [
                          "install",
                          "--ignore-scripts",
                          "--no-audit",
                          "--no-fund",
                          "--prefer-offline",
                          "--fetch-retries=0",
                          "--fetch-timeout=20000",
                          ...flags,
                      ]
                    : [
                          "install",
                          "--ignore-scripts",
                          "--no-frozen-lockfile",
                          "--prefer-offline",
                          ...flags,
                      ];
            return await run(location, manager, args);
        }

        const consumer = await project("explicit-peer", {
            "@jax-js/jax": manifest.devDependencies["@jax-js/jax"],
            typescript: manifest.devDependencies.typescript,
        });
        await install(consumer, manager === "npm" ? ["--strict-peer-deps"] : []);
        const installed = join(consumer, "node_modules/einx-js");
        assert.ok((await realpath(installed)).startsWith(consumer + sep));

        for (const name of ["dist/index.js", "dist/index.d.ts"]) {
            await access(join(installed, name));
        }

        for (const name of ["src", "tests", "node_modules", "LICENSE"]) {
            await assert.rejects(access(join(installed, name)), { code: "ENOENT" });
        }

        await copyFile(
            new URL("./fixtures/consumer.ts", import.meta.url),
            join(consumer, "consumer.ts"),
        );
        await writeFile(
            join(consumer, "tsconfig.json"),
            JSON.stringify({
                compilerOptions: {
                    target: "ES2022",
                    lib: ["ESNext", "DOM"],
                    module: "NodeNext",
                    strict: true,
                    skipLibCheck: false,
                    types: [],
                    outDir: "build",
                },
                include: ["consumer.ts"],
            }),
        );
        await access(join(consumer, "node_modules/.bin/tsc"));
        const compiler = manager === "npm" ? ["exec", "--", "tsc"] : ["exec", "tsc"];
        await run(consumer, manager, [...compiler, "-p", "tsconfig.json"]);
        const result = await run(consumer, process.execPath, ["build/consumer.js"]);
        assert.match(result.stdout, /einx-js consumer passed/);

        const automatic = await project("automatic-peer", {});
        await install(automatic);
        await run(automatic, process.execPath, [
            "--input-type=module",
            "--eval",
            'const einx = await import("einx-js"); if (typeof einx.dot !== "function") throw new Error("Missing public export");',
        ]);

        const missing = await project("missing-peer", {});
        await writeFile(
            join(missing, "pnpm-workspace.yaml"),
            "autoInstallPeers: false\nstrictPeerDependencies: false\n",
        );
        await install(missing, manager === "npm" ? ["--legacy-peer-deps"] : []);
        await assert.rejects(
            run(missing, process.execPath, [
                "--input-type=module",
                "--eval",
                'await import("einx-js")',
            ]),
            { stderr: /einx-js requires @jax-js\/jax \^0\.1\.27\. Install it with: pnpm add/ },
        );

        const incompatible = await project("incompatible-peer", {
            "@jax-js/jax": "file:./invalid-jax",
        });
        await mkdir(join(incompatible, "invalid-jax"));
        await writeFile(
            join(incompatible, "invalid-jax/package.json"),
            JSON.stringify({
                name: "@jax-js/jax",
                version: "0.0.0",
                type: "module",
                exports: "./index.js",
            }),
        );
        await writeFile(join(incompatible, "invalid-jax/index.js"), "export {};\n");
        await assert.rejects(
            install(incompatible, manager === "npm" ? ["--strict-peer-deps"] : []),
            (error: unknown) => {
                assert.ok(error instanceof Error && "stdout" in error && "stderr" in error);
                assert.match(
                    String(error.stdout) + String(error.stderr),
                    /ERESOLVE|ERR_PNPM_PEER_DEP_ISSUES/,
                );
                return true;
            },
        );
    });
}
