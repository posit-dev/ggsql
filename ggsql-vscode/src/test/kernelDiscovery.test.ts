import * as assert from 'assert';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import * as vscode from 'vscode';
import type * as positron from '@posit-dev/positron';
import {
	GgsqlRuntimeManager,
	discoverKernelPaths,
	generateMetadata,
	isKernelAccessible,
	probeKernel,
	resolveConfiguredPath,
	resolveKernelStrategy,
	selectKernelCandidates,
	type KernelCandidate,
	type KernelProbe,
} from '../manager';

const EXTENSION_ID = 'ggsql.ggsql';

// Directories created by the helpers below, removed in suiteTeardown.
const tempDirs: string[] = [];

function tempDir(): string {
	const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ggsql-kernel-'));
	tempDirs.push(dir);
	return dir;
}

const binaryName = process.platform === 'win32' ? 'ggsql-jupyter.exe' : 'ggsql-jupyter';

function realExtension(): vscode.Extension<unknown> {
	const extension = vscode.extensions.getExtension(EXTENSION_ID);
	assert.ok(extension, `extension ${EXTENSION_ID} not found`);
	return extension;
}

/** Write an executable stand-in for the kernel into `dir`. */
function writeStubKernel(dir: string, mode = 0o755): string {
	fs.mkdirSync(dir, { recursive: true });
	const kernelPath = path.join(dir, binaryName);
	fs.writeFileSync(kernelPath, '#!/bin/sh\nexit 0\n');
	fs.chmodSync(kernelPath, mode);
	return kernelPath;
}

/**
 * Build a directory that looks like an installed platform VSIX: a kernel at
 * bundled/bin/, and the icon generateMetadata reads from the extension folder.
 */
function extensionDirWithBundle(mode = 0o755): { extensionPath: string; kernelPath: string } {
	const extensionPath = tempDir();
	fs.mkdirSync(path.join(extensionPath, 'resources'), { recursive: true });
	fs.copyFileSync(
		path.join(realExtension().extensionPath, 'resources', 'ggsql-icon.svg'),
		path.join(extensionPath, 'resources', 'ggsql-icon.svg'),
	);
	return { extensionPath, kernelPath: writeStubKernel(path.join(extensionPath, 'bundled', 'bin'), mode) };
}

function contextFor(extensionPath: string): vscode.ExtensionContext {
	return { extensionPath } as vscode.ExtensionContext;
}

/**
 * True when a native installer has put a kernel on this machine. Those paths are
 * hard-coded absolutes that no environment variable can redirect, so a test
 * needing "no host kernel anywhere" has to stand aside on such a machine. CI
 * never has one, which is where the regression matters.
 */
function systemInstallPresent(): boolean {
	return [
		'/usr/local/bin/ggsql-jupyter',
		'/usr/bin/ggsql-jupyter',
		'/Applications/ggsql.app/Contents/MacOS/ggsql-jupyter',
		path.join(process.env.PROGRAMFILES || 'C:\\Program Files', 'ggsql', 'ggsql-jupyter.exe'),
	].some(p => fs.existsSync(p));
}

// Environment host discovery reads. Saved and restored around any test that
// redirects it, so no other suite sees a doctored environment.
const HOST_ENV_KEYS = ['HOME', 'USERPROFILE', 'APPDATA', 'LOCALAPPDATA', 'PATH'] as const;
let savedEnv: Partial<Record<string, string | undefined>> = {};

function isolateHostEnv(homeDir: string): void {
	for (const key of HOST_ENV_KEYS) {
		savedEnv[key] = process.env[key];
	}
	process.env.HOME = homeDir;
	process.env.USERPROFILE = homeDir;
	process.env.APPDATA = path.join(homeDir, 'AppData', 'Roaming');
	process.env.LOCALAPPDATA = path.join(homeDir, 'AppData', 'Local');
	// An empty directory as PATH makes the which/where lookup fail, so whatever
	// the developer has installed cannot contribute a candidate.
	process.env.PATH = tempDir();
}

function restoreHostEnv(): void {
	for (const [key, value] of Object.entries(savedEnv)) {
		if (value === undefined) {
			delete process.env[key];
		} else {
			process.env[key] = value;
		}
	}
	savedEnv = {};
}

/**
 * A WorkspaceConfiguration stub covering the two members
 * resolveKernelStrategy() uses. Using a stub keeps the precedence cases
 * independent of the test instance's settings; the suite below pins the same
 * behaviour against the real configuration service.
 */
function fakeConfig(values: {
	strategy?: { global?: string; workspace?: string; workspaceFolder?: string };
	kernelPath?: string;
}): vscode.WorkspaceConfiguration {
	return {
		get: (key: string, defaultValue?: unknown) =>
			key === 'kernelPath' ? (values.kernelPath ?? defaultValue) : defaultValue,
		inspect: (key: string) =>
			key === 'kernelStrategy'
				? {
					key: 'ggsql.kernelStrategy',
					defaultValue: 'bundled',
					globalValue: values.strategy?.global,
					workspaceValue: values.strategy?.workspace,
					workspaceFolderValue: values.strategy?.workspaceFolder,
				}
				: undefined,
	} as unknown as vscode.WorkspaceConfiguration;
}

const HOST: KernelCandidate[] = [
	{ kernelPath: '/usr/local/bin/ggsql-jupyter', source: 'System' },
	{ kernelPath: '/opt/homebrew/bin/ggsql-jupyter', source: 'Path' },
];
const hostKernels = () => HOST;
const noHostKernels = () => [];

suite('kernel strategy', () => {
	test('the manifest declares bundled as the default', () => {
		// The rest of the suite assumes this default; it is also what makes the
		// extension work with no kernel installed.
		const property = realExtension().packageJSON.contributes.configuration.properties['ggsql.kernelStrategy'];
		assert.ok(property, 'ggsql.kernelStrategy is not contributed');
		assert.strictEqual(property.default, 'bundled');
		assert.deepStrictEqual(property.enum, ['bundled', 'environment', 'path']);
		// A shorter enumDescriptions silently misaligns the settings UI, pairing
		// each description with the wrong value.
		assert.strictEqual(property.enumDescriptions.length, property.enum.length);
	});

	test('an unset strategy resolves to bundled', () => {
		assert.strictEqual(resolveKernelStrategy(fakeConfig({})), 'bundled');
	});

	test('a configured kernelPath still means path', () => {
		// Migration: users who set ggsql.kernelPath before the strategy setting
		// existed must keep getting the kernel they named.
		assert.strictEqual(
			resolveKernelStrategy(fakeConfig({ kernelPath: '/opt/ggsql/ggsql-jupyter' })),
			'path',
		);
	});

	test('a whitespace-only kernelPath does not imply path', () => {
		assert.strictEqual(resolveKernelStrategy(fakeConfig({ kernelPath: '   ' })), 'bundled');
	});

	test('an explicit strategy overrides a configured kernelPath', () => {
		// Otherwise a user could never keep a path around while asking for the
		// bundled kernel.
		const config = fakeConfig({
			strategy: { global: 'bundled' },
			kernelPath: '/opt/ggsql/ggsql-jupyter',
		});
		assert.strictEqual(resolveKernelStrategy(config), 'bundled');
	});

	test('workspace scope wins over global scope', () => {
		const config = fakeConfig({ strategy: { global: 'environment', workspace: 'bundled' } });
		assert.strictEqual(resolveKernelStrategy(config), 'bundled');
	});

	test('workspace folder scope wins over workspace scope', () => {
		const config = fakeConfig({ strategy: { workspace: 'bundled', workspaceFolder: 'environment' } });
		assert.strictEqual(resolveKernelStrategy(config), 'environment');
	});

	test('an unknown strategy falls back to bundled', () => {
		// A hand-edited settings.json is not validated before it reaches here.
		const config = fakeConfig({ strategy: { global: 'whatever' } });
		assert.strictEqual(resolveKernelStrategy(config), 'bundled');
	});
});

suite('kernel candidate selection', () => {
	const bundled = '/ext/ggsql.ggsql-0.5.0-darwin-arm64/bundled/bin/ggsql-jupyter';

	test('bundled offers only the bundled kernel, with host kernels behind it', () => {
		const selection = selectKernelCandidates('bundled', bundled, undefined, hostKernels);
		assert.deepStrictEqual(selection.candidates, [{ kernelPath: bundled, source: 'Bundled' }]);
		// Reachable only when the bundled kernel turns out not to run.
		assert.deepStrictEqual(selection.fallback(), HOST);
	});

	test('the host lookup is not performed while choosing the bundled kernel', () => {
		// The PATH lookup shells out to which/where. The default strategy is the
		// common case and must not pay for it.
		let calls = 0;
		const counted = () => {
			calls++;
			return HOST;
		};
		selectKernelCandidates('bundled', bundled, undefined, counted);
		assert.strictEqual(calls, 0);
	});

	test('bundled falls back to host kernels when the build carries none', () => {
		// The platform-neutral VSIX, and the win32-arm64 build, ship no kernel
		// and must keep working through a host install.
		const selection = selectKernelCandidates('bundled', undefined, undefined, hostKernels);
		assert.deepStrictEqual(selection.candidates, HOST);
		assert.deepStrictEqual(selection.fallback(), []);
	});

	test('no bundle and no host kernel yields no candidates', () => {
		// Regression test for the phantom runtime: a ggsql runtime registered
		// against a kernel that is not there fails at session start with KS-19.
		const selection = selectKernelCandidates('bundled', undefined, undefined, noHostKernels);
		assert.deepStrictEqual(selection.candidates, []);
		assert.deepStrictEqual(selection.fallback(), []);
	});

	test('environment puts host kernels ahead of the bundled one', () => {
		const selection = selectKernelCandidates('environment', bundled, undefined, hostKernels);
		// One tier: the bundled kernel already stands behind the host ones.
		assert.deepStrictEqual(selection.candidates, [...HOST, { kernelPath: bundled, source: 'Bundled' }]);
		assert.deepStrictEqual(selection.fallback(), []);
	});

	test('environment falls back to the bundled kernel', () => {
		const selection = selectKernelCandidates('environment', bundled, undefined, noHostKernels);
		assert.deepStrictEqual(selection.candidates, [{ kernelPath: bundled, source: 'Bundled' }]);
	});

	test('path uses the configured kernel alone', () => {
		// Neither the bundled kernel nor a host install may quietly stand in for
		// the one the user named, so there is no fallback tier either.
		const selection = selectKernelCandidates('path', bundled, '/opt/ggsql/ggsql-jupyter', hostKernels);
		assert.deepStrictEqual(selection.candidates, [
			{ kernelPath: '/opt/ggsql/ggsql-jupyter', source: 'Setting' },
		]);
		assert.deepStrictEqual(selection.fallback(), []);
	});

	test('path with no configured kernel behaves as bundled', () => {
		const selection = selectKernelCandidates('path', bundled, undefined, hostKernels);
		assert.deepStrictEqual(selection.candidates, [{ kernelPath: bundled, source: 'Bundled' }]);
		// Including the dead-end notice, which an empty setting must not suppress.
		assert.strictEqual(selection.strategy, 'bundled');
		assert.deepStrictEqual(selection.fallback(), HOST);
	});
});

suite('kernel strategy from real settings', () => {
	// The stubbed inspect() above cannot prove any of this: only the real
	// configuration service distinguishes a set value from a default, and only
	// discoverKernelPaths proves the settings actually reach the precedence rule.
	const config = () => vscode.workspace.getConfiguration('ggsql');

	async function set(key: string, value: string | undefined): Promise<void> {
		await config().update(key, value, vscode.ConfigurationTarget.Global);
	}

	teardown(async () => {
		await set('kernelStrategy', undefined);
		await set('kernelPath', undefined);
	});

	test('an unset strategy resolves to the declared default', () => {
		assert.strictEqual(resolveKernelStrategy(config()), 'bundled');
	});

	test('a kernelPath in real settings migrates to the path strategy', async () => {
		await set('kernelPath', '/opt/ggsql/ggsql-jupyter');
		assert.strictEqual(resolveKernelStrategy(config()), 'path');
	});

	test('an explicitly set strategy wins over a configured path', async () => {
		await set('kernelPath', '/opt/ggsql/ggsql-jupyter');
		await set('kernelStrategy', 'environment');
		assert.strictEqual(resolveKernelStrategy(config()), 'environment');
	});

	test('the path strategy discovers the configured kernel', async () => {
		const configured = writeStubKernel(tempDir());
		await set('kernelStrategy', 'path');
		await set('kernelPath', configured);
		// A bundled kernel is present and must lose to the setting.
		const { extensionPath } = extensionDirWithBundle();
		assert.deepStrictEqual(
			discoverKernelPaths(contextFor(extensionPath)).candidates,
			[{ kernelPath: configured, source: 'Setting' }],
		);
	});

	test('the environment strategy puts the bundled kernel last', async () => {
		const { extensionPath, kernelPath } = extensionDirWithBundle();
		await set('kernelStrategy', 'environment');
		const candidates = discoverKernelPaths(contextFor(extensionPath)).candidates;
		// Whether this machine has host kernels is unknown, but the bundled one
		// is the fallback either way, so it comes last.
		assert.strictEqual(candidates.at(-1)?.source, 'Bundled');
		assert.strictEqual(candidates.at(-1)?.kernelPath, kernelPath);
	});
});

suite('resolving a configured kernel path', () => {
	test('an absolute path is used as given', () => {
		const configured = path.join(tempDir(), binaryName);
		assert.strictEqual(resolveConfiguredPath(configured), configured);
	});

	test('a bare name is looked up on PATH', () => {
		const name = process.platform === 'win32' ? 'cmd.exe' : 'sh';
		assert.ok(
			path.isAbsolute(resolveConfiguredPath(name)),
			`${name} did not resolve to an absolute path`,
		);
	});

	test('a bare name that is not on PATH is kept, then rejected', async () => {
		// Kept rather than dropped so that discovery reports the user's setting
		// as inaccessible instead of ignoring it without a word.
		const name = 'ggsql-jupyter-not-a-real-binary';
		assert.strictEqual(resolveConfiguredPath(name), name);
		assert.strictEqual(await isKernelAccessible(name), false);
	});
});

suite('kernel accessibility', () => {
	test('a bare binary name is not accessible', async () => {
		// Anything non-absolute reaching this check means the PATH lookup
		// failed; accepting it is the other half of the phantom runtime.
		assert.strictEqual(await isKernelAccessible(binaryName), false);
	});

	test('an executable file is accessible', async () => {
		assert.strictEqual(await isKernelAccessible(writeStubKernel(tempDir())), true);
	});

	test('a missing file is not accessible', async () => {
		assert.strictEqual(await isKernelAccessible(path.join(tempDir(), binaryName)), false);
	});

	test('a directory is not accessible', async () => {
		// Directories carry the executable bit on POSIX, so an access() check
		// on its own would pass one.
		assert.strictEqual(await isKernelAccessible(tempDir()), false);
	});
});

suite('bundled kernel discovery', () => {
	test('the bundled kernel is the only candidate under the default strategy', () => {
		const { extensionPath, kernelPath } = extensionDirWithBundle();
		const candidates = discoverKernelPaths(contextFor(extensionPath)).candidates;
		assert.deepStrictEqual(candidates, [{ kernelPath, source: 'Bundled' }]);
	});

	test('a bundled kernel missing its executable bit is repaired', function () {
		// Insurance against an unpack that drops the bit: without the repair the
		// binary would be dropped as inaccessible and no runtime would appear.
		if (process.platform === 'win32') {
			this.skip();
		}
		const { extensionPath, kernelPath } = extensionDirWithBundle(0o644);
		const candidates = discoverKernelPaths(contextFor(extensionPath)).candidates;
		assert.deepStrictEqual(candidates, [{ kernelPath, source: 'Bundled' }]);
		assert.ok(fs.statSync(kernelPath).mode & 0o111, 'the executable bit was not restored');
	});
});

suite('host kernel discovery', () => {
	let home: string;

	setup(() => {
		home = tempDir();
		isolateHostEnv(home);
	});

	teardown(() => {
		restoreHostEnv();
	});

	test('a user Jupyter kernelspec is found when the build has no kernel', function () {
		if (systemInstallPresent()) {
			this.skip();
		}
		const kernel = writeStubKernel(path.join(home, '.local', 'share', 'jupyter', 'kernels', 'ggsql'));
		const candidates = discoverKernelPaths(contextFor(tempDir())).candidates;
		assert.deepStrictEqual(candidates, [{ kernelPath: kernel, source: 'Jupyter' }]);
		for (const candidate of candidates) {
			assert.ok(path.isAbsolute(candidate.kernelPath), `${candidate.kernelPath} is not absolute`);
			assert.ok(fs.existsSync(candidate.kernelPath), `${candidate.kernelPath} does not exist`);
		}
	});

	test('one kernel reachable by two paths is reported once', function () {
		// The realistic duplicate is a kernelspec symlinked to the installed
		// binary. Both the macOS and Linux kernelspec locations are checked on
		// every platform, so two of them can name one file.
		if (process.platform === 'win32' || systemInstallPresent()) {
			this.skip();
		}
		const real = writeStubKernel(path.join(home, 'opt'));
		for (const dir of [
			path.join(home, 'Library', 'Jupyter', 'kernels', 'ggsql'),
			path.join(home, '.local', 'share', 'jupyter', 'kernels', 'ggsql'),
		]) {
			fs.mkdirSync(dir, { recursive: true });
			fs.symlinkSync(real, path.join(dir, binaryName));
		}
		const candidates = discoverKernelPaths(contextFor(tempDir())).candidates;
		assert.strictEqual(
			candidates.length,
			1,
			`expected one candidate, got ${candidates.map(c => c.kernelPath).join(', ')}`,
		);
	});

	test('a bundled kernel outranks an installed one', () => {
		const hostKernel = writeStubKernel(path.join(home, '.local', 'share', 'jupyter', 'kernels', 'ggsql'));
		const { extensionPath, kernelPath } = extensionDirWithBundle();
		const selection = discoverKernelPaths(contextFor(extensionPath));
		assert.deepStrictEqual(selection.candidates, [{ kernelPath, source: 'Bundled' }]);
		// The installed one is not gone, just behind: it is what the fallback
		// tier reaches for when the bundled kernel cannot run.
		assert.deepStrictEqual(
			selection.fallback(),
			[{ kernelPath: hostKernel, source: 'Jupyter' }],
		);
	});
});

suite('runtime registration', () => {
	async function collect(
		runtimes: AsyncGenerator<positron.LanguageRuntimeMetadata>,
	): Promise<positron.LanguageRuntimeMetadata[]> {
		const collected: positron.LanguageRuntimeMetadata[] = [];
		for await (const runtime of runtimes) {
			collected.push(runtime);
		}
		return collected;
	}

	/** An in-memory stand-in for context.globalState, which the probe cache uses. */
	function memoryState(): vscode.Memento {
		const store = new Map<string, unknown>();
		return {
			keys: () => [...store.keys()],
			get: (key: string, defaultValue?: unknown) =>
				store.has(key) ? store.get(key) : defaultValue,
			update: async (key: string, value: unknown) => {
				store.set(key, value);
			},
		} as unknown as vscode.Memento;
	}

	/**
	 * A manager over a stand-in extension directory.
	 *
	 * The probe defaults to passing: a stand-in kernel cannot be a real
	 * executable on every platform, so running one for real is left to the
	 * `kernel probe` suite and these tests inject the verdict instead.
	 */
	function managerFor(
		extensionPath: string,
		kernelSpecDir: string,
		options: { probe?: KernelProbe; globalState?: vscode.Memento } = {},
	): { manager: GgsqlRuntimeManager; globalState: vscode.Memento } {
		const globalState = options.globalState ?? memoryState();
		const context = {
			extensionPath,
			globalState,
			extension: { packageJSON: { version: realExtension().packageJSON.version } },
		} as unknown as vscode.ExtensionContext;
		const manager = new GgsqlRuntimeManager(context, {
			kernelSpecDir,
			probe: options.probe ?? (async () => true),
		});
		return { manager, globalState };
	}

	/** The key reportNoUsableKernel stamps once it has warned for this version. */
	const NOTICE_KEY = 'ggsql.noUsableKernelNotice';

	// The dead-end notice is fire-and-forget, so it is captured rather than
	// awaited. Stubbing it also keeps the suite from raising real notifications
	// in the test window.
	let warnings: string[] = [];
	let realShowWarningMessage: typeof vscode.window.showWarningMessage;

	setup(() => {
		warnings = [];
		realShowWarningMessage = vscode.window.showWarningMessage;
		(vscode.window as unknown as Record<string, unknown>).showWarningMessage =
			(message: string) => {
				warnings.push(message);
				return Promise.resolve(undefined);
			};
	});

	teardown(() => {
		(vscode.window as unknown as Record<string, unknown>).showWarningMessage =
			realShowWarningMessage;
	});

	test('the bundled kernel is registered as a single runtime', async () => {
		const { extensionPath, kernelPath } = extensionDirWithBundle();
		const runtimes = await collect(managerFor(extensionPath, tempDir()).manager.discoverAllRuntimes());
		assert.strictEqual(runtimes.length, 1);
		assert.strictEqual(runtimes[0].runtimeId, 'ggsql-bundled');
		assert.strictEqual(runtimes[0].runtimePath, kernelPath);
		assert.strictEqual(runtimes[0].runtimeName, 'ggsql');
	});

	test('discovery advertises the bundled kernel to Jupyter', async () => {
		// Quarto and Jupyter resolve ggsql through this spec. It is rewritten on
		// every window open because an extension update leaves the previous one
		// pointing into a directory that no longer exists.
		const { extensionPath, kernelPath } = extensionDirWithBundle();
		const kernelSpecDir = tempDir();
		await collect(managerFor(extensionPath, kernelSpecDir).manager.discoverAllRuntimes());
		const spec = JSON.parse(fs.readFileSync(path.join(kernelSpecDir, 'kernel.json'), 'utf8'));
		assert.strictEqual(spec.argv[0], kernelPath);
		assert.strictEqual(spec.language, 'ggsql');
	});

	test('a bundled path that is not an executable file registers nothing', async () => {
		// The accessibility filter is what stands between a broken bundle and a
		// runtime that fails at session start. A directory where the binary
		// should be exists and carries the executable bit, so only the isFile()
		// check rejects it — and no kernel spec may be written either.
		const extensionPath = tempDir();
		fs.mkdirSync(path.join(extensionPath, 'resources'), { recursive: true });
		fs.copyFileSync(
			path.join(realExtension().extensionPath, 'resources', 'ggsql-icon.svg'),
			path.join(extensionPath, 'resources', 'ggsql-icon.svg'),
		);
		fs.mkdirSync(path.join(extensionPath, 'bundled', 'bin', binaryName), { recursive: true });

		const kernelSpecDir = tempDir();
		const runtimes = await collect(managerFor(extensionPath, kernelSpecDir).manager.discoverAllRuntimes());
		assert.deepStrictEqual(runtimes, []);
		assert.strictEqual(fs.existsSync(path.join(kernelSpecDir, 'kernel.json')), false);
	});

	test('a bundled kernel that cannot run hands over to an installed one', async function () {
		// The bundled kernel is built for the platform, not for every system on
		// it: one built against newer shared libraries than the host provides
		// execs and then dies under the dynamic linker. Nothing on the
		// filesystem shows that, so the host install has to be reachable.
		if (systemInstallPresent()) {
			this.skip();
		}
		const home = tempDir();
		isolateHostEnv(home);
		try {
			const hostKernel = writeStubKernel(
				path.join(home, '.local', 'share', 'jupyter', 'kernels', 'ggsql'),
			);
			const { extensionPath, kernelPath } = extensionDirWithBundle();
			const { manager } = managerFor(extensionPath, tempDir(), {
				probe: async candidate => candidate !== kernelPath,
			});

			const runtimes = await collect(manager.discoverAllRuntimes());
			assert.strictEqual(runtimes.length, 1);
			assert.strictEqual(runtimes[0].runtimePath, hostKernel);
			// Named for where it came from, which is how the handover is
			// disclosed without interrupting the user.
			assert.strictEqual(runtimes[0].runtimeName, 'ggsql (Jupyter)');
			// A fallback that works is not worth interrupting anyone over.
			assert.deepStrictEqual(warnings, []);
		} finally {
			restoreHostEnv();
		}
	});

	test('a bundled kernel that cannot run is not advertised to Jupyter', async function () {
		// The kernel spec outlives the window and is what Quarto resolves, so
		// pointing it at a binary that does not run would break tools that
		// never see this extension's fallback.
		if (systemInstallPresent()) {
			this.skip();
		}
		isolateHostEnv(tempDir());
		try {
			const { extensionPath } = extensionDirWithBundle();
			const kernelSpecDir = tempDir();
			const { manager } = managerFor(extensionPath, kernelSpecDir, {
				probe: async () => false,
			});

			const runtimes = await collect(manager.discoverAllRuntimes());
			assert.deepStrictEqual(runtimes, []);
			assert.strictEqual(fs.existsSync(path.join(kernelSpecDir, 'kernel.json')), false);
		} finally {
			restoreHostEnv();
		}
	});

	test('a bundled kernel that cannot run and no installed one warns once', async function () {
		if (systemInstallPresent()) {
			this.skip();
		}
		isolateHostEnv(tempDir());
		try {
			const { extensionPath } = extensionDirWithBundle();
			const globalState = memoryState();

			const first = managerFor(extensionPath, tempDir(), {
				probe: async () => false,
				globalState,
			});
			assert.deepStrictEqual(await collect(first.manager.discoverAllRuntimes()), []);
			assert.strictEqual(warnings.length, 1);
			assert.match(warnings[0], /cannot run on this system/);
			assert.strictEqual(globalState.get(NOTICE_KEY), realExtension().packageJSON.version);

			// Discovery runs on every window open; the notice must not repeat.
			const second = managerFor(extensionPath, tempDir(), {
				probe: async () => false,
				globalState,
			});
			assert.deepStrictEqual(await collect(second.manager.discoverAllRuntimes()), []);
			assert.strictEqual(warnings.length, 1, 'the dead-end notice was shown twice');
		} finally {
			restoreHostEnv();
		}
	});

	test('a bundled kernel that runs is not re-probed on the next window', async () => {
		const { extensionPath } = extensionDirWithBundle();
		const globalState = memoryState();
		let probes = 0;
		const probe: KernelProbe = async () => {
			probes++;
			return true;
		};

		const first = managerFor(extensionPath, tempDir(), { probe, globalState });
		assert.strictEqual((await collect(first.manager.discoverAllRuntimes())).length, 1);
		assert.strictEqual(probes, 1);

		const second = managerFor(extensionPath, tempDir(), { probe, globalState });
		assert.strictEqual((await collect(second.manager.discoverAllRuntimes())).length, 1);
		assert.strictEqual(probes, 1, 'the bundled kernel was probed again');
	});

	test('a bundled kernel that runs never reaches for an installed one', async () => {
		// The fallback costs a PATH lookup that shells out. The default
		// strategy is the common case and must not pay for it.
		const home = tempDir();
		isolateHostEnv(home);
		try {
			writeStubKernel(path.join(home, '.local', 'share', 'jupyter', 'kernels', 'ggsql'));
			const { extensionPath, kernelPath } = extensionDirWithBundle();
			const { manager } = managerFor(extensionPath, tempDir());

			const runtimes = await collect(manager.discoverAllRuntimes());
			assert.strictEqual(runtimes.length, 1);
			assert.strictEqual(runtimes[0].runtimePath, kernelPath);
			assert.strictEqual(runtimes[0].runtimeName, 'ggsql');
		} finally {
			restoreHostEnv();
		}
	});

	test('a build with no kernel and nothing installed warns too', async function () {
		// The win32-arm64 and platform-neutral VSIXes carry no kernel at all.
		// That dead end is the same one, and gets the same notice.
		if (systemInstallPresent()) {
			this.skip();
		}
		isolateHostEnv(tempDir());
		try {
			const globalState = memoryState();
			const { manager } = managerFor(tempDir(), tempDir(), { globalState });
			assert.deepStrictEqual(await collect(manager.discoverAllRuntimes()), []);
			assert.strictEqual(warnings.length, 1);
			// Worded for a build that never carried a kernel, not a broken one.
			assert.match(warnings[0], /does not include a kernel/);
			assert.strictEqual(globalState.get(NOTICE_KEY), realExtension().packageJSON.version);
		} finally {
			restoreHostEnv();
		}
	});

	test('a machine with no kernel at all registers nothing', async function () {
		if (systemInstallPresent()) {
			this.skip();
		}
		// The W6 requirement stated in terms of what Positron receives, rather
		// than what the precedence rule returns.
		isolateHostEnv(tempDir());
		try {
			const runtimes = await collect(managerFor(tempDir(), tempDir()).manager.discoverAllRuntimes());
			assert.deepStrictEqual(runtimes, []);
		} finally {
			restoreHostEnv();
		}
	});
});

suite('runtime metadata', () => {
	// generateMetadata reads resources/ggsql-icon.svg from the extension folder,
	// so these use the real one with a stand-in version.
	function context(): vscode.ExtensionContext {
		return {
			extensionPath: realExtension().extensionPath,
			extension: { packageJSON: { version: '9.9.9' } },
		} as unknown as vscode.ExtensionContext;
	}

	test('the bundled runtime id survives an extension update', () => {
		// The bundled kernel lives under the versioned extension directory. An
		// id derived from that path would change on every update, dropping the
		// workspace's runtime affinity and its restorable sessions.
		const before = generateMetadata(context(), {
			kernelPath: '/ext/ggsql.ggsql-0.5.0-darwin-arm64/bundled/bin/ggsql-jupyter',
			source: 'Bundled',
		});
		const after = generateMetadata(context(), {
			kernelPath: '/ext/ggsql.ggsql-0.6.0-darwin-arm64/bundled/bin/ggsql-jupyter',
			source: 'Bundled',
		});
		assert.strictEqual(before.runtimeId, after.runtimeId);
	});

	test('the bundled runtime is named plain ggsql', () => {
		// It is the default, so there is nothing to distinguish it from.
		const metadata = generateMetadata(context(), {
			kernelPath: '/ext/ggsql.ggsql-0.5.0/bundled/bin/ggsql-jupyter',
			source: 'Bundled',
		});
		assert.strictEqual(metadata.runtimeName, 'ggsql');
	});

	test('other runtimes keep a per-path id and a qualified name', () => {
		const system = generateMetadata(context(), {
			kernelPath: '/usr/local/bin/ggsql-jupyter',
			source: 'System',
		});
		const setting = generateMetadata(context(), {
			kernelPath: '/opt/ggsql/ggsql-jupyter',
			source: 'Setting',
		});
		assert.strictEqual(system.runtimeName, 'ggsql (System)');
		assert.strictEqual(setting.runtimeName, 'ggsql (Setting)');
		assert.notStrictEqual(system.runtimeId, setting.runtimeId);
		assert.notStrictEqual(system.runtimeId, 'ggsql-bundled');
	});
});

suite('kernel probe', () => {
	// The probe is what separates a kernel that is present from one that runs.
	// The failure it exists for is a binary built against newer shared
	// libraries than the host provides: exec succeeds, the dynamic linker then
	// rejects it, and the process exits non-zero.

	test('a binary that exits non-zero does not pass', async function () {
		if (process.platform === 'win32') {
			this.skip();
		}
		const dir = tempDir();
		const kernelPath = path.join(dir, binaryName);
		fs.writeFileSync(kernelPath, '#!/bin/sh\nexit 1\n');
		fs.chmodSync(kernelPath, 0o755);
		assert.strictEqual(await probeKernel(kernelPath), false);
	});

	test('a binary that exits zero passes', async function () {
		if (process.platform === 'win32') {
			this.skip();
		}
		assert.strictEqual(await probeKernel(writeStubKernel(tempDir())), true);
	});

	test('a file that is not executable at all does not pass', async () => {
		// The nearest reachable stand-in for a binary the loader rejects: the
		// spawn fails rather than the process exiting non-zero, and the probe
		// has to treat both the same way.
		const kernelPath = path.join(tempDir(), binaryName);
		fs.writeFileSync(kernelPath, 'not a real executable\n');
		fs.chmodSync(kernelPath, 0o644);
		assert.strictEqual(await probeKernel(kernelPath), false);
	});

	test('a missing binary does not pass', async () => {
		assert.strictEqual(await probeKernel(path.join(tempDir(), binaryName)), false);
	});
});

suiteTeardown(() => {
	for (const dir of tempDirs) {
		fs.rmSync(dir, { recursive: true, force: true });
	}
});
