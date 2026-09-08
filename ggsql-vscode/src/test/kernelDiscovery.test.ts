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

/** The version the stubbed probe reports, standing in for a real kernel's. */
const STUB_VERSION = '1.2.3';

function realExtension(): vscode.Extension<unknown> {
	const extension = vscode.extensions.getExtension(EXTENSION_ID);
	assert.ok(extension, `extension ${EXTENSION_ID} not found`);
	return extension;
}

/** Write an executable stand-in for the kernel into `dir`. */
function writeStubKernel(dir: string, mode = 0o755, script = '#!/bin/sh\nexit 0\n'): string {
	fs.mkdirSync(dir, { recursive: true });
	const kernelPath = path.join(dir, binaryName);
	fs.writeFileSync(kernelPath, script);
	fs.chmodSync(kernelPath, mode);
	return kernelPath;
}

/**
 * Build a directory that looks like an installed platform-neutral VSIX: no
 * kernel, but the icon generateMetadata reads from the extension folder.
 */
function extensionDir(): string {
	const extensionPath = tempDir();
	fs.mkdirSync(path.join(extensionPath, 'resources'), { recursive: true });
	fs.copyFileSync(
		path.join(realExtension().extensionPath, 'resources', 'ggsql-icon.svg'),
		path.join(extensionPath, 'resources', 'ggsql-icon.svg'),
	);
	return extensionPath;
}

/** The same, as a platform VSIX: with a kernel at bundled/bin/. */
function extensionDirWithBundle(mode = 0o755): { extensionPath: string; kernelPath: string } {
	const extensionPath = extensionDir();
	return { extensionPath, kernelPath: writeStubKernel(path.join(extensionPath, 'bundled', 'bin'), mode) };
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

function contextFor(extensionPath: string, globalState = memoryState()): vscode.ExtensionContext {
	return {
		extensionPath,
		globalState,
		extension: { packageJSON: { version: realExtension().packageJSON.version } },
	} as unknown as vscode.ExtensionContext;
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

const HOST: KernelCandidate[] = [
	{ kernelPath: '/usr/local/bin/ggsql-jupyter', source: 'System' },
	{ kernelPath: '/opt/homebrew/bin/ggsql-jupyter', source: 'Path' },
];

suite('kernel candidate selection', () => {
	const bundled = '/ext/ggsql.ggsql-0.5.0-darwin-arm64/bundled/bin/ggsql-jupyter';
	const configured = '/opt/ggsql/ggsql-jupyter';

	test('every kernel found is offered, bundled ahead of installed ones', () => {
		// A machine with several ggsql installs shows all of them and lets the
		// user pick; the bundled one leads, which makes it the default.
		assert.deepStrictEqual(selectKernelCandidates(bundled, undefined, HOST), [
			{ kernelPath: bundled, source: 'Bundled' },
			...HOST,
		]);
	});

	test('a configured kernel leads the list', () => {
		// The user named a binary, so it is the one to offer first — but it no
		// longer suppresses the others.
		assert.deepStrictEqual(selectKernelCandidates(bundled, configured, HOST), [
			{ kernelPath: configured, source: 'Setting' },
			{ kernelPath: bundled, source: 'Bundled' },
			...HOST,
		]);
	});

	test('a build that carries no kernel offers the installed ones', () => {
		// The platform-neutral VSIX ships no kernel and must keep working
		// through a host install.
		assert.deepStrictEqual(selectKernelCandidates(undefined, undefined, HOST), HOST);
	});

	test('no bundle and no host kernel yields no candidates', () => {
		// Regression test for the phantom runtime: a ggsql runtime registered
		// against a kernel that is not there fails at session start with KS-19.
		assert.deepStrictEqual(selectKernelCandidates(undefined, undefined, []), []);
	});
});

suite('discovery from real settings', () => {
	// Only discoverKernelPaths proves the setting actually reaches the
	// precedence rule.
	const config = () => vscode.workspace.getConfiguration('ggsql');

	teardown(async () => {
		await config().update('kernelPath', undefined, vscode.ConfigurationTarget.Global);
	});

	test('a configured kernelPath is offered ahead of the bundled kernel', async () => {
		const configured = writeStubKernel(tempDir());
		await config().update('kernelPath', configured, vscode.ConfigurationTarget.Global);

		const { extensionPath, kernelPath } = extensionDirWithBundle();
		const candidates = discoverKernelPaths(contextFor(extensionPath));
		assert.deepStrictEqual(candidates.slice(0, 2), [
			{ kernelPath: configured, source: 'Setting' },
			{ kernelPath, source: 'Bundled' },
		]);
	});

	test('an unset kernelPath contributes no candidate', () => {
		const { extensionPath, kernelPath } = extensionDirWithBundle();
		const candidates = discoverKernelPaths(contextFor(extensionPath));
		assert.strictEqual(candidates[0].kernelPath, kernelPath);
		assert.ok(
			!candidates.some(candidate => candidate.source === 'Setting'),
			'an empty setting produced a candidate',
		);
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
	test('the bundled kernel leads the candidates', () => {
		const { extensionPath, kernelPath } = extensionDirWithBundle();
		const candidates = discoverKernelPaths(contextFor(extensionPath));
		assert.deepStrictEqual(candidates[0], { kernelPath, source: 'Bundled' });
	});

	test('a bundled kernel missing its executable bit is repaired', function () {
		// Insurance against an unpack that drops the bit: without the repair the
		// binary would be dropped as inaccessible and no runtime would appear.
		if (process.platform === 'win32') {
			this.skip();
		}
		const { extensionPath, kernelPath } = extensionDirWithBundle(0o644);
		const candidates = discoverKernelPaths(contextFor(extensionPath));
		assert.deepStrictEqual(candidates[0], { kernelPath, source: 'Bundled' });
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
		const candidates = discoverKernelPaths(contextFor(tempDir()));
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
		const candidates = discoverKernelPaths(contextFor(tempDir()));
		assert.strictEqual(
			candidates.length,
			1,
			`expected one candidate, got ${candidates.map(c => c.kernelPath).join(', ')}`,
		);
	});

	test('an installed kernel sits behind the bundled one', function () {
		if (systemInstallPresent()) {
			this.skip();
		}
		const hostKernel = writeStubKernel(path.join(home, '.local', 'share', 'jupyter', 'kernels', 'ggsql'));
		const { extensionPath, kernelPath } = extensionDirWithBundle();
		assert.deepStrictEqual(discoverKernelPaths(contextFor(extensionPath)), [
			{ kernelPath, source: 'Bundled' },
			{ kernelPath: hostKernel, source: 'Jupyter' },
		]);
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

	/**
	 * A manager over a stand-in extension directory.
	 *
	 * The probe defaults to reporting a version: a stand-in kernel cannot be a
	 * real executable on every platform, so running one for real is left to the
	 * `kernel probe` suite and these tests inject the verdict instead.
	 */
	function managerFor(
		extensionPath: string,
		kernelSpecDir: string,
		options: { probe?: KernelProbe; globalState?: vscode.Memento } = {},
	): { manager: GgsqlRuntimeManager; globalState: vscode.Memento } {
		const globalState = options.globalState ?? memoryState();
		const manager = new GgsqlRuntimeManager(contextFor(extensionPath, globalState), {
			kernelSpecDir,
			probe: options.probe ?? (async () => ({ version: STUB_VERSION })),
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

	test('the bundled kernel is registered under the version it reports', async () => {
		const { extensionPath, kernelPath } = extensionDirWithBundle();
		const runtimes = await collect(managerFor(extensionPath, tempDir()).manager.discoverAllRuntimes());
		assert.strictEqual(runtimes[0].runtimeId, 'ggsql-bundled');
		assert.strictEqual(runtimes[0].runtimePath, kernelPath);
		assert.strictEqual(runtimes[0].runtimeName, `ggsql ${STUB_VERSION}`);
	});

	test('every runnable kernel is registered, bundled first', async function () {
		// The picker shows each ggsql install on the machine rather than only
		// the extension's own, which is how a user keeps using theirs.
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
			const kernelSpecDir = tempDir();
			const runtimes = await collect(
				managerFor(extensionPath, kernelSpecDir).manager.discoverAllRuntimes(),
			);

			assert.deepStrictEqual(
				runtimes.map(runtime => [runtime.runtimeName, runtime.runtimePath]),
				[
					[`ggsql ${STUB_VERSION}`, kernelPath],
					[`ggsql ${STUB_VERSION} (Jupyter)`, hostKernel],
				],
			);
			// Every runtime is a distinct one as far as Positron is concerned.
			assert.strictEqual(new Set(runtimes.map(runtime => runtime.runtimeId)).size, 2);
			// One spec is written, for the kernel that leads the list.
			const spec = JSON.parse(fs.readFileSync(path.join(kernelSpecDir, 'kernel.json'), 'utf8'));
			assert.strictEqual(spec.argv[0], kernelPath);
		} finally {
			restoreHostEnv();
		}
	});

	test('discovery advertises the leading kernel to Jupyter', async () => {
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

		// "Registers nothing" is a claim about every runtime discovery yields,
		// so the developer's own installs have to be out of the picture.
		isolateHostEnv(tempDir());
		try {
			const kernelSpecDir = tempDir();
			const runtimes = await collect(
				managerFor(extensionPath, kernelSpecDir).manager.discoverAllRuntimes(),
			);
			assert.deepStrictEqual(runtimes, []);
			assert.strictEqual(fs.existsSync(path.join(kernelSpecDir, 'kernel.json')), false);
		} finally {
			restoreHostEnv();
		}
	});

	test('a bundled kernel that cannot run leaves the installed one registered', async function () {
		// The bundled kernel is built for the platform, not for every system on
		// it: one built against newer shared libraries than the host provides
		// execs and then dies under the dynamic linker. Nothing on the
		// filesystem shows that, so the host install has to remain usable.
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
				probe: async candidate =>
					candidate === kernelPath ? undefined : { version: STUB_VERSION },
			});

			const runtimes = await collect(manager.discoverAllRuntimes());
			assert.strictEqual(runtimes.length, 1);
			assert.strictEqual(runtimes[0].runtimePath, hostKernel);
			// Named for where it came from, which is how the handover is
			// disclosed without interrupting the user.
			assert.strictEqual(runtimes[0].runtimeName, `ggsql ${STUB_VERSION} (Jupyter)`);
			// A fallback that works is not worth interrupting anyone over.
			assert.deepStrictEqual(warnings, []);
		} finally {
			restoreHostEnv();
		}
	});

	test('an installed kernel too old to report a version is still registered', async function () {
		// Kernels released before the --version flag exit non-zero on it. Only
		// the bundled kernel has to pass the probe; dropping the others would
		// take away the install the user already had.
		if (systemInstallPresent()) {
			this.skip();
		}
		const home = tempDir();
		isolateHostEnv(home);
		try {
			const hostKernel = writeStubKernel(
				path.join(home, '.local', 'share', 'jupyter', 'kernels', 'ggsql'),
			);
			const { manager } = managerFor(extensionDir(), tempDir(), { probe: async () => undefined });

			const runtimes = await collect(manager.discoverAllRuntimes());
			assert.strictEqual(runtimes.length, 1);
			assert.strictEqual(runtimes[0].runtimePath, hostKernel);
			// No version to interpolate, so the name says only where it came from.
			assert.strictEqual(runtimes[0].runtimeName, 'ggsql (Jupyter)');
			assert.deepStrictEqual(warnings, []);
		} finally {
			restoreHostEnv();
		}
	});

	test('a bundled kernel that cannot run is not advertised to Jupyter', async function () {
		// The kernel spec outlives the window and is what Quarto resolves, so
		// pointing it at a binary that does not run would break tools that
		// never see this extension's other candidates.
		if (systemInstallPresent()) {
			this.skip();
		}
		isolateHostEnv(tempDir());
		try {
			const { extensionPath } = extensionDirWithBundle();
			const kernelSpecDir = tempDir();
			const { manager } = managerFor(extensionPath, kernelSpecDir, {
				probe: async () => undefined,
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
				probe: async () => undefined,
				globalState,
			});
			assert.deepStrictEqual(await collect(first.manager.discoverAllRuntimes()), []);
			assert.strictEqual(warnings.length, 1);
			assert.match(warnings[0], /cannot run on this system/);
			assert.strictEqual(globalState.get(NOTICE_KEY), realExtension().packageJSON.version);

			// Discovery runs on every window open; the notice must not repeat.
			const second = managerFor(extensionPath, tempDir(), {
				probe: async () => undefined,
				globalState,
			});
			assert.deepStrictEqual(await collect(second.manager.discoverAllRuntimes()), []);
			assert.strictEqual(warnings.length, 1, 'the dead-end notice was shown twice');
		} finally {
			restoreHostEnv();
		}
	});

	test('a kernel that reported its version is not re-probed on the next window', async () => {
		const { extensionPath } = extensionDirWithBundle();
		const globalState = memoryState();
		let probes = 0;
		const probe: KernelProbe = async () => {
			probes++;
			return { version: STUB_VERSION };
		};

		// The count is per kernel, so it only means one thing if the bundled
		// kernel is the only one discovery can see.
		isolateHostEnv(tempDir());
		try {
			const first = managerFor(extensionPath, tempDir(), { probe, globalState });
			assert.strictEqual((await collect(first.manager.discoverAllRuntimes()))[0].runtimeVersion, STUB_VERSION);
			assert.strictEqual(probes, 1);

			const second = managerFor(extensionPath, tempDir(), { probe, globalState });
			assert.strictEqual((await collect(second.manager.discoverAllRuntimes()))[0].runtimeVersion, STUB_VERSION);
			assert.strictEqual(probes, 1, 'the kernel was probed again');
		} finally {
			restoreHostEnv();
		}
	});

	test('a kernel replaced in place is probed again', async () => {
		// An installer upgrading a kernel leaves its path alone, so a cache
		// keyed on the path alone would keep reporting the old version.
		const { extensionPath, kernelPath } = extensionDirWithBundle();
		const globalState = memoryState();
		let version = '1.0.0';
		const probe: KernelProbe = async () => ({ version });

		const first = managerFor(extensionPath, tempDir(), { probe, globalState });
		assert.strictEqual((await collect(first.manager.discoverAllRuntimes()))[0].runtimeVersion, '1.0.0');

		fs.appendFileSync(kernelPath, '# upgraded\n');
		version = '2.0.0';

		const second = managerFor(extensionPath, tempDir(), { probe, globalState });
		assert.strictEqual((await collect(second.manager.discoverAllRuntimes()))[0].runtimeVersion, '2.0.0');
	});

	test('a build with no kernel and nothing installed warns too', async function () {
		// The platform-neutral VSIX carries no kernel at all. That dead end
		// is the same one, and gets the same notice.
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
	// so these use the real one.
	const context = () => contextFor(realExtension().extensionPath);

	test('the bundled runtime id survives an extension update', () => {
		// The bundled kernel lives under the versioned extension directory. An
		// id derived from that path would change on every update, dropping the
		// workspace's runtime affinity and its restorable sessions.
		const before = generateMetadata(context(), {
			kernelPath: '/ext/ggsql.ggsql-0.5.0-darwin-arm64/bundled/bin/ggsql-jupyter',
			source: 'Bundled',
			version: '0.5.0',
		});
		const after = generateMetadata(context(), {
			kernelPath: '/ext/ggsql.ggsql-0.6.0-darwin-arm64/bundled/bin/ggsql-jupyter',
			source: 'Bundled',
			version: '0.6.0',
		});
		assert.strictEqual(before.runtimeId, after.runtimeId);
	});

	test('the bundled runtime is named for the version the kernel reports', () => {
		// As Positron names its own runtimes, and the kernel is what runs the
		// query, so its version is the one worth showing. It is the default, so
		// there is nothing to qualify it against.
		const metadata = generateMetadata(context(), {
			kernelPath: '/ext/ggsql.ggsql-0.5.0/bundled/bin/ggsql-jupyter',
			source: 'Bundled',
			version: '0.5.0',
		});
		assert.strictEqual(metadata.runtimeName, 'ggsql 0.5.0');
		assert.strictEqual(metadata.runtimeVersion, '0.5.0');
		assert.strictEqual(metadata.languageVersion, '0.5.0');
		// The console tab is per session, where the version adds nothing.
		assert.strictEqual(metadata.runtimeShortName, 'ggsql');
	});

	test('other runtimes keep a per-path id and a qualified name', () => {
		const system = generateMetadata(context(), {
			kernelPath: '/usr/local/bin/ggsql-jupyter',
			source: 'System',
			version: '0.4.0',
		});
		const setting = generateMetadata(context(), {
			kernelPath: '/opt/ggsql/ggsql-jupyter',
			source: 'Setting',
			version: '0.4.0',
		});
		assert.strictEqual(system.runtimeName, 'ggsql 0.4.0 (System)');
		assert.strictEqual(setting.runtimeName, 'ggsql 0.4.0 (Setting)');
		assert.notStrictEqual(system.runtimeId, setting.runtimeId);
		assert.notStrictEqual(system.runtimeId, 'ggsql-bundled');
	});

	test('a kernel that reports no version falls back to the extension version', () => {
		const metadata = generateMetadata(context(), {
			kernelPath: '/usr/local/bin/ggsql-jupyter',
			source: 'System',
		});
		// Nothing to interpolate, so the name says only where it came from.
		assert.strictEqual(metadata.runtimeName, 'ggsql (System)');
		assert.strictEqual(metadata.runtimeVersion, realExtension().packageJSON.version);
	});
});

suite('metadata validation', () => {
	function managerFor(extensionPath: string, probe?: KernelProbe): GgsqlRuntimeManager {
		return new GgsqlRuntimeManager(contextFor(extensionPath), {
			kernelSpecDir: tempDir(),
			probe: probe ?? (async () => ({ version: STUB_VERSION })),
		});
	}

	test('bundled metadata from a superseded extension version is repointed', async () => {
		// The fixed bundled runtime id is what carries runtime affinity and
		// restorable sessions across an update; the path it was stored with
		// points into the extension directory that update removed.
		const superseded = extensionDirWithBundle();
		const current = extensionDirWithBundle();
		const stale = generateMetadata(contextFor(superseded.extensionPath), {
			kernelPath: superseded.kernelPath,
			source: 'Bundled',
			version: '0.1.0',
		});

		const validated = await managerFor(current.extensionPath).validateMetadata(stale);
		assert.strictEqual(validated.runtimeId, stale.runtimeId);
		assert.strictEqual(validated.runtimePath, current.kernelPath);
		assert.strictEqual(validated.runtimeVersion, STUB_VERSION);
	});

	test('metadata for a kernel that is no longer there is rejected', async () => {
		// Rejecting is how Positron learns to drop a runtime it stored for a
		// kernel that has since been uninstalled.
		const { extensionPath } = extensionDirWithBundle();
		const gone = generateMetadata(contextFor(extensionPath), {
			kernelPath: path.join(tempDir(), binaryName),
			source: 'System',
			version: '0.1.0',
		});
		await assert.rejects(
			() => managerFor(extensionPath).validateMetadata(gone),
			/No usable ggsql kernel/,
		);
	});

	test('bundled metadata is rejected when the bundled kernel cannot run', async () => {
		const { extensionPath, kernelPath } = extensionDirWithBundle();
		const metadata = generateMetadata(contextFor(extensionPath), {
			kernelPath,
			source: 'Bundled',
			version: '0.1.0',
		});
		await assert.rejects(
			() => managerFor(extensionPath, async () => undefined).validateMetadata(metadata),
			/No usable ggsql kernel/,
		);
	});
});

suite('kernel probe', () => {
	// The probe reads the version to show in the picker, and is what separates a
	// kernel that is present from one that runs. The failure it exists for is a
	// binary built against newer shared libraries than the host provides: exec
	// succeeds, the dynamic linker then rejects it, and the process exits
	// non-zero.

	test('the reported version is read from the output', async function () {
		if (process.platform === 'win32') {
			this.skip();
		}
		const kernelPath = writeStubKernel(
			tempDir(),
			0o755,
			'#!/bin/sh\necho "ggsql-jupyter 1.2.3"\n',
		);
		assert.deepStrictEqual(await probeKernel(kernelPath), { version: '1.2.3' });
	});

	test('a binary that exits zero without a version still passes', async function () {
		// Its runtime is offered without a version rather than dropped.
		if (process.platform === 'win32') {
			this.skip();
		}
		assert.deepStrictEqual(await probeKernel(writeStubKernel(tempDir())), { version: undefined });
	});

	test('a binary that exits non-zero does not pass', async function () {
		if (process.platform === 'win32') {
			this.skip();
		}
		const dir = tempDir();
		const kernelPath = path.join(dir, binaryName);
		fs.writeFileSync(kernelPath, '#!/bin/sh\nexit 1\n');
		fs.chmodSync(kernelPath, 0o755);
		assert.strictEqual(await probeKernel(kernelPath), undefined);
	});

	test('a kernel older than --version passes, without one', async function () {
		// Kernels released before the flag reject it the way clap does, exiting
		// non-zero. That is indistinguishable by exit status from a binary the
		// loader killed, so the probe falls back to `--help`, which every
		// version answers. Dropping these would take away the install the user
		// already had.
		if (process.platform === 'win32') {
			this.skip();
		}
		const kernelPath = writeStubKernel(
			tempDir(),
			0o755,
			'#!/bin/sh\ncase "$1" in --help) exit 0 ;; *) echo "error: unexpected argument" >&2; exit 2 ;; esac\n',
		);
		assert.deepStrictEqual(await probeKernel(kernelPath), { version: undefined });
	});

	test('a binary that rejects every argument does not pass', async function () {
		// The `--help` fallback must not become a way back in for a kernel that
		// exec's and then dies, which is the failure the probe exists for.
		if (process.platform === 'win32') {
			this.skip();
		}
		const kernelPath = writeStubKernel(tempDir(), 0o755, '#!/bin/sh\nexit 127\n');
		assert.strictEqual(await probeKernel(kernelPath), undefined);
	});

	test('a file that is not executable at all does not pass', async () => {
		// The nearest reachable stand-in for a binary the loader rejects: the
		// spawn fails rather than the process exiting non-zero, and the probe
		// has to treat both the same way.
		const kernelPath = path.join(tempDir(), binaryName);
		fs.writeFileSync(kernelPath, 'not a real executable\n');
		fs.chmodSync(kernelPath, 0o644);
		assert.strictEqual(await probeKernel(kernelPath), undefined);
	});

	test('a missing binary does not pass', async () => {
		assert.strictEqual(await probeKernel(path.join(tempDir(), binaryName)), undefined);
	});
});

suiteTeardown(() => {
	for (const dir of tempDirs) {
		fs.rmSync(dir, { recursive: true, force: true });
	}
});
