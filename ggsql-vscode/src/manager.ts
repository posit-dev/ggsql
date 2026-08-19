/*
 * ggsql Language Runtime Manager
 *
 * Implements the Positron LanguageRuntimeManager interface to provide
 * ggsql runtime capabilities by wrapping the ggsql-jupyter kernel.
 */

import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import * as cp from 'child_process';
import * as crypto from 'crypto';
import type * as positron from '@posit-dev/positron';
import type { JupyterKernelSpec, PositronSupervisorApi } from './types';
import { log, showLog } from './extension';

/** Where a kernel candidate was discovered */
type KernelSource = 'Bundled' | 'Setting' | 'Jupyter' | 'System' | 'Path';

/**
 * How to pick a kernel, from the `ggsql.kernelStrategy` setting.
 *
 * - `bundled`: the kernel shipped inside the extension.
 * - `environment`: a kernel installed on the machine, falling back to the
 *   bundled one.
 * - `path`: the binary named by `ggsql.kernelPath`.
 */
export type KernelStrategy = 'bundled' | 'environment' | 'path';

const KERNEL_STRATEGIES: readonly string[] = ['bundled', 'environment', 'path'];

/**
 * A discovered ggsql-jupyter kernel candidate
 */
export interface KernelCandidate {
    /** Path to the ggsql-jupyter binary */
    kernelPath: string;
    /** Human-readable label for where this was found */
    source: KernelSource;
}

/** Platform-specific file name of the kernel executable */
function kernelBinaryName(): string {
    return process.platform === 'win32' ? 'ggsql-jupyter.exe' : 'ggsql-jupyter';
}

/**
 * Look a binary up on the system PATH.
 *
 * Returns undefined when it is not there. Callers must not fall back to the
 * bare name: a bare name satisfies every existence check further down and so
 * registers a runtime that cannot start.
 */
function findOnPath(binaryName: string): string | undefined {
    try {
        const cmd = process.platform === 'win32' ? 'where' : 'which';
        const resolved = cp.execFileSync(cmd, [binaryName], {
            encoding: 'utf8',
            timeout: 5000,
        }).trim().split(/\r?\n/)[0];
        if (resolved && path.isAbsolute(resolved)) {
            log(`Resolved '${binaryName}' to '${resolved}'`);
            return resolved;
        }
    } catch {
        // which/where exit non-zero when the name is not on PATH
    }
    log(`'${binaryName}' is not on PATH`);
    return undefined;
}

/**
 * Absolutise `ggsql.kernelPath`.
 *
 * A bare name is looked up on PATH; if that fails the configured value is kept
 * as-is, so that discovery rejects it as inaccessible and logs it back to the
 * user rather than silently ignoring the setting.
 */
export function resolveConfiguredPath(configuredPath: string): string {
    if (path.isAbsolute(configuredPath)) {
        return configuredPath;
    }
    return findOnPath(configuredPath) ?? configuredPath;
}

/**
 * Restore the executable bit on the bundled kernel if it is missing.
 *
 * `vsce` preserves the bit through package and install, so this should never
 * fire; it is insurance against an unpack that drops it, which would otherwise
 * present as the bundled kernel silently not being discovered.
 */
function ensureExecutable(binaryPath: string): void {
    if (process.platform === 'win32') {
        return;
    }
    try {
        fs.accessSync(binaryPath, fs.constants.X_OK);
        return;
    } catch {
        // Fall through and try to fix it
    }
    try {
        fs.chmodSync(binaryPath, fs.statSync(binaryPath).mode | 0o111);
        log(`Restored the executable bit on ${binaryPath}`);
    } catch (err) {
        log(`Could not make ${binaryPath} executable: ${err}`);
    }
}

/**
 * Path to the kernel shipped inside the extension, or undefined for a build
 * that carries none (the platform-neutral VSIX).
 */
function bundledKernelPath(context: vscode.ExtensionContext): string | undefined {
    const bundled = path.join(context.extensionPath, 'bundled', 'bin', kernelBinaryName());
    if (!fs.existsSync(bundled)) {
        return undefined;
    }
    ensureExecutable(bundled);
    return bundled;
}

/**
 * Find kernels installed on the machine: Jupyter kernelspec locations, then the
 * install locations of the native packages, then PATH.
 */
function discoverHostKernels(): KernelCandidate[] {
    const candidates: KernelCandidate[] = [];
    const binaryName = kernelBinaryName();

    // Jupyter kernelspec locations
    const homeDir = process.env.HOME || process.env.USERPROFILE || '';
    const kernelspecPaths = [
        // User kernelspec (macOS)
        path.join(homeDir, 'Library', 'Jupyter', 'kernels', 'ggsql', binaryName),
        // User kernelspec (Linux)
        path.join(homeDir, '.local', 'share', 'jupyter', 'kernels', 'ggsql', binaryName),
        // User kernelspec (Windows)
        path.join(
            process.env.APPDATA || path.join(homeDir, 'AppData', 'Roaming'),
            'jupyter', 'kernels', 'ggsql', binaryName
        ),
        // System kernelspec (macOS)
        path.join('/usr', 'local', 'share', 'jupyter', 'kernels', 'ggsql', binaryName),
        // System kernelspec (Linux)
        path.join('/usr', 'share', 'jupyter', 'kernels', 'ggsql', binaryName),
    ];
    for (const p of kernelspecPaths) {
        if (fs.existsSync(p)) {
            candidates.push({ kernelPath: p, source: 'Jupyter' });
        }
    }

    // Cargo-packager install locations
    const packagerPaths: string[] = [];
    if (process.platform === 'darwin') {
        // PKG installer (current)
        packagerPaths.push('/usr/local/bin/ggsql-jupyter');
        // Legacy DMG / .app bundle install
        packagerPaths.push('/Applications/ggsql.app/Contents/MacOS/ggsql-jupyter');
    } else if (process.platform === 'win32') {
        const programFiles = process.env.PROGRAMFILES || 'C:\\Program Files';
        packagerPaths.push(path.join(programFiles, 'ggsql', 'ggsql-jupyter.exe'));
        const localAppData = process.env.LOCALAPPDATA;
        if (localAppData) {
            packagerPaths.push(path.join(localAppData, 'ggsql', 'ggsql-jupyter.exe'));
        }
    } else {
        // Linux deb package
        packagerPaths.push('/usr/bin/ggsql-jupyter');
    }
    for (const p of packagerPaths) {
        if (fs.existsSync(p)) {
            candidates.push({ kernelPath: p, source: 'System' });
        }
    }

    // PATH, last of the host locations
    const onPath = findOnPath(binaryName);
    if (onPath) {
        candidates.push({ kernelPath: onPath, source: 'Path' });
    }

    return candidates;
}

/**
 * Resolve `ggsql.kernelStrategy`.
 *
 * Migration for users who configured `ggsql.kernelPath` before the strategy
 * setting existed: a non-empty path with no explicitly set strategy still
 * means "use that path", so their setting keeps working untouched.
 */
export function resolveKernelStrategy(config: vscode.WorkspaceConfiguration): KernelStrategy {
    const inspected = config.inspect<string>('kernelStrategy');
    const explicit = inspected?.workspaceFolderValue
        ?? inspected?.workspaceValue
        ?? inspected?.globalValue;

    if (explicit !== undefined) {
        if (KERNEL_STRATEGIES.includes(explicit)) {
            return explicit as KernelStrategy;
        }
        log(`Ignoring unknown ggsql.kernelStrategy '${explicit}'`);
    } else if (config.get<string>('kernelPath', '').trim() !== '') {
        return 'path';
    }

    return 'bundled';
}

/**
 * The kernels a window should consider, in priority order, plus the tier to
 * fall back to when none of them can run.
 */
export interface KernelSelection {
    /** The strategy these candidates were chosen under. */
    strategy: KernelStrategy;
    /** Candidates to offer, best first. */
    candidates: KernelCandidate[];
    /**
     * Host kernels to consult when nothing in `candidates` turns out to be
     * runnable. A callback, not a list, so that the common case — a bundled
     * kernel that runs — never pays for the PATH lookup.
     */
    fallback: () => KernelCandidate[];
}

const NO_FALLBACK = (): KernelCandidate[] => [];

/**
 * Apply a strategy to the places a kernel can come from.
 *
 * `hostKernels` is a callback so that the common case — the bundled kernel with
 * the default strategy — does not pay for a PATH lookup it will not use.
 */
export function selectKernelCandidates(
    strategy: KernelStrategy,
    bundledPath: string | undefined,
    configuredPath: string | undefined,
    hostKernels: () => KernelCandidate[],
): KernelSelection {
    const bundled: KernelCandidate[] = bundledPath
        ? [{ kernelPath: bundledPath, source: 'Bundled' }]
        : [];

    let effective = strategy;
    if (strategy === 'path') {
        if (configuredPath) {
            // The user named a binary. Nothing may quietly stand in for it, so
            // there is no fallback tier here.
            return {
                strategy,
                candidates: [{ kernelPath: configuredPath, source: 'Setting' }],
                fallback: NO_FALLBACK,
            };
        }
        // Nothing to point at. Treat it as the default rather than registering
        // no runtime at all.
        log('ggsql.kernelStrategy is "path" but ggsql.kernelPath is empty; using the bundled kernel');
        effective = 'bundled';
    } else if (strategy === 'environment') {
        // Host kernels are already ahead of the bundled one, so a failure among
        // them falls through to it within the same tier.
        return { strategy, candidates: [...hostKernels(), ...bundled], fallback: NO_FALLBACK };
    }

    if (bundled.length > 0) {
        // The bundled kernel is built for this platform but not for every
        // system it can be installed on: it can be too new for the host's
        // shared libraries, which only shows up when it is run. Host kernels
        // stand behind it for exactly that case.
        return { strategy: effective, candidates: bundled, fallback: hostKernels };
    }
    // A build that carries no kernel still looks for a host install, or it
    // would offer nothing at all.
    return { strategy: effective, candidates: hostKernels(), fallback: NO_FALLBACK };
}

/**
 * Drop candidates that name a file an earlier candidate already named, keeping
 * the highest-priority occurrence.
 */
function dedupeCandidates(candidates: KernelCandidate[]): KernelCandidate[] {
    const seen = new Set<string>();
    const deduped: KernelCandidate[] = [];
    for (const candidate of candidates) {
        let resolved: string;
        try {
            resolved = fs.realpathSync(candidate.kernelPath);
        } catch {
            resolved = candidate.kernelPath;
        }
        if (!seen.has(resolved)) {
            seen.add(resolved);
            deduped.push(candidate);
        } else {
            log(`Skipping duplicate kernel path: ${candidate.kernelPath} (resolves to ${resolved})`);
        }
    }
    return deduped;
}

/**
 * Discover the ggsql-jupyter kernels this window should offer, in priority
 * order.
 */
export function discoverKernelPaths(context: vscode.ExtensionContext): KernelSelection {
    const config = vscode.workspace.getConfiguration('ggsql');
    const strategy = resolveKernelStrategy(config);
    log(`Kernel strategy: ${strategy}`);

    const configuredPath = config.get<string>('kernelPath', '').trim();

    const selection = selectKernelCandidates(
        strategy,
        bundledKernelPath(context),
        configuredPath === '' ? undefined : resolveConfiguredPath(configuredPath),
        discoverHostKernels,
    );

    return {
        strategy: selection.strategy,
        candidates: dedupeCandidates(selection.candidates),
        fallback: () => dedupeCandidates(selection.fallback()),
    };
}

/**
 * Check that a candidate is a file this process can execute.
 *
 * A path that is not absolute is rejected: discovery absolutises every source
 * it can, so a bare name reaching here means the PATH lookup failed, and
 * accepting it would register a runtime that fails at session start.
 */
export async function isKernelAccessible(kernelPath: string): Promise<boolean> {
    if (!path.isAbsolute(kernelPath)) {
        return false;
    }
    try {
        const stats = await fs.promises.stat(kernelPath);
        if (!stats.isFile()) {
            return false;
        }
        await fs.promises.access(kernelPath, fs.constants.X_OK);
        return true;
    } catch {
        return false;
    }
}

/** How long the probe waits for the kernel to report its version. */
const KERNEL_PROBE_TIMEOUT_MS = 15000;

/** Where the last successful probe is remembered, to keep it to one per update. */
const PROBE_CACHE_KEY = 'ggsql.bundledKernelProbe';

/** Where the dead-end notice records the version it has already reported. */
const NO_KERNEL_NOTICE_KEY = 'ggsql.noUsableKernelNotice';

/** Install instructions offered when no kernel on this machine can run. */
const INSTALL_DOCS_URL = 'https://ggsql.org/get_started/installation.html';

interface ProbeCacheEntry {
    extensionVersion: string;
    kernelPath: string;
}

/** Runs a kernel binary and reports whether it started. */
export type KernelProbe = (kernelPath: string) => Promise<boolean>;

/**
 * Run the kernel and see whether it starts.
 *
 * An accessibility check cannot answer this. A binary built against newer
 * shared libraries than the host provides passes every filesystem test and
 * still fails: the kernel is exec'd successfully and then the dynamic linker
 * rejects it, so the process exits non-zero before it can serve a session.
 * `--version` is the cheapest thing that exercises that whole path.
 */
export function probeKernel(kernelPath: string): Promise<boolean> {
    return new Promise(resolve => {
        cp.execFile(
            kernelPath,
            ['--version'],
            { timeout: KERNEL_PROBE_TIMEOUT_MS, windowsHide: true },
            err => {
                if (err) {
                    log(`Kernel probe failed for ${kernelPath}: ${err.message}`);
                }
                resolve(!err);
            },
        );
    });
}

/**
 * Probe the bundled kernel, remembering a success across windows.
 *
 * Only a success is cached, and only for the extension version that produced
 * it: a failure is cheap to repeat (the linker gives up immediately) and
 * re-running it means a host that gains the libraries the kernel needs starts
 * working without waiting for an extension update.
 */
async function probeBundledKernel(
    context: vscode.ExtensionContext,
    kernelPath: string,
    probe: KernelProbe,
): Promise<boolean> {
    const extensionVersion = context.extension.packageJSON.version as string;
    const cached = context.globalState.get<ProbeCacheEntry>(PROBE_CACHE_KEY);
    if (cached?.extensionVersion === extensionVersion && cached.kernelPath === kernelPath) {
        return true;
    }

    const ok = await probe(kernelPath);
    if (ok) {
        await context.globalState.update(PROBE_CACHE_KEY, { extensionVersion, kernelPath });
    }
    return ok;
}

/**
 * Decide whether a candidate can actually serve a session.
 *
 * The bundled kernel is additionally run, because it is the one the extension
 * chose rather than the user, and it is the one that can be wrong about the
 * system it landed on. A kernel the user installed is taken at its word.
 */
async function canRunKernel(
    context: vscode.ExtensionContext,
    candidate: KernelCandidate,
    probe: KernelProbe,
): Promise<boolean> {
    if (!await isKernelAccessible(candidate.kernelPath)) {
        return false;
    }
    if (candidate.source !== 'Bundled') {
        return true;
    }
    return probeBundledKernel(context, candidate.kernelPath, probe);
}

/**
 * Tell the user that nothing on this machine can run ggsql queries.
 *
 * Only for the dead end: a fallback that succeeds is reported by the runtime's
 * name in the picker and by the log, and needs no interruption. Shown once per
 * extension version, and never awaited, so discovery does not sit waiting for
 * the notification to be dismissed.
 */
function reportNoUsableKernel(
    context: vscode.ExtensionContext,
    bundledRejected: boolean,
): void {
    const version = context.extension.packageJSON.version as string;
    if (context.globalState.get<string>(NO_KERNEL_NOTICE_KEY) === version) {
        return;
    }
    void context.globalState.update(NO_KERNEL_NOTICE_KEY, version);

    const reason = bundledRejected
        ? 'The ggsql kernel bundled with this extension cannot run on this system.'
        : 'This build of the ggsql extension does not include a kernel.';
    log(`${reason} No kernel installed on this machine could be used instead.`);

    const install = 'Install ggsql';
    const showOutput = 'Show Log';
    void vscode.window
        .showWarningMessage(`${reason} Install ggsql to run queries.`, install, showOutput)
        .then(choice => {
            if (choice === install) {
                void vscode.env.openExternal(vscode.Uri.parse(INSTALL_DOCS_URL));
            } else if (choice === showOutput) {
                showLog();
            }
        });
}

/**
 * Stable runtime identifier for a candidate.
 *
 * Hashing the path gives one identifier per installed kernel, which is what
 * Positron needs to keep runtime affinity and restorable sessions across
 * windows. The bundled kernel lives inside the versioned extension directory,
 * so its path changes on every extension update: it gets a fixed identifier
 * instead, or each update would look like a different runtime.
 */
function runtimeIdFor(candidate: KernelCandidate): string {
    if (candidate.source === 'Bundled') {
        return 'ggsql-bundled';
    }
    const pathHash = crypto.createHash('sha256').update(candidate.kernelPath).digest('hex').substring(0, 12);
    return `ggsql-${pathHash}`;
}

/**
 * Generate runtime metadata for a ggsql kernel candidate
 */
export function generateMetadata(
    context: vscode.ExtensionContext,
    candidate: KernelCandidate,
): positron.LanguageRuntimeMetadata {
    const version = context.extension.packageJSON.version as string;

    const iconPath = path.join(context.extensionPath, 'resources', 'ggsql-icon.svg');
    const base64Icon = fs.readFileSync(iconPath).toString('base64');

    return {
        runtimeId: runtimeIdFor(candidate),
        runtimePath: candidate.kernelPath,
        // The bundled kernel is the default, so it is just "ggsql". Only a
        // kernel the user went out of their way to use is worth qualifying.
        runtimeName: candidate.source === 'Bundled' ? 'ggsql' : `ggsql (${candidate.source})`,
        runtimeShortName: 'ggsql',
        runtimeVersion: version,
        runtimeSource: 'ggsql',
        languageId: 'ggsql',
        languageName: 'ggsql',
        languageVersion: version,
        base64EncodedIconSvg: base64Icon,
        startupBehavior: 'explicit' as positron.LanguageRuntimeStartupBehavior,
        sessionLocation: 'workspace' as positron.LanguageRuntimeSessionLocation,
        extraRuntimeData: {}
    };
}

/**
 * Create a Jupyter kernel spec for ggsql-jupyter
 *
 * @param kernelPath - Path to the ggsql-jupyter executable
 */
function createKernelSpec(kernelPath: string, readerUri?: string): JupyterKernelSpec {
    const argv = [kernelPath, '-f', '{connection_file}'];
    if (readerUri) {
        argv.push('--reader', readerUri);
    }

    return {
        argv,
        display_name: 'ggsql',
        language: 'ggsql',
        interrupt_mode: 'signal',
        env: { RUST_LOG: 'error' },
        kernel_protocol_version: '5.3',
    };
}

/**
 * Get the user-level Jupyter kernelspec directory for ggsql.
 */
function getUserJupyterKernelDir(): string {
    const homeDir = process.env.HOME || process.env.USERPROFILE || '';
    switch (process.platform) {
        case 'darwin':
            return path.join(homeDir, 'Library', 'Jupyter', 'kernels', 'ggsql');
        case 'win32':
            return path.join(
                process.env.APPDATA || path.join(homeDir, 'AppData', 'Roaming'),
                'jupyter', 'kernels', 'ggsql'
            );
        default:
            return path.join(homeDir, '.local', 'share', 'jupyter', 'kernels', 'ggsql');
    }
}

/**
 * Get the Jupyter kernelspec directory for ggsql.
 *
 * If a Python virtual environment or non-base conda environment is active
 * (detected via process.env), uses the environment-level path so that
 * Jupyter's `prefer_environment_over_user()` precedence applies naturally.
 * Otherwise falls back to the user-level kernelspec directory.
 */
function getJupyterKernelDir(): string {
    // Prefer virtual environment path when active. Jupyter gives these
    // precedence over user-level paths when running inside the same env.
    const virtualEnv = process.env.VIRTUAL_ENV;
    if (virtualEnv) {
        return path.join(virtualEnv, 'share', 'jupyter', 'kernels', 'ggsql');
    }

    const condaPrefix = process.env.CONDA_PREFIX;
    const condaEnv = process.env.CONDA_DEFAULT_ENV;
    if (condaPrefix && condaEnv && condaEnv !== 'base') {
        return path.join(condaPrefix, 'share', 'jupyter', 'kernels', 'ggsql');
    }

    return getUserJupyterKernelDir();
}

/**
 * Write a ggsql kernel.json to the given directory.
 *
 * Only writes if the content has changed to avoid unnecessary disk writes.
 */
function writeKernelJson(kernelDir: string, kernelPath: string): void {
    const kernelSpec = {
        argv: [kernelPath, '-f', '{connection_file}'],
        display_name: 'ggsql',
        language: 'ggsql',
        interrupt_mode: 'signal',
        env: { RUST_LOG: 'error' },
        metadata: { debugger: false }
    };

    const kernelJsonPath = path.join(kernelDir, 'kernel.json');
    const kernelSpecJson = JSON.stringify(kernelSpec, null, 2);

    try {
        const existing = fs.existsSync(kernelJsonPath)
            ? fs.readFileSync(kernelJsonPath, 'utf8')
            : null;

        if (existing !== kernelSpecJson) {
            fs.mkdirSync(kernelDir, { recursive: true });
            fs.writeFileSync(kernelJsonPath, kernelSpecJson);
            log(`Wrote ggsql kernel spec to ${kernelJsonPath}`);
        }
    } catch (err) {
        log(`Failed to write ggsql kernel spec: ${err}`);
    }
}

/**
 * Ensure a Jupyter kernel spec is installed so that external tools like
 * Quarto can discover ggsql. Called from session creation/restoration.
 *
 * Writes to the active virtualenv/conda env if detected, otherwise the
 * user-level kernelspec directory.
 */
function ensureKernelSpecInstalled(kernelPath: string): void {
    writeKernelJson(getJupyterKernelDir(), kernelPath);
}

/**
 * Create the dynamic state for a ggsql runtime session.
 *
 * @param sessionName The name Positron holds for the session, when restoring
 *   one. New sessions have no name yet and get the default.
 */
export function createDynState(sessionName?: string): positron.LanguageRuntimeDynState {
    return {
        inputPrompt: 'ggsql> ',
        continuationPrompt: '... ',
        sessionName: sessionName || 'ggsql',
    };
}

/**
 * Get the Positron Supervisor API, activating the extension if needed.
 *
 * The supervisor is a soft dependency: it is declared nowhere in
 * package.json, because an extensionDependencies entry would stop this
 * extension activating at all in VS Code, where the supervisor does not
 * exist. Awaiting activate() here gives the same ordering guarantee that a
 * declared dependency would.
 */
export async function getSupervisorApi(): Promise<PositronSupervisorApi> {
    const supervisorExt = vscode.extensions.getExtension<PositronSupervisorApi>(
        'positron.positron-supervisor'
    );

    if (!supervisorExt) {
        throw new Error('Positron Supervisor extension not found');
    }

    return supervisorExt.activate();
}

/**
 * Overrides for GgsqlRuntimeManager's environment.
 */
export interface RuntimeManagerOptions {
    /**
     * Directory the discovered kernel is advertised in, as a Jupyter kernel
     * spec. Defaults to the user-level Jupyter kernels directory.
     *
     * Discovery writes that spec as a side effect, so tests point this at a
     * temp directory: otherwise running discovery would repoint the real
     * kernelspec — the one Quarto and Jupyter resolve — at a test fixture.
     */
    kernelSpecDir?: string;

    /**
     * Liveness probe for the bundled kernel. Defaults to running it.
     *
     * Tests override it because a stand-in kernel cannot be a real executable
     * on every platform: a shell script named ggsql-jupyter.exe is not
     * something Windows can spawn.
     */
    probe?: KernelProbe;
}

/**
 * ggsql Language Runtime Manager
 *
 * Manages the lifecycle of ggsql runtime sessions in Positron.
 */
export class GgsqlRuntimeManager implements positron.LanguageRuntimeManager {
    /**
     * Run discovery on every window open rather than trusting Positron's
     * cross-window cache.
     *
     * ggsql runtimes are not marked cacheable: ggsql.kernelStrategy and
     * ggsql.kernelPath are workspace scoped, and the host kernels a machine
     * offers change as packages come and go. A cache hit would therefore
     * register a stale set of candidates on warm starts.
     */
    public readonly alwaysRediscover = true;

    private _context: vscode.ExtensionContext;
    private _kernelSpecDir: string;
    private _probe: KernelProbe;

    constructor(context: vscode.ExtensionContext, options: RuntimeManagerOptions = {}) {
        this._context = context;
        this._kernelSpecDir = options.kernelSpecDir ?? getUserJupyterKernelDir();
        this._probe = options.probe ?? probeKernel;
    }

    /**
     * Discover available ggsql runtimes.
     *
     * Returns all accessible ggsql kernel binaries found on the system.
     */
    discoverAllRuntimes(): AsyncGenerator<positron.LanguageRuntimeMetadata> {
        const context = this._context;
        const kernelSpecDir = this._kernelSpecDir;
        const probe = this._probe;

        const generator = async function* discoverGgsqlRuntimes() {
            log('Discovering ggsql runtimes...');

            const selection = discoverKernelPaths(context);
            log(`Found ${selection.candidates.length} kernel candidate(s)`);

            let registered = 0;
            let bundledRejected = false;

            async function* offer(candidates: KernelCandidate[]) {
                for (const candidate of candidates) {
                    if (await canRunKernel(context, candidate, probe)) {
                        // Write the kernel spec to the user kernelspec dir
                        // immediately so that Quarto/Jupyter can discover ggsql
                        // even if no session is ever started. The bundled kernel
                        // additionally needs this on every extension update, or
                        // the spec keeps pointing into the removed extension
                        // directory. Only a kernel that has proven it runs is
                        // advertised this way: the spec outlives the window, and
                        // Quarto has no discovery of its own to fall back on.
                        if (candidate.source === 'System' || candidate.source === 'Bundled') {
                            writeKernelJson(kernelSpecDir, candidate.kernelPath);
                        }

                        const metadata = generateMetadata(context, candidate);
                        log(`Yielding runtime: ${metadata.runtimeName} (${metadata.runtimeId}) at ${candidate.kernelPath}`);
                        registered++;
                        yield metadata;
                    } else {
                        if (candidate.source === 'Bundled') {
                            bundledRejected = true;
                        }
                        log(`Skipping unusable kernel (${candidate.source}): ${candidate.kernelPath}`);
                    }
                }
            }

            yield* offer(selection.candidates);

            if (registered === 0) {
                const fallback = selection.fallback();
                if (fallback.length > 0) {
                    log(`No usable kernel among the primary candidates; trying ${fallback.length} installed kernel(s)`);
                    yield* offer(fallback);
                }
            }

            if (registered === 0 && selection.strategy !== 'path') {
                // Under the path strategy the user named a binary that did not
                // work out, which the log already reports; there is nothing to
                // install that would change it.
                reportNoUsableKernel(context, bundledRejected);
            }

            log('Runtime discovery complete');
        };

        return generator();
    }

    /**
     * Get the recommended runtime for the workspace.
     *
     * Returns undefined - ggsql doesn't auto-start.
     */
    async recommendedWorkspaceRuntime(): Promise<positron.LanguageRuntimeMetadata | undefined> {
        return undefined;
    }

    /**
     * Create a new ggsql runtime session.
     */
    async createSession(
        runtimeMetadata: positron.LanguageRuntimeMetadata,
        sessionMetadata: positron.RuntimeSessionMetadata
    ): Promise<positron.LanguageRuntimeSession> {
        const supervisorApi = await getSupervisorApi();

        // Create the kernel spec using the runtime's kernel path
        const kernelSpec = createKernelSpec(runtimeMetadata.runtimePath);

        const dynState = createDynState();

        // Advertise this kernel to external tools (Quarto, Jupyter)
        ensureKernelSpecInstalled(runtimeMetadata.runtimePath);

        // Create the session using the supervisor
        const session = await supervisorApi.createSession(
            runtimeMetadata,
            sessionMetadata,
            kernelSpec,
            dynState
        );

        return session;
    }

    /**
     * Restore an existing ggsql runtime session.
     */
    async restoreSession(
        runtimeMetadata: positron.LanguageRuntimeMetadata,
        sessionMetadata: positron.RuntimeSessionMetadata,
        sessionName: string
    ): Promise<positron.LanguageRuntimeSession> {
        const supervisorApi = await getSupervisorApi();

        const dynState = createDynState(sessionName);

        // Re-advertise this kernel on restore
        ensureKernelSpecInstalled(runtimeMetadata.runtimePath);

        const session = await supervisorApi.restoreSession(
            runtimeMetadata,
            sessionMetadata,
            dynState
        );

        return session;
    }

    /**
     * Validate an existing session.
     */
    async validateSession(sessionId: string): Promise<boolean> {
        const supervisorApi = await getSupervisorApi();
        return supervisorApi.validateSession(sessionId);
    }
}
