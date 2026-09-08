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
 * A discovered ggsql-jupyter kernel candidate
 */
export interface KernelCandidate {
    /** Path to the ggsql-jupyter binary */
    kernelPath: string;
    /** Human-readable label for where this was found */
    source: KernelSource;
}

/** What a kernel reported about itself when it was run. */
export interface KernelInfo {
    /**
     * The version it printed for `--version`, absent when it printed none.
     * Kernels older than the flag do not answer it.
     */
    version?: string;
}

/** A candidate that can serve a session, with what it reported about itself. */
export interface RunnableKernel extends KernelCandidate, KernelInfo { }

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
 * Order the places a kernel can come from.
 *
 * Every one of them is offered, so a machine with several ggsql installs shows
 * them all and the user picks. Order decides which the picker lists first and,
 * for two paths naming one file, which occurrence survives deduplication.
 */
export function selectKernelCandidates(
    bundledPath: string | undefined,
    configuredPath: string | undefined,
    hostKernels: KernelCandidate[],
): KernelCandidate[] {
    const candidates: KernelCandidate[] = [];
    // A kernel the user named leads, ahead of the bundled default.
    if (configuredPath) {
        candidates.push({ kernelPath: configuredPath, source: 'Setting' });
    }
    if (bundledPath) {
        candidates.push({ kernelPath: bundledPath, source: 'Bundled' });
    }
    candidates.push(...hostKernels);
    return candidates;
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
export function discoverKernelPaths(context: vscode.ExtensionContext): KernelCandidate[] {
    const configuredPath = vscode.workspace
        .getConfiguration('ggsql')
        .get<string>('kernelPath', '')
        .trim();

    return dedupeCandidates(selectKernelCandidates(
        bundledKernelPath(context),
        configuredPath === '' ? undefined : resolveConfiguredPath(configuredPath),
        discoverHostKernels(),
    ));
}

/**
 * Stat a candidate, requiring a file this process can execute.
 *
 * A path that is not absolute is rejected: discovery absolutises every source
 * it can, so a bare name reaching here means the PATH lookup failed, and
 * accepting it would register a runtime that fails at session start.
 */
async function statKernel(kernelPath: string): Promise<fs.Stats | undefined> {
    if (!path.isAbsolute(kernelPath)) {
        return undefined;
    }
    try {
        const stats = await fs.promises.stat(kernelPath);
        if (!stats.isFile()) {
            return undefined;
        }
        await fs.promises.access(kernelPath, fs.constants.X_OK);
        return stats;
    } catch {
        return undefined;
    }
}

/** Whether a candidate is a file this process can execute. */
export async function isKernelAccessible(kernelPath: string): Promise<boolean> {
    return (await statKernel(kernelPath)) !== undefined;
}

/** How long the probe waits for the kernel to report its version. */
const KERNEL_PROBE_TIMEOUT_MS = 15000;

/** Where successful probes are remembered, to keep them to one per install. */
const PROBE_CACHE_KEY = 'ggsql.kernelProbes';

/** Where the dead-end notice records the version it has already reported. */
const NO_KERNEL_NOTICE_KEY = 'ggsql.noUsableKernelNotice';

/** Install instructions offered when no kernel on this machine can run. */
const INSTALL_DOCS_URL = 'https://ggsql.org/get_started/installation.html';

/** The version in the kernel's `--version` output, as in `ggsql-jupyter 0.4.1`. */
const VERSION_PATTERN = /\b(\d+\.\d+\.\d+\S*)/;

/**
 * Runs a kernel binary and reports what it said about itself, or undefined when
 * it did not run.
 */
export type KernelProbe = (kernelPath: string) => Promise<KernelInfo | undefined>;

/** What running the kernel with one argument came to. */
type RunOutcome =
    /** It was exec'd, and either exited zero or did not. */
    | { exec: true; ok: boolean; stdout: string; reason: string }
    /** It never started, so nothing can be concluded from its output. */
    | { exec: false; reason: string };

/**
 * Run the kernel with a single argument and report how far it got.
 *
 * The distinction that matters is between a binary that never started and one
 * that ran and exited non-zero: only the second says anything about the
 * argument it was given.
 */
function runKernel(kernelPath: string, arg: string): Promise<RunOutcome> {
    return new Promise(resolve => {
        // On Windows a file that is not a valid executable fails the
        // CreateProcess call itself, which Node surfaces as a synchronous
        // throw from execFile (`spawn UNKNOWN`) rather than a callback error.
        try {
            cp.execFile(
                kernelPath,
                [arg],
                { timeout: KERNEL_PROBE_TIMEOUT_MS, windowsHide: true },
                (err, stdout) => {
                    if (!err) {
                        resolve({ exec: true, ok: true, stdout, reason: '' });
                    } else if (typeof err.code === 'number') {
                        // An exit status, rather than one of the errno strings
                        // a failure to spawn reports, so the binary did start.
                        resolve({ exec: true, ok: false, stdout, reason: err.message });
                    } else {
                        resolve({ exec: false, reason: err.message });
                    }
                },
            );
        } catch (err) {
            resolve({ exec: false, reason: (err as Error).message });
        }
    });
}

/**
 * Run the kernel and read the version it reports.
 */
export async function probeKernel(kernelPath: string): Promise<KernelInfo | undefined> {
    const reportedVersion = await runKernel(kernelPath, '--version');
    if (!reportedVersion.exec) {
        log(`Kernel probe failed for ${kernelPath}: ${reportedVersion.reason}`);
        return undefined;
    }
    if (reportedVersion.ok) {
        const version = VERSION_PATTERN.exec(reportedVersion.stdout)?.[1];
        if (!version) {
            log(`Kernel at ${kernelPath} reported no version: ${reportedVersion.stdout.trim()}`);
        }
        return { version };
    }

    // A kernel released before `--version` rejects the argument, which says
    // nothing about whether it runs: one killed by the dynamic linker for want
    // of a shared library exits non-zero in exactly the same way. `--help` is
    // the question every version answers, so let that settle it rather than
    // reading an exit status or matching the wording of an error.
    const help = await runKernel(kernelPath, '--help');
    if (!help.exec || !help.ok) {
        log(`Kernel probe failed for ${kernelPath}: ${help.reason}`);
        return undefined;
    }
    log(`Kernel at ${kernelPath} predates \`--version\`, so is offered without one`);
    return { version: undefined };
}

interface ProbeCacheEntry extends KernelInfo {
    mtimeMs: number;
    size: number;
}

/**
 * The successful probes remembered across windows, keyed by kernel path.
 *
 * A hit saves a spawn per kernel per window open. The file's mtime and size are
 * part of the entry, so an install upgraded in place is probed again rather than
 * reporting the version it used to have. A failure is not remembered: it is
 * cheap to repeat, and a host that gains the shared libraries the bundled kernel
 * needs should start working without waiting for an extension update.
 */
class ProbeCache {
    private readonly stored: Record<string, ProbeCacheEntry>;
    private readonly used: Record<string, ProbeCacheEntry> = {};

    constructor(private readonly context: vscode.ExtensionContext) {
        this.stored = context.globalState.get<Record<string, ProbeCacheEntry>>(PROBE_CACHE_KEY) ?? {};
    }

    async run(kernelPath: string, stats: fs.Stats, probe: KernelProbe): Promise<KernelInfo | undefined> {
        const cached = this.stored[kernelPath];
        if (cached && cached.mtimeMs === stats.mtimeMs && cached.size === stats.size) {
            this.used[kernelPath] = cached;
            return { version: cached.version };
        }

        const info = await probe(kernelPath);
        if (info) {
            this.used[kernelPath] = { version: info.version, mtimeMs: stats.mtimeMs, size: stats.size };
        }
        return info;
    }

    /**
     * Persist the probes this pass used, dropping any kernel it did not see —
     * without which the paths of every superseded extension version would
     * accumulate.
     */
    async flush(): Promise<void> {
        if (JSON.stringify(this.used) !== JSON.stringify(this.stored)) {
            await this.context.globalState.update(PROBE_CACHE_KEY, this.used);
        }
    }
}

/**
 * Decide whether a candidate can serve a session, and read its version.
 */
async function inspectKernel(
    candidate: KernelCandidate,
    cache: ProbeCache,
    probe: KernelProbe,
): Promise<RunnableKernel | undefined> {
    const stats = await statKernel(candidate.kernelPath);
    if (!stats) {
        return undefined;
    }

    const info = await cache.run(candidate.kernelPath, stats, probe);
    if (info) {
        return { ...candidate, ...info };
    }
    // Only the bundled kernel has to prove it runs. It is built for this
    // platform but not for every system it can be installed on, and that
    // failure is invisible to the filesystem. A kernel the user installed is
    // their own business, so it is still offered when it would not start.
    return candidate.source === 'Bundled' ? undefined : { ...candidate };
}

/**
 * Tell the user that nothing on this machine can run ggsql queries.
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
 * windows.
 */
const BUNDLED_RUNTIME_ID = 'ggsql-bundled';

function runtimeIdFor(candidate: KernelCandidate): string {
    if (candidate.source === 'Bundled') {
        return BUNDLED_RUNTIME_ID;
    }
    const pathHash = crypto.createHash('sha256').update(candidate.kernelPath).digest('hex').substring(0, 12);
    return `ggsql-${pathHash}`;
}

/**
 * Generate runtime metadata for a ggsql kernel
 */
export function generateMetadata(
    context: vscode.ExtensionContext,
    kernel: RunnableKernel,
): positron.LanguageRuntimeMetadata {
    // The kernel is what runs the query, so its version is the one to show. A
    // kernel too old to report one falls back to the extension's.
    const version = kernel.version ?? context.extension.packageJSON.version as string;

    const iconPath = path.join(context.extensionPath, 'resources', 'ggsql-icon.svg');
    const base64Icon = fs.readFileSync(iconPath).toString('base64');

    // As Positron's own runtimes are named: the language, its version, and a
    // qualifier saying which install this is. The bundled kernel is the default,
    // so it carries no qualifier.
    const named = kernel.version ? `ggsql ${kernel.version}` : 'ggsql';
    const runtimeName = kernel.source === 'Bundled' ? named : `${named} (${kernel.source})`;

    return {
        runtimeId: runtimeIdFor(kernel),
        runtimePath: kernel.kernelPath,
        runtimeName,
        runtimeShortName: 'ggsql',
        runtimeVersion: version,
        runtimeSource: 'ggsql',
        languageId: 'ggsql',
        languageName: 'ggsql',
        languageVersion: version,
        base64EncodedIconSvg: base64Icon,
        startupBehavior: 'explicit' as positron.LanguageRuntimeStartupBehavior,
        sessionLocation: 'workspace' as positron.LanguageRuntimeSessionLocation,
        extraRuntimeData: {},
        // Without this the frontend never tells the kernel how large the Plots
        // pane is, which it needs only to pre-render a new plot at the right
        // size; every other render carries its own size. A string cast because
        // `positron` is imported as a type — as with `startupBehavior` above.
        uiSubscriptions: ['did_change_plots_render_settings' as positron.UiRuntimeNotifications]
    };
}

/**
 * Create a Jupyter kernel spec for ggsql-jupyter
 *
 * `--session-mode` tells the kernel where its output goes, which it cannot work
 * out for itself: a plot comm always lands in the Plots pane, so a notebook
 * session using one would leave its cell empty. Left off, the kernel guesses
 * from the session id — which is why `writeKernelJson` does not pass it.
 *
 * @param kernelPath - Path to the ggsql-jupyter executable
 * @param readerUri - Data source the kernel should open, if not the default
 * @param sessionMode - What kind of session this is, when known
 */
function createKernelSpec(
    kernelPath: string,
    readerUri?: string,
    sessionMode?: positron.LanguageRuntimeSessionMode
): JupyterKernelSpec {
    const argv = [kernelPath, '-f', '{connection_file}'];
    if (readerUri) {
        argv.push('--reader', readerUri);
    }
    if (sessionMode) {
        // The enum's values are already the kernel's spelling (`console`,
        // `notebook`, `background`), so there is nothing to translate.
        argv.push('--session-mode', sessionMode);
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
     * How a kernel is run to read its version. Defaults to running it.
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

            const candidates = discoverKernelPaths(context);
            log(`Found ${candidates.length} kernel candidate(s)`);

            const cache = new ProbeCache(context);
            let registered = 0;
            let bundledRejected = false;

            for (const candidate of candidates) {
                const kernel = await inspectKernel(candidate, cache, probe);
                if (!kernel) {
                    if (candidate.source === 'Bundled') {
                        bundledRejected = true;
                    }
                    log(`Skipping unusable kernel (${candidate.source}): ${candidate.kernelPath}`);
                    continue;
                }

                // Advertise the leading kernel as a Jupyter kernel spec, so that
                // Quarto and Jupyter can discover ggsql even if no session is
                // ever started, and so that the spec stops pointing into an
                // extension directory an update has removed. A kernel found as a
                // kernelspec is already advertised where Jupyter looks.
                if (registered === 0 && kernel.source !== 'Jupyter') {
                    writeKernelJson(kernelSpecDir, kernel.kernelPath);
                }

                const metadata = generateMetadata(context, kernel);
                log(`Yielding runtime: ${metadata.runtimeName} (${metadata.runtimeId}) at ${kernel.kernelPath}`);
                registered++;
                yield metadata;
            }

            await cache.flush();

            if (registered === 0) {
                reportNoUsableKernel(context, bundledRejected);
            }

            log('Runtime discovery complete');
        };

        return generator();
    }

    /**
     * Refresh metadata Positron stored for this workspace.
     *
     * The bundled kernel's `runtimePath` names the versioned extension
     * directory, which an update removes; its `runtimeId` is fixed precisely so
     * that the runtime survives, which it only does if the path is regenerated
     * here. A kernel that has since been uninstalled has no candidate to match,
     * and rejecting it is how Positron learns to drop it.
     */
    async validateMetadata(
        metadata: positron.LanguageRuntimeMetadata,
    ): Promise<positron.LanguageRuntimeMetadata> {
        // Deliberately not flushed: this walks only as far as the match, so
        // persisting the pass would drop the probes discovery cached.
        const cache = new ProbeCache(this._context);

        for (const candidate of discoverKernelPaths(this._context)) {
            if (runtimeIdFor(candidate) !== metadata.runtimeId) {
                continue;
            }
            const kernel = await inspectKernel(candidate, cache, this._probe);
            if (kernel) {
                return generateMetadata(this._context, kernel);
            }
            break;
        }

        throw new Error(`No usable ggsql kernel for runtime ${metadata.runtimeId}`);
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
        const kernelSpec = createKernelSpec(
            runtimeMetadata.runtimePath,
            undefined,
            sessionMetadata.sessionMode
        );

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

        // No kernel spec here on purpose: the supervisor replays the argv the
        // session was created with, and a session's mode never changes, so the
        // restored kernel keeps the `--session-mode` it started with.
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
