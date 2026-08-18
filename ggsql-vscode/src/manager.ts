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
import { log } from './extension';

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
): KernelCandidate[] {
    const bundled: KernelCandidate[] = bundledPath
        ? [{ kernelPath: bundledPath, source: 'Bundled' }]
        : [];

    if (strategy === 'path') {
        if (configuredPath) {
            return [{ kernelPath: configuredPath, source: 'Setting' }];
        }
        // Nothing to point at. Treat it as the default rather than registering
        // no runtime at all.
        log('ggsql.kernelStrategy is "path" but ggsql.kernelPath is empty; using the bundled kernel');
    } else if (strategy === 'environment') {
        return [...hostKernels(), ...bundled];
    }

    // A build that carries no kernel still looks for a host install, or it
    // would offer nothing at all.
    return bundled.length > 0 ? bundled : hostKernels();
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
    const config = vscode.workspace.getConfiguration('ggsql');
    const strategy = resolveKernelStrategy(config);
    log(`Kernel strategy: ${strategy}`);

    const configuredPath = config.get<string>('kernelPath', '').trim();

    return dedupeCandidates(selectKernelCandidates(
        strategy,
        bundledKernelPath(context),
        configuredPath === '' ? undefined : resolveConfiguredPath(configuredPath),
        discoverHostKernels,
    ));
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

    constructor(context: vscode.ExtensionContext, options: RuntimeManagerOptions = {}) {
        this._context = context;
        this._kernelSpecDir = options.kernelSpecDir ?? getUserJupyterKernelDir();
    }

    /**
     * Discover available ggsql runtimes.
     *
     * Returns all accessible ggsql kernel binaries found on the system.
     */
    discoverAllRuntimes(): AsyncGenerator<positron.LanguageRuntimeMetadata> {
        const context = this._context;
        const kernelSpecDir = this._kernelSpecDir;

        const generator = async function* discoverGgsqlRuntimes() {
            log('Discovering ggsql runtimes...');

            const candidates = discoverKernelPaths(context);
            log(`Found ${candidates.length} kernel candidate(s)`);

            for (const candidate of candidates) {
                const accessible = await isKernelAccessible(candidate.kernelPath);
                if (accessible) {
                    // Write the kernel spec to the user kernelspec dir
                    // immediately so that Quarto/Jupyter can discover ggsql
                    // even if no session is ever started. The bundled kernel
                    // additionally needs this on every extension update, or the
                    // spec keeps pointing into the removed extension directory.
                    if (candidate.source === 'System' || candidate.source === 'Bundled') {
                        writeKernelJson(kernelSpecDir, candidate.kernelPath);
                    }

                    const metadata = generateMetadata(context, candidate);
                    log(`Yielding runtime: ${metadata.runtimeName} (${metadata.runtimeId}) at ${candidate.kernelPath}`);
                    yield metadata;
                } else {
                    log(`Skipping inaccessible kernel (${candidate.source}): ${candidate.kernelPath}`);
                }
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
