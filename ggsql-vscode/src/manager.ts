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
type KernelSource = 'Setting' | 'Jupyter' | 'System' | 'Path';

/**
 * A discovered ggsql-jupyter kernel candidate
 */
interface KernelCandidate {
    /** Absolute path to the ggsql-jupyter binary (or bare name for PATH fallback) */
    kernelPath: string;
    /** Human-readable label for where this was found */
    source: KernelSource;
}

/**
 * Try to resolve a binary name to its absolute path via the system PATH.
 * Returns the original value if resolution fails or the path is already absolute.
 */
function resolveToAbsolutePath(binaryPath: string): string {
    if (path.isAbsolute(binaryPath)) {
        return binaryPath;
    }
    try {
        const cmd = process.platform === 'win32' ? 'where' : 'which';
        const resolved = cp.execFileSync(cmd, [binaryPath], {
            encoding: 'utf8',
            timeout: 5000,
        }).trim().split(/\r?\n/)[0];
        if (resolved && path.isAbsolute(resolved)) {
            log(`Resolved '${binaryPath}' to '${resolved}'`);
            return resolved;
        }
    } catch {
        log(`Could not resolve '${binaryPath}' to absolute path, using as-is`);
    }
    return binaryPath;
}

/**
 * Discover all available ggsql-jupyter kernel binaries
 *
 * Checks in priority order:
 * 1. Configured path in settings
 * 2. Jupyter kernelspec locations (user and system)
 * 3. Cargo-packager install locations
 * 4. Fall back to PATH
 *
 * Returns deduplicated candidates, keeping the highest-priority occurrence.
 */
function discoverKernelPaths(): KernelCandidate[] {
    const candidates: KernelCandidate[] = [];
    const binaryName = process.platform === 'win32' ? 'ggsql-jupyter.exe' : 'ggsql-jupyter';

    // 1. User-configured setting (highest priority)
    const config = vscode.workspace.getConfiguration('ggsql');
    const configuredPath = config.get<string>('kernelPath', '');
    if (configuredPath && configuredPath.trim() !== '') {
        candidates.push({ kernelPath: configuredPath, source: 'Setting' });
    }

    // 2. Jupyter kernelspec locations
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

    // 3. Cargo-packager install locations
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

    // 4. PATH fallback (last resort)
    candidates.push({ kernelPath: resolveToAbsolutePath(binaryName), source: 'Path' });

    // Deduplicate by resolved absolute path
    const seen = new Set<string>();
    const deduped: KernelCandidate[] = [];
    for (const candidate of candidates) {
        if (!path.isAbsolute(candidate.kernelPath)) {
            // Non-absolute paths (PATH fallback) can't be deduplicated
            deduped.push(candidate);
            continue;
        }
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
 * Check if a kernel executable exists and is accessible
 */
async function isKernelAccessible(kernelPath: string): Promise<boolean> {
    if (path.isAbsolute(kernelPath)) {
        try {
            await fs.promises.access(kernelPath, fs.constants.X_OK);
            return true;
        } catch {
            return false;
        }
    }

    // For non-absolute paths (relying on PATH), always return true
    // and let the actual kernel startup fail with a proper error message
    return true;
}

/**
 * Generate runtime metadata for a ggsql kernel candidate
 */
function generateMetadata(
    context: vscode.ExtensionContext,
    candidate: KernelCandidate,
): positron.LanguageRuntimeMetadata {
    const version = context.extension.packageJSON.version as string;

    const iconPath = path.join(context.extensionPath, 'resources', 'ggsql-icon.svg');
    const base64Icon = fs.readFileSync(iconPath).toString('base64');

    const pathHash = crypto.createHash('sha256').update(candidate.kernelPath).digest('hex').substring(0, 12);
    return {
        runtimeId: `ggsql-${pathHash}`,
        runtimePath: candidate.kernelPath,
        runtimeName: `ggsql (${candidate.source})`,
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
 */
function createDynState(): positron.LanguageRuntimeDynState {
    return {
        inputPrompt: 'ggsql> ',
        continuationPrompt: '... ',
        sessionName: 'ggsql',
    };
}

/**
 * ggsql Language Runtime Manager
 *
 * Manages the lifecycle of ggsql runtime sessions in Positron.
 */
export class GgsqlRuntimeManager implements positron.LanguageRuntimeManager {
    private _context: vscode.ExtensionContext;
    private _sessions: Map<string, positron.LanguageRuntimeSession> = new Map();

    constructor(context: vscode.ExtensionContext) {
        this._context = context;
    }

    /**
     * Discover available ggsql runtimes.
     *
     * Returns all accessible ggsql kernel binaries found on the system.
     */
    discoverAllRuntimes(): AsyncGenerator<positron.LanguageRuntimeMetadata> {
        const context = this._context;

        const generator = async function* discoverGgsqlRuntimes() {
            log('Discovering ggsql runtimes...');

            const candidates = discoverKernelPaths();
            log(`Found ${candidates.length} kernel candidate(s)`);

            for (const candidate of candidates) {
                const accessible = await isKernelAccessible(candidate.kernelPath);
                if (accessible) {
                    // When a system install is found, write the kernel spec to
                    // the user kernelspec dir immediately so that Quarto/Jupyter
                    // can discover ggsql even if no session is ever started.
                    if (candidate.source === 'System') {
                        writeKernelJson(getUserJupyterKernelDir(), candidate.kernelPath);
                    }

                    const metadata = generateMetadata(context, candidate);
                    log(`Yielding runtime: ${metadata.runtimeName} (${metadata.runtimeId}) at ${candidate.kernelPath}`);
                    yield metadata;
                } else {
                    log(`Skipping inaccessible kernel: ${candidate.kernelPath}`);
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
        // Get the Positron Supervisor extension
        const supervisorExt = vscode.extensions.getExtension<PositronSupervisorApi>(
            'positron.positron-supervisor'
        );

        if (!supervisorExt) {
            throw new Error('Positron Supervisor extension not found');
        }

        // Ensure the extension is activated
        const supervisorApi = await supervisorExt.activate();

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

        // Track the session
        this._sessions.set(sessionMetadata.sessionId, session);

        // Remove from tracking when session ends
        session.onDidEndSession(() => {
            this._sessions.delete(sessionMetadata.sessionId);
        });

        return session;
    }

    /**
     * Restore an existing ggsql runtime session.
     */
    async restoreSession(
        runtimeMetadata: positron.LanguageRuntimeMetadata,
        sessionMetadata: positron.RuntimeSessionMetadata
    ): Promise<positron.LanguageRuntimeSession> {
        // Get the Positron Supervisor extension
        const supervisorExt = vscode.extensions.getExtension<PositronSupervisorApi>(
            'positron.positron-supervisor'
        );

        if (!supervisorExt) {
            throw new Error('Positron Supervisor extension not found');
        }

        const supervisorApi = await supervisorExt.activate();

        const dynState = createDynState();

        // Re-advertise this kernel on restore
        ensureKernelSpecInstalled(runtimeMetadata.runtimePath);

        const session = await supervisorApi.restoreSession(
            runtimeMetadata,
            sessionMetadata,
            dynState
        );

        this._sessions.set(sessionMetadata.sessionId, session);

        session.onDidEndSession(() => {
            this._sessions.delete(sessionMetadata.sessionId);
        });

        return session;
    }

    /**
     * Validate an existing session.
     */
    async validateSession(sessionId: string): Promise<boolean> {
        const supervisorExt = vscode.extensions.getExtension<PositronSupervisorApi>(
            'positron.positron-supervisor'
        );

        if (!supervisorExt) {
            return false;
        }

        const supervisorApi = await supervisorExt.activate();
        return supervisorApi.validateSession(sessionId);
    }
}
