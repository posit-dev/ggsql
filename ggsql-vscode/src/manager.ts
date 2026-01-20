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
import type * as positron from '@posit-dev/positron';
import type { JupyterKernelSpec, JupyterSession, JupyterKernel, PositronSupervisorApi } from './types';
import { log } from './extension';

/**
 * Get the path to the ggsql-jupyter kernel executable
 *
 * Checks in order:
 * 1. Configured path in settings
 * 2. Jupyter kernelspec location (user)
 * 3. Jupyter kernelspec location (system)
 * 4. Fall back to PATH
 */
function getKernelPath(): string {
    const config = vscode.workspace.getConfiguration('ggsql');
    const configuredPath = config.get<string>('kernelPath', '');

    if (configuredPath && configuredPath.trim() !== '') {
        return configuredPath;
    }

    // Check Jupyter kernelspec locations
    const homeDir = process.env.HOME || process.env.USERPROFILE || '';
    const kernelName = 'ggsql';
    const binaryName = process.platform === 'win32' ? 'ggsql-jupyter.exe' : 'ggsql-jupyter';

    // Common Jupyter kernel locations
    const possiblePaths = [
        // User kernelspec (macOS/Linux)
        path.join(homeDir, 'Library', 'Jupyter', 'kernels', kernelName, binaryName),
        // User kernelspec (Linux)
        path.join(homeDir, '.local', 'share', 'jupyter', 'kernels', kernelName, binaryName),
        // System kernelspec (macOS)
        path.join('/usr', 'local', 'share', 'jupyter', 'kernels', kernelName, binaryName),
        // System kernelspec (Linux)
        path.join('/usr', 'share', 'jupyter', 'kernels', kernelName, binaryName),
    ];

    for (const kernelPath of possiblePaths) {
        if (fs.existsSync(kernelPath)) {
            log(`Found kernel at: ${kernelPath}`);
            return kernelPath;
        }
    }

    // Fall back to PATH
    log('Kernel not found in standard locations, falling back to PATH');
    return 'ggsql-jupyter';
}

/**
 * Check if the kernel executable exists and is accessible
 */
async function isKernelAvailable(): Promise<boolean> {
    const kernelPath = getKernelPath();

    // If it's an absolute path, check if the file exists
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
 * Generate runtime metadata for ggsql
 */
function generateMetadata(
    context: vscode.ExtensionContext
): positron.LanguageRuntimeMetadata {
    const kernelPath = getKernelPath();
    const version = context.extension.packageJSON.version as string;

    let base64Icon: string;
    const iconPath = path.join(context.extensionPath, 'resources', 'ggsql-icon.svg');
    base64Icon = fs.readFileSync(iconPath).toString('base64');

    return {
        runtimeId: 'ggsql-jupyter',
        runtimePath: kernelPath,
        runtimeName: `ggsql ${version}`,
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
 * Uses the startKernel callback to manually start the kernel process,
 * giving us more control over the launch process.
 *
 * @param workspacePath - Optional workspace path to use as the kernel's working directory
 */
function createKernelSpec(workspacePath?: string): JupyterKernelSpec {
    const kernelPath = getKernelPath();

    return {
        // argv is empty when using startKernel callback
        argv: [],
        display_name: 'ggsql',
        language: 'ggsql',
        interrupt_mode: 'signal',
        env: {},
        kernel_protocol_version: '5.3',
        startKernel: async (session: JupyterSession, kernel: JupyterKernel) => {
            kernel.log(`Starting ggsql kernel with connection file: ${session.state.connectionFile}`);
            kernel.log(`Working directory: ${workspacePath ?? 'inherited from parent'}`);

            const connectionFile = session.state.connectionFile;

            // Start the kernel process
            const proc = cp.spawn(kernelPath, ['-f', connectionFile], {
                stdio: ['ignore', 'pipe', 'pipe'],
                detached: false,
                cwd: workspacePath
            });

            // Log stdout and stderr
            proc.stdout?.on('data', (data) => {
                const msg = data.toString();
                kernel.log(msg);
            });

            proc.stderr?.on('data', (data) => {
                const msg = data.toString();
                kernel.log(msg);
            });

            proc.on('error', (err) => {
                const msg = `Failed to start kernel: ${err.message}`;
                kernel.log(msg);
                throw new Error(msg);
            });

            proc.on('exit', (code, signal) => {
                kernel.log(`Kernel exited (code: ${code}, signal: ${signal})`);
            });

            // Wait a moment for the kernel to start binding sockets
            await new Promise(resolve => setTimeout(resolve, 1000));

            // Check if process is still running
            if (proc.exitCode !== null) {
                throw new Error(`Kernel process exited immediately with code ${proc.exitCode}`);
            }

            kernel.log('Connecting to kernel session...');

            // Connect to the session
            await kernel.connectToSession(session);

            kernel.log('Connected to ggsql kernel');
        }
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
     * Returns a single ggsql runtime if the kernel is available.
     */
    discoverAllRuntimes(): AsyncGenerator<positron.LanguageRuntimeMetadata> {
        const context = this._context;

        const generator = async function* discoverGgsqlRuntimes() {
            log('Discovering ggsql runtimes...');

            // Check if the kernel is available
            const available = await isKernelAvailable();
            log(`Kernel available: ${available}`);

            if (available) {
                const metadata = generateMetadata(context);
                log(`Yielding runtime: ${metadata.runtimeName} (${metadata.runtimeId})`);
                yield metadata;
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

        // Get workspace path for kernel's working directory
        const workspaceFolders = vscode.workspace.workspaceFolders;
        const workspacePath = workspaceFolders?.[0]?.uri.fsPath;

        // Create the kernel spec
        const kernelSpec = createKernelSpec(workspacePath);

        // Create the dynamic state
        const dynState: positron.LanguageRuntimeDynState = {
            inputPrompt: 'ggsql> ',
            continuationPrompt: '... ',
            sessionName: 'ggsql'
        };

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

        const dynState: positron.LanguageRuntimeDynState = {
            inputPrompt: 'ggsql> ',
            continuationPrompt: '... ',
            sessionName: 'ggsql'
        };

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
