/*
 * Type definitions for the Positron Supervisor API
 *
 * These types define the interfaces used to wrap Jupyter kernels
 * as Positron language runtimes.
 */

import type * as positron from '@posit-dev/positron';

/**
 * State information for a Jupyter session
 */
export interface JupyterSessionState {
    /** The Jupyter session identifier */
    sessionId: string;

    /** The log file the kernel is writing to */
    logFile: string;

    /** The profile file the kernel is writing to */
    profileFile?: string;

    /** The connection file specifying ZeroMQ ports, signing keys, etc. */
    connectionFile: string;

    /** The ID of the kernel's process, or 0 if not running */
    processId: number;
}

/**
 * A Jupyter session
 */
export interface JupyterSession {
    readonly state: JupyterSessionState;
}

/**
 * Interface for interacting with a Jupyter kernel
 */
export interface JupyterKernel {
    connectToSession(session: JupyterSession): Promise<void>;
    log(msg: string): void;
}

/**
 * Jupyter kernel specification
 *
 * See: https://jupyter-client.readthedocs.io/en/stable/kernels.html#kernel-specs
 */
export interface JupyterKernelSpec {
    /** Command used to start the kernel and command line arguments */
    argv: string[];

    /** The kernel's display name */
    display_name: string;

    /** The language the kernel executes */
    language: string;

    /** Interrupt mode (signal or message) */
    interrupt_mode?: 'signal' | 'message';

    /** Environment variables to set when starting the kernel */
    env?: NodeJS.ProcessEnv;

    /**
     * The Jupyter protocol version to use.
     * When >= 5.5, uses handshake to negotiate ports (JEP 66)
     */
    kernel_protocol_version: string;

    /** Optional preflight command to run before starting the kernel */
    startup_command?: string;

    /**
     * Optional function to start the kernel.
     * If provided, argv is ignored.
     */
    startKernel?: (session: JupyterSession, kernel: JupyterKernel) => Promise<void>;
}

/**
 * A language runtime session that wraps a Jupyter kernel
 */
export interface JupyterLanguageRuntimeSession extends positron.LanguageRuntimeSession {
    /** Log a message to the kernel's output channel */
    emitJupyterLog(message: string, logLevel?: number): void;

    /** Show an output channel */
    showOutput(channel?: positron.LanguageRuntimeSessionChannel): void;

    /** List available output channels */
    listOutputChannels(): positron.LanguageRuntimeSessionChannel[];

    /** Call a method on the kernel */
    callMethod(method: string, ...args: unknown[]): Promise<unknown>;

    /** Get the kernel log file path */
    getKernelLogFile(): string;
}

/**
 * The Positron Supervisor API
 */
export interface PositronSupervisorApi {
    /**
     * Create a session for a Jupyter-compatible kernel
     */
    createSession(
        runtimeMetadata: positron.LanguageRuntimeMetadata,
        sessionMetadata: positron.RuntimeSessionMetadata,
        kernel: JupyterKernelSpec,
        dynState: positron.LanguageRuntimeDynState,
        extra?: JupyterKernelExtra
    ): Promise<JupyterLanguageRuntimeSession>;

    /**
     * Validate an existing session
     */
    validateSession(sessionId: string): Promise<boolean>;

    /**
     * Restore a session for a Jupyter-compatible kernel
     */
    restoreSession(
        runtimeMetadata: positron.LanguageRuntimeMetadata,
        sessionMetadata: positron.RuntimeSessionMetadata,
        dynState: positron.LanguageRuntimeDynState
    ): Promise<JupyterLanguageRuntimeSession>;
}

/**
 * Extra functionality for kernels
 */
export interface JupyterKernelExtra {
    attachOnStartup?: {
        init: (args: string[]) => void;
        attach: () => Promise<void>;
    };
    sleepOnStartup?: {
        init: (args: string[], delay: number) => void;
    };
}
