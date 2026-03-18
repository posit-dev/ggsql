/*
 * Connection Drivers for Positron's Connections pane
 *
 * Registers drivers that let users create database connections via the
 * "New Connection" dialog. Each driver generates a `-- @connect:` meta-command
 * that the ggsql-jupyter kernel interprets to switch readers.
 */

import * as os from 'os';
import * as path from 'path';
import * as fs from 'fs';
import * as toml from 'toml';
import type * as positron from '@posit-dev/positron';

type PositronApi = positron.PositronApi;
type ConnectionsDriverMetadata = positron.ConnectionsDriverMetadata & { description?: string };

/**
 * Create the set of ggsql connection drivers to register with Positron.
 */
export function createConnectionDrivers(
    positronApi: PositronApi
): positron.ConnectionsDriver[] {
    return [
        createDuckDBDriver(positronApi),
        createSnowflakeDefaultDriver(positronApi),
        createSnowflakePasswordDriver(positronApi),
        createSnowflakeSSODriver(positronApi),
        createSnowflakePATDriver(positronApi),
        createOdbcDriver(positronApi),
    ];
}

// ============================================================================
// DuckDB
// ============================================================================

/**
 * DuckDB connection driver.
 *
 * Inputs: optional database file path (empty = in-memory).
 */
function createDuckDBDriver(
    positronApi: PositronApi
): positron.ConnectionsDriver {
    return {
        driverId: 'ggsql-duckdb',
        metadata: {
            languageId: 'ggsql',
            name: 'DuckDB',
            inputs: [
                {
                    id: 'database',
                    label: 'Database',
                    type: 'string',
                    value: '',
                },
            ],
        },
        generateCode: (inputs) => {
            const db = inputs.find((i) => i.id === 'database')?.value?.trim();
            if (!db) {
                return '-- @connect: duckdb://memory';
            }
            return `-- @connect: duckdb://${db}`;
        },
        connect: async (code: string) => {
            await positronApi.runtime.executeCode('ggsql', code, false);
        },
    };
}

// ============================================================================
// Snowflake — shared helpers
// ============================================================================

interface SnowflakeConnectionEntry {
    name: string;
    account?: string;
}

/**
 * Find the Snowflake connections.toml file, checking standard locations.
 */
function findSnowflakeConnectionsToml(): string | undefined {
    const candidates: string[] = [];

    // 1. $SNOWFLAKE_HOME/connections.toml
    const snowflakeHome = process.env.SNOWFLAKE_HOME;
    if (snowflakeHome) {
        candidates.push(path.join(snowflakeHome, 'connections.toml'));
    }

    // 2. ~/.snowflake/connections.toml
    const home = os.homedir();
    candidates.push(path.join(home, '.snowflake', 'connections.toml'));

    // 3. Platform-specific paths
    if (process.platform === 'darwin') {
        candidates.push(
            path.join(home, 'Library', 'Application Support', 'snowflake', 'connections.toml')
        );
    } else if (process.platform === 'linux') {
        const xdgConfig = process.env.XDG_CONFIG_HOME || path.join(home, '.config');
        candidates.push(path.join(xdgConfig, 'snowflake', 'connections.toml'));
    } else if (process.platform === 'win32') {
        candidates.push(
            path.join(home, 'AppData', 'Local', 'snowflake', 'connections.toml')
        );
    }

    for (const candidate of candidates) {
        if (fs.existsSync(candidate)) {
            return candidate;
        }
    }
    return undefined;
}

/**
 * Read Snowflake connection entries from connections.toml.
 */
function readSnowflakeConnections(): {
    connections: SnowflakeConnectionEntry[];
    defaultConnection?: string;
} {
    const tomlPath = findSnowflakeConnectionsToml();
    if (!tomlPath) {
        return { connections: [] };
    }

    try {
        const content = fs.readFileSync(tomlPath, 'utf-8');
        const parsed = toml.parse(content);

        const defaultConnection =
            process.env.SNOWFLAKE_DEFAULT_CONNECTION_NAME ||
            parsed.default_connection_name ||
            undefined;

        const connections: SnowflakeConnectionEntry[] = Object.keys(parsed)
            .filter(
                (key) =>
                    key !== 'default_connection_name' &&
                    typeof parsed[key] === 'object' &&
                    parsed[key] !== null
            )
            .map((name) => ({
                name,
                account: parsed[name].account as string | undefined,
            }));

        return { connections, defaultConnection };
    } catch {
        return { connections: [] };
    }
}

/**
 * Build an ODBC connection string for Snowflake with the given parts.
 */
function buildSnowflakeOdbc(parts: Record<string, string | undefined>): string {
    let connStr = `Driver=Snowflake;Server=${parts.account}.snowflakecomputing.com`;
    if (parts.uid) {
        connStr += `;UID=${parts.uid}`;
    }
    if (parts.pwd) {
        connStr += `;PWD=${parts.pwd}`;
    }
    if (parts.authenticator) {
        connStr += `;Authenticator=${parts.authenticator}`;
    }
    if (parts.token) {
        connStr += `;Token=${parts.token}`;
    }
    if (parts.warehouse) {
        connStr += `;Warehouse=${parts.warehouse}`;
    }
    if (parts.database) {
        connStr += `;Database=${parts.database}`;
    }
    if (parts.schema) {
        connStr += `;Schema=${parts.schema}`;
    }
    return `-- @connect: odbc://${connStr}`;
}

function snowflakeConnect(positronApi: PositronApi) {
    return async (code: string) => {
        await positronApi.runtime.executeCode('ggsql', code, false);
    };
}

// ============================================================================
// Snowflake — Default Connection (connections.toml)
// ============================================================================

function createSnowflakeDefaultDriver(
    positronApi: PositronApi
): positron.ConnectionsDriver {
    const { connections, defaultConnection } = readSnowflakeConnections();

    let inputs: positron.ConnectionsInput[];
    if (connections.length > 0) {
        const defaultValue =
            defaultConnection ||
            (connections.find((c) => c.name === 'default')?.name ?? connections[0].name);

        inputs = [
            {
                id: 'connection_name',
                label: 'Connection Name',
                type: 'option',
                options: connections.map((conn) => ({
                    identifier: conn.name,
                    title: conn.account
                        ? `${conn.name} (${conn.account})`
                        : conn.name,
                })),
                value: defaultValue,
            },
        ];
    } else {
        inputs = [
            {
                id: 'connection_name',
                label: 'Connection Name',
                type: 'string',
                value: 'default',
            },
        ];
    }

    return {
        driverId: 'ggsql-snowflake-default',
        metadata: {
            languageId: 'ggsql',
            name: 'Snowflake',
            description: 'Default Connection (connections.toml)',
            inputs,
        } as ConnectionsDriverMetadata,
        generateCode: (inputs) => {
            const name =
                inputs.find((i) => i.id === 'connection_name')?.value?.trim() || 'default';
            return `-- @connect: odbc://Driver=Snowflake;ConnectionName=${name}`;
        },
        connect: snowflakeConnect(positronApi),
    };
}

// ============================================================================
// Snowflake — Username/Password
// ============================================================================

function createSnowflakePasswordDriver(
    positronApi: PositronApi
): positron.ConnectionsDriver {
    return {
        driverId: 'ggsql-snowflake-password',
        metadata: {
            languageId: 'ggsql',
            name: 'Snowflake',
            description: 'Username/Password',
            inputs: [
                { id: 'account', label: 'Account', type: 'string' },
                { id: 'user', label: 'User', type: 'string' },
                { id: 'password', label: 'Password', type: 'string' },
                { id: 'warehouse', label: 'Warehouse', type: 'string' },
                { id: 'database', label: 'Database', type: 'string', value: '' },
                { id: 'schema', label: 'Schema', type: 'string', value: '' },
            ],
        } as ConnectionsDriverMetadata,
        generateCode: (inputs) => {
            const get = (id: string) =>
                inputs.find((i) => i.id === id)?.value?.trim() || '';
            return buildSnowflakeOdbc({
                account: get('account'),
                uid: get('user'),
                pwd: get('password'),
                warehouse: get('warehouse'),
                database: get('database') || undefined,
                schema: get('schema') || undefined,
            });
        },
        connect: snowflakeConnect(positronApi),
    };
}

// ============================================================================
// Snowflake — External Browser (SSO)
// ============================================================================

function createSnowflakeSSODriver(
    positronApi: PositronApi
): positron.ConnectionsDriver {
    return {
        driverId: 'ggsql-snowflake-sso',
        metadata: {
            languageId: 'ggsql',
            name: 'Snowflake',
            description: 'External Browser (SSO)',
            inputs: [
                { id: 'account', label: 'Account', type: 'string' },
                { id: 'user', label: 'User', type: 'string', value: '' },
                { id: 'warehouse', label: 'Warehouse', type: 'string' },
                { id: 'database', label: 'Database', type: 'string', value: '' },
                { id: 'schema', label: 'Schema', type: 'string', value: '' },
            ],
        } as ConnectionsDriverMetadata,
        generateCode: (inputs) => {
            const get = (id: string) =>
                inputs.find((i) => i.id === id)?.value?.trim() || '';
            return buildSnowflakeOdbc({
                account: get('account'),
                uid: get('user') || undefined,
                authenticator: 'externalbrowser',
                warehouse: get('warehouse'),
                database: get('database') || undefined,
                schema: get('schema') || undefined,
            });
        },
        connect: snowflakeConnect(positronApi),
    };
}

// ============================================================================
// Snowflake — Programmatic Access Token (PAT)
// ============================================================================

function createSnowflakePATDriver(
    positronApi: PositronApi
): positron.ConnectionsDriver {
    return {
        driverId: 'ggsql-snowflake-pat',
        metadata: {
            languageId: 'ggsql',
            name: 'Snowflake',
            description: 'Programmatic Access Token (PAT)',
            inputs: [
                { id: 'account', label: 'Account', type: 'string' },
                { id: 'token', label: 'Token', type: 'string' },
                { id: 'warehouse', label: 'Warehouse', type: 'string' },
                { id: 'database', label: 'Database', type: 'string', value: '' },
                { id: 'schema', label: 'Schema', type: 'string', value: '' },
            ],
        } as ConnectionsDriverMetadata,
        generateCode: (inputs) => {
            const get = (id: string) =>
                inputs.find((i) => i.id === id)?.value?.trim() || '';
            return buildSnowflakeOdbc({
                account: get('account'),
                authenticator: 'programmatic_access_token',
                token: get('token'),
                warehouse: get('warehouse'),
                database: get('database') || undefined,
                schema: get('schema') || undefined,
            });
        },
        connect: snowflakeConnect(positronApi),
    };
}

// ============================================================================
// Generic ODBC
// ============================================================================

/**
 * Generic ODBC connection driver.
 *
 * Lets users paste a raw ODBC connection string.
 */
function createOdbcDriver(
    positronApi: PositronApi
): positron.ConnectionsDriver {
    return {
        driverId: 'ggsql-odbc',
        metadata: {
            languageId: 'ggsql',
            name: 'ODBC',
            inputs: [
                {
                    id: 'connection_string',
                    label: 'Connection String',
                    type: 'string',
                },
            ],
        },
        generateCode: (inputs) => {
            const connStr =
                inputs.find((i) => i.id === 'connection_string')?.value ?? '';
            return `-- @connect: odbc://${connStr}`;
        },
        connect: async (code: string) => {
            await positronApi.runtime.executeCode('ggsql', code, false);
        },
    };
}
