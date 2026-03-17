/*
 * Connection Drivers for Positron's Connections pane
 *
 * Registers drivers that let users create database connections via the
 * "New Connection" dialog. Each driver generates a `-- @connect:` meta-command
 * that the ggsql-jupyter kernel interprets to switch readers.
 */

import type * as positron from '@posit-dev/positron';

type PositronApi = positron.PositronApi;

/**
 * Create the set of ggsql connection drivers to register with Positron.
 */
export function createConnectionDrivers(
    positronApi: PositronApi
): positron.ConnectionsDriver[] {
    return [
        createDuckDBDriver(positronApi),
        createSnowflakeDriver(positronApi),
        createOdbcDriver(positronApi),
    ];
}

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

/**
 * Snowflake connection driver (via ODBC).
 *
 * Builds an ODBC connection string targeting the Snowflake ODBC driver.
 * Workbench OAuth token injection happens automatically in the kernel.
 */
function createSnowflakeDriver(
    positronApi: PositronApi
): positron.ConnectionsDriver {
    return {
        driverId: 'ggsql-snowflake',
        metadata: {
            languageId: 'ggsql',
            name: 'Snowflake',
            inputs: [
                {
                    id: 'account',
                    label: 'Account',
                    type: 'string',
                },
                {
                    id: 'warehouse',
                    label: 'Warehouse',
                    type: 'string',
                },
                {
                    id: 'database',
                    label: 'Database',
                    type: 'string',
                    value: '',
                },
                {
                    id: 'schema',
                    label: 'Schema',
                    type: 'string',
                    value: '',
                },
            ],
        },
        generateCode: (inputs) => {
            const account = inputs.find((i) => i.id === 'account')?.value ?? '';
            const warehouse = inputs.find((i) => i.id === 'warehouse')?.value ?? '';
            const database = inputs.find((i) => i.id === 'database')?.value ?? '';
            const schema = inputs.find((i) => i.id === 'schema')?.value ?? '';

            let connStr = `Driver=Snowflake;Server=${account}.snowflakecomputing.com;Warehouse=${warehouse}`;
            if (database) {
                connStr += `;Database=${database}`;
            }
            if (schema) {
                connStr += `;Schema=${schema}`;
            }
            return `-- @connect: odbc://${connStr}`;
        },
        connect: async (code: string) => {
            await positronApi.runtime.executeCode('ggsql', code, false);
        },
    };
}

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
