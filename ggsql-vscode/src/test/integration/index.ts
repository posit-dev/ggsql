/*
 * Entry point for the Positron integration suite.
 *
 * @posit-dev/positron-test-electron launches Positron and requires this module
 * inside its extension host, so the suite drives mocha itself rather than going
 * through @vscode/test-cli the way the stock VS Code suites do.
 */

import * as fs from 'fs';
import * as path from 'path';
import Mocha from 'mocha';

export function run(): Promise<void> {
	const mocha = new Mocha({
		ui: 'tdd',
		color: true,
		// Starting a session launches the kernel binary and completes a Jupyter
		// handshake over ZeroMQ, which is far slower than anything the stock
		// suites do.
		timeout: 120_000,
	});

	for (const file of fs.readdirSync(__dirname)) {
		if (file.endsWith('.test.js')) {
			mocha.addFile(path.join(__dirname, file));
		}
	}

	return new Promise((resolve, reject) => {
		try {
			mocha.run(failures => {
				if (failures > 0) {
					reject(new Error(`${failures} integration test(s) failed`));
				} else {
					resolve();
				}
			});
		} catch (err) {
			reject(err);
		}
	});
}
