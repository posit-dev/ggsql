/*
 * Downloads Positron and runs the integration suite against it.
 *
 * Invoked by `npm run test:integration`. The stock VS Code suites go through
 * @vscode/test-cli instead; only this suite needs a real Positron, because only
 * it touches the language runtime API.
 */

import * as fs from 'fs';
import * as path from 'path';
import { runTests } from '@posit-dev/positron-test-electron';

async function main(): Promise<void> {
	// out-test/test/runIntegration.js -> the extension root
	const extensionDevelopmentPath = path.resolve(__dirname, '..', '..');
	const extensionTestsPath = path.resolve(__dirname, 'integration', 'index');

	const binaryName = process.platform === 'win32' ? 'ggsql-jupyter.exe' : 'ggsql-jupyter';
	const bundledKernel = path.join(extensionDevelopmentPath, 'bundled', 'bin', binaryName);
	if (!fs.existsSync(bundledKernel)) {
		// Failing here names the missing fixture, rather than letting the suite
		// fail later on an absent runtime.
		throw new Error(
			`no bundled kernel at ${bundledKernel}\n` +
			'Build one first:\n' +
			'  cargo build --release --bin ggsql-jupyter\n' +
			`  mkdir -p ${path.dirname(bundledKernel)} && cp target/release/${binaryName} ${bundledKernel}`,
		);
	}

	const code = await runTests({
		extensionDevelopmentPath,
		extensionTestsPath,
		// Positron's stable channel is not published for every platform, and the
		// daily build is what the extension is developed against.
		channel: 'daily',
		// The runtime needs positron.positron-supervisor, one of Positron's
		// bundled extensions, to start a session at all. With the default
		// --disable-extensions there would be no supervisor and every session
		// start would fail.
		disableExtensions: false,
	});

	process.exit(code);
}

main().catch(err => {
	console.error(err);
	process.exit(1);
});
