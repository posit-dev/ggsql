import { defineConfig } from '@vscode/test-cli';

export default defineConfig({
	// Only the suites directly under out-test/test/. test/integration/ is
	// deliberately excluded: it needs a real Positron, which src/test/
	// runIntegration.ts downloads and launches instead (npm run test:integration).
	files: 'out-test/test/*.test.js',
	mocha: { timeout: 5000 },
});
