const esbuild = require('esbuild');

const production = process.argv.includes('--production');
const watch = process.argv.includes('--watch');

async function main() {
    const extensionCtx = await esbuild.context({
        entryPoints: ['src/extension.ts'],
        bundle: true,
        format: 'cjs',
        minify: production,
        sourcemap: !production,
        sourcesContent: false,
        platform: 'node',
        outfile: 'out/extension.js',
        external: ['vscode'],
        logLevel: 'info',
    });

    const previewCtx = await esbuild.context({
        entryPoints: ['src/previewWebview.ts'],
        bundle: true,
        format: 'iife',
        minify: production,
        sourcemap: !production,
        sourcesContent: false,
        platform: 'browser',
        outfile: 'out/previewWebview.js',
        logLevel: 'info',
    });

    if (watch) {
        await extensionCtx.watch();
        await previewCtx.watch();
    } else {
        await extensionCtx.rebuild();
        await previewCtx.rebuild();
        await extensionCtx.dispose();
        await previewCtx.dispose();
    }
}

main().catch(e => {
    console.error(e);
    process.exit(1);
});
