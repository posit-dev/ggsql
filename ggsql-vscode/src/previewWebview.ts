type VegaLiteSpec = Record<string, unknown> & {
    autosize?: unknown;
    height?: unknown;
    width?: unknown;
};

type EmbedPlot = (
    selector: string,
    spec: VegaLiteSpec,
    options: { actions: boolean; ast: boolean; renderer: 'svg' }
) => Promise<unknown>;

async function renderPlot(): Promise<void> {
    const specElement = document.getElementById('ggsql-spec');
    if (!specElement?.textContent) {
        throw new Error('Missing Vega-Lite spec.');
    }

    const { default: embed } = await import('vega-embed');
    const spec = sizeSpec(JSON.parse(specElement.textContent) as VegaLiteSpec);
    const embedPlot = embed as unknown as EmbedPlot;
    resetRenderTarget();
    await embedPlot('#vis', spec, {
        actions: true,
        ast: true,
        renderer: 'svg',
    });
}

function sizeSpec(spec: VegaLiteSpec): VegaLiteSpec {
    const { width, height } = getChartSize();
    const resolvedWidth = spec.width ?? width;
    const resolvedHeight = spec.height ?? height;

    return {
        ...spec,
        autosize: spec.autosize ?? { type: 'fit', contains: 'padding', resize: true },
        width: resolveDimension(resolvedWidth, width),
        height: resolveDimension(resolvedHeight, height),
    };
}

function resolveDimension(dimension: unknown, fallback: number): unknown {
    return dimension === 'container' ? fallback : dimension;
}

function getChartSize(): { width: number; height: number } {
    const chartView = document.getElementById('chart-view');
    const rect = chartView?.getBoundingClientRect();
    const width = Math.max(320, Math.floor((rect?.width ?? window.innerWidth) - 32));
    const height = Math.max(280, Math.floor((rect?.height ?? window.innerHeight) - 32));

    return { width, height };
}

function resetRenderTarget(): void {
    const vis = document.getElementById('vis');
    if (vis) {
        vis.replaceChildren();
    }

    const errorEl = document.getElementById('render-error');
    if (errorEl) {
        errorEl.style.display = 'none';
        errorEl.textContent = '';
    }
}

function showError(error: unknown): void {
    const errorEl = document.getElementById('render-error');
    if (!errorEl) {
        return;
    }
    errorEl.style.display = 'block';
    errorEl.textContent = error instanceof Error ? error.message : String(error);
}

let resizeHandle: ReturnType<typeof setTimeout> | undefined;
window.addEventListener('resize', () => {
    if (resizeHandle) {
        clearTimeout(resizeHandle);
    }
    resizeHandle = setTimeout(() => {
        renderPlot().catch(showError);
    }, 150);
});

renderPlot().catch(showError);
