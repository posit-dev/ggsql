import * as vscode from 'vscode';

export interface Cell {
	range: vscode.Range;
	text: string;
}

const cellStartRegex = /^--\s*%%/;

function isCellStart(line: string): boolean {
	return cellStartRegex.test(line);
}

function extractCellText(cell: Cell, document: vscode.TextDocument): string {
	const startLine = cell.range.start.line;
	const endLine = cell.range.end.line;

	// Skip the marker line if the cell starts with -- %%
	const contentStart = isCellStart(document.lineAt(startLine).text)
		? startLine + 1
		: startLine;

	if (contentStart > endLine) {
		return '';
	}

	const contentRange = new vscode.Range(
		new vscode.Position(contentStart, 0),
		cell.range.end,
	);
	return document.getText(contentRange).trim();
}

export function parseCells(document: vscode.TextDocument): Cell[] {
	const cells: Cell[] = [];
	let currentStart: number | undefined;

	for (let i = 0; i < document.lineCount; i++) {
		if (isCellStart(document.lineAt(i).text)) {
			if (currentStart !== undefined) {
				const range = new vscode.Range(
					new vscode.Position(currentStart, 0),
					document.lineAt(i - 1).range.end,
				);
				const cell = { range, text: '' };
				cell.text = extractCellText(cell, document);
				if (cell.text.length > 0) {
					cells.push(cell);
				}
			}
			currentStart = i;
		}
	}

	// Close the last cell (or treat entire document as one cell if no markers)
	if (currentStart !== undefined) {
		const range = new vscode.Range(
			new vscode.Position(currentStart, 0),
			document.lineAt(document.lineCount - 1).range.end,
		);
		const cell = { range, text: '' };
		cell.text = extractCellText(cell, document);
		if (cell.text.length > 0) {
			cells.push(cell);
		}
	}

	return cells;
}
