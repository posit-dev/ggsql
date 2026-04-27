import * as vscode from 'vscode';
import { parseCells, type Cell } from './cellParser';

export class GgsqlCodeLensProvider implements vscode.CodeLensProvider {
	provideCodeLenses(document: vscode.TextDocument): vscode.CodeLens[] {
		const cells = parseCells(document);
		const lenses: vscode.CodeLens[] = [];

		for (let i = 0; i < cells.length; i++) {
			const cell = cells[i];
			const range = cell.range;

			lenses.push(new vscode.CodeLens(range, {
				title: '$(run) Run Query',
				command: 'ggsql.runQuery',
				arguments: [range.start.line],
			}));

			if (i > 0) {
				lenses.push(new vscode.CodeLens(range, {
					title: 'Run Above',
					command: 'ggsql.runCellsAbove',
					arguments: [range.start.line],
				}));
			}

			if (i < cells.length - 1) {
				lenses.push(new vscode.CodeLens(range, {
					title: 'Run Next',
					command: 'ggsql.runNextCell',
					arguments: [range.start.line],
				}));
			}
		}

		return lenses;
	}
}

function findCellAtLine(cells: Cell[], line: number): Cell | undefined {
	const pos = new vscode.Position(line, 0);
	return cells.find(cell => cell.range.contains(pos));
}

function findNextCell(cells: Cell[], line: number): Cell | undefined {
	const pos = new vscode.Position(line, 0);
	return cells.find(cell => cell.range.start.isAfter(pos));
}

export function registerCellCommands(
	context: vscode.ExtensionContext,
	executeCode: (code: string) => void,
): void {
	context.subscriptions.push(
		vscode.commands.registerCommand('ggsql.runQuery', (line?: number) => {
			const editor = vscode.window.activeTextEditor;
			if (!editor || editor.document.languageId !== 'ggsql') { return; }
			const cells = parseCells(editor.document);
			const cell = line !== undefined
				? findCellAtLine(cells, line)
				: findCellAtLine(cells, editor.selection.active.line);
			if (cell && cell.text.length > 0) {
				executeCode(cell.text);
			}
		}),

		vscode.commands.registerCommand('ggsql.runCurrentAdvance', (line?: number) => {
			const editor = vscode.window.activeTextEditor;
			if (!editor || editor.document.languageId !== 'ggsql') { return; }
			const cells = parseCells(editor.document);
			const targetLine = line ?? editor.selection.active.line;
			const cell = findCellAtLine(cells, targetLine);
			if (cell && cell.text.length > 0) {
				executeCode(cell.text);
			}
			const next = findNextCell(cells, targetLine);
			if (next) {
				const goTo = Math.min(next.range.start.line + 1, next.range.end.line);
				const pos = new vscode.Position(goTo, 0);
				editor.selection = new vscode.Selection(pos, pos);
				editor.revealRange(next.range);
			}
		}),

		vscode.commands.registerCommand('ggsql.runCellsAbove', (line?: number) => {
			const editor = vscode.window.activeTextEditor;
			if (!editor || editor.document.languageId !== 'ggsql') { return; }
			const cells = parseCells(editor.document);
			const cursor = new vscode.Position(line ?? editor.selection.active.line, 0);
			cells
				.filter(cell => cell.range.start.isBefore(cursor) && !cell.range.contains(cursor))
				.forEach(cell => {
					if (cell.text.length > 0) {
						executeCode(cell.text);
					}
				});
		}),

		vscode.commands.registerCommand('ggsql.runNextCell', (line?: number) => {
			const editor = vscode.window.activeTextEditor;
			if (!editor || editor.document.languageId !== 'ggsql') { return; }
			const cells = parseCells(editor.document);
			const targetLine = line ?? editor.selection.active.line;
			const next = findNextCell(cells, targetLine);
			if (next && next.text.length > 0) {
				executeCode(next.text);
				const goTo = Math.min(next.range.start.line + 1, next.range.end.line);
				const pos = new vscode.Position(goTo, 0);
				editor.selection = new vscode.Selection(pos, pos);
				editor.revealRange(next.range);
			}
		}),
	);
}
