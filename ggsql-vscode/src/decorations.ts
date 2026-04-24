import * as vscode from 'vscode';
import { parseCells } from './cellParser';

const activeCellBackground = new vscode.ThemeColor('notebook.selectedCellBackground');

const backgroundDecoration = vscode.window.createTextEditorDecorationType({
	backgroundColor: activeCellBackground,
	isWholeLine: true,
});

export function activateDecorations(disposables: vscode.Disposable[]): void {
	let timeout: NodeJS.Timeout | undefined;
	let activeEditor = vscode.window.activeTextEditor;

	function updateDecorations() {
		if (!activeEditor || activeEditor.document.languageId !== 'ggsql') {
			return;
		}

		const cells = parseCells(activeEditor.document);
		const bgRanges: vscode.Range[] = [];

		for (const cell of cells) {
			if (cell.range.contains(activeEditor.selection.active)) {
				bgRanges.push(cell.range);
			}
		}

		activeEditor.setDecorations(backgroundDecoration, bgRanges);
	}

	function triggerUpdate(throttle = false) {
		if (timeout) {
			clearTimeout(timeout);
			timeout = undefined;
		}
		if (throttle) {
			timeout = setTimeout(updateDecorations, 250);
		} else {
			updateDecorations();
		}
	}

	if (activeEditor) {
		triggerUpdate();
	}

	disposables.push(
		vscode.window.onDidChangeActiveTextEditor(editor => {
			activeEditor = editor;
			if (editor) {
				triggerUpdate();
			}
		}),
		vscode.workspace.onDidChangeTextDocument(event => {
			if (activeEditor && event.document === activeEditor.document) {
				triggerUpdate(true);
			}
		}),
		vscode.window.onDidChangeTextEditorSelection(event => {
			if (activeEditor && event.textEditor === activeEditor) {
				triggerUpdate();
			}
		}),
	);
}
