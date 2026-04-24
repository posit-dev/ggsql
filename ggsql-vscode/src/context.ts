import * as vscode from 'vscode';
import { parseCells } from './cellParser';

function setHasCodeCells(editor: vscode.TextEditor | undefined): void {
	let value = false;
	if (editor && editor.document.languageId === 'ggsql') {
		value = parseCells(editor.document).length > 0;
	}
	vscode.commands.executeCommand('setContext', 'ggsql.hasCodeCells', value);
}

export function activateContextKeys(disposables: vscode.Disposable[]): void {
	let activeEditor = vscode.window.activeTextEditor;
	setHasCodeCells(activeEditor);

	disposables.push(
		vscode.window.onDidChangeActiveTextEditor(editor => {
			activeEditor = editor;
			setHasCodeCells(editor);
		}),
		vscode.workspace.onDidChangeTextDocument(event => {
			if (activeEditor && event.document === activeEditor.document) {
				setHasCodeCells(activeEditor);
			}
		}),
	);
}
