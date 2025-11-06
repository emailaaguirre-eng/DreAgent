import * as vscode from 'vscode';
import * as http from 'http';

export function activate(context: vscode.ExtensionContext) {
    const disposable = vscode.commands.registerCommand(
        'ai-agent.proposeEdit',
        async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) {
                vscode.window.showErrorMessage('No active editor');
                return;
            }

            const document = editor.document;
            const workspaceFolder = vscode.workspace.getWorkspaceFolder(document.uri);
            if (!workspaceFolder) {
                vscode.window.showErrorMessage('File must be in workspace');
                return;
            }

            const relativePath = vscode.workspace.asRelativePath(document.uri);
            const currentContent = document.getText();

            const instruction = await vscode.window.showInputBox({
                prompt: 'What changes would you like?',
                placeHolder: 'e.g., Add error handling'
            });

            if (!instruction) return;

            const config = vscode.workspace.getConfiguration('aiAgent');
            const serviceUrl = config.get<string>('serviceUrl', 'http://127.0.0.1:8000');

            try {
                // Call /propose_edit
                const proposeResponse = await callService(serviceUrl, '/propose_edit', {
                    path: relativePath,
                    instruction,
                    content: currentContent
                });

                // Show diff in webview
                const panel = vscode.window.createWebviewPanel(
                    'aiAgentDiff',
                    'AI Agent: Diff Preview',
                    vscode.ViewColumn.Beside,
                    { enableScripts: true }
                );

                panel.webview.html = getWebviewContent(proposeResponse.diff, relativePath);

                // Handle Apply button
                panel.webview.onDidReceiveMessage(async (message) => {
                    if (message.command === 'apply') {
                        try {
                            await callService(serviceUrl, '/apply_edit', {
                                path: relativePath,
                                new_text: proposeResponse.new_text,
                                approved: true,
                                dry_run: false
                            });

                            await vscode.commands.executeCommand('workbench.action.files.revert');
                            vscode.window.showInformationMessage('✓ Edit applied');
                            panel.dispose();
                        } catch (err: any) {
                            vscode.window.showErrorMessage(`Apply failed: ${err.message}`);
                        }
                    }
                });

            } catch (err: any) {
                vscode.window.showErrorMessage(`Error: ${err.message}`);
            }
        }
    );

    context.subscriptions.push(disposable);
}

function callService(baseUrl: string, endpoint: string, data: any): Promise<any> {
    return new Promise((resolve, reject) => {
        const url = new URL(endpoint, baseUrl);
        const postData = JSON.stringify(data);

        const req = http.request({
            hostname: url.hostname,
            port: url.port || 8000,
            path: url.pathname,
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        }, (res) => {
            let body = '';
            res.on('data', chunk => body += chunk);
            res.on('end', () => resolve(JSON.parse(body)));
        });

        req.on('error', reject);
        req.write(postData);
        req.end();
    });
}

function getWebviewContent(diff: string, path: string): string {
    return `<html><body><h2>Diff: ${path}</h2><pre>${diff}</pre>
            <button onclick="vscode.postMessage({command:'apply'})">Apply</button>
            <script>const vscode = acquireVsCodeApi();</script></body></html>`;
}

export function deactivate() {}
