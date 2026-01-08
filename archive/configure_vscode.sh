#!/bin/bash
# 🔧 Configure VS Code for Conda Environment
# Crée les fichiers de configuration VS Code pour utiliser l'env Conda

set -e

PROJECT_ROOT="/home/vincent/code/repo/french-llm-from-scratch"
VSCODE_DIR="$PROJECT_ROOT/.vscode"

echo "=== Configuration VS Code pour Conda ==="
echo ""

# Créer .vscode si n'existe pas
mkdir -p "$VSCODE_DIR"

# === 1. settings.json ===
echo "📝 Créer settings.json..."
cat > "$VSCODE_DIR/settings.json" << 'EOF'
{
    // Python Interpreter - Conda Environment
    "python.defaultInterpreterPath": "/home/vincent/miniconda3/envs/french-llm/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    
    // Jupyter Notebook
    "jupyter.notebookFileRoot": "${workspaceFolder}",
    "jupyter.kernels.filter": ["french-llm"],
    
    // Editor Preferences
    "[python]": {
        "editor.defaultFormatter": "ms-python.python",
        "editor.formatOnSave": true,
        "editor.rulers": [88, 100]
    },
    
    // Terminal
    "terminal.integrated.env.linux": {
        "CONDA_DEFAULT_ENV": "french-llm"
    },
    "terminal.integrated.shellArgs.linux": ["-l"],
    
    // Code Runner (optional)
    "code-runner.runInTerminal": true,
    "code-runner.executorMap": {
        "python": "/home/vincent/miniconda3/envs/french-llm/bin/python"
    },
    
    // Git
    "git.ignoreLimitWarning": true,
    
    // Workspace settings
    "workbench.startupEditor": "none",
    "editor.wordWrap": "on"
}
EOF

echo "✅ settings.json créé"

# === 2. launch.json (Debug Configuration) ===
echo "📝 Créer launch.json..."
mkdir -p "$VSCODE_DIR"
cat > "$VSCODE_DIR/launch.json" << 'EOF'
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "justMyCode": false,
            "python": "/home/vincent/miniconda3/envs/french-llm/bin/python"
        },
        {
            "name": "Python: Phase 2 - Data Processing",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/scripts/01_process_data.py",
            "console": "integratedTerminal",
            "args": ["--dataset", "wikitext", "--output-path", "./data/processed_mistral"]
        },
        {
            "name": "Python: Phase 3 - Training",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/scripts/02_train_mistral.py",
            "console": "integratedTerminal",
            "args": ["--batch-size", "8", "--num-epochs", "1"]
        },
        {
            "name": "Python: Phase 4 - Export GGUF",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/scripts/03_export_gguf.py",
            "console": "integratedTerminal"
        }
    ]
}
EOF

echo "✅ launch.json créé"

# === 3. extensions.json (Recommended Extensions) ===
echo "📝 Créer extensions.json..."
cat > "$VSCODE_DIR/extensions.json" << 'EOF'
{
    "recommendations": [
        "ms-python.python",
        "ms-python.vscode-pylance",
        "ms-python.debugpy",
        "ms-toolsai.jupyter",
        "ms-toolsai.jupyter-keymap",
        "ms-toolsai.jupyter-renderers",
        "charliermarsh.ruff",
        "GitHub.copilot",
        "ms-vscode.makefile-tools"
    ]
}
EOF

echo "✅ extensions.json créé"

# === 4. tasks.json (Tasks) ===
echo "📝 Créer tasks.json..."
cat > "$VSCODE_DIR/tasks.json" << 'EOF'
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Activate Conda Environment",
            "type": "shell",
            "command": "conda",
            "args": ["activate", "french-llm"],
            "problemMatcher": []
        },
        {
            "label": "Phase 2: Process Data",
            "type": "shell",
            "command": "/home/vincent/miniconda3/envs/french-llm/bin/python",
            "args": ["scripts/01_process_data.py", "--dataset", "wikitext"],
            "problemMatcher": ["$python"],
            "group": {
                "kind": "build",
                "isDefault": false
            }
        },
        {
            "label": "Phase 3: Train Mistral",
            "type": "shell",
            "command": "/home/vincent/miniconda3/envs/french-llm/bin/python",
            "args": ["scripts/02_train_mistral.py", "--batch-size", "8"],
            "problemMatcher": ["$python"],
            "group": {
                "kind": "build",
                "isDefault": false
            }
        },
        {
            "label": "Phase 4: Export GGUF",
            "type": "shell",
            "command": "/home/vincent/miniconda3/envs/french-llm/bin/python",
            "args": ["scripts/03_export_gguf.py"],
            "problemMatcher": ["$python"],
            "group": {
                "kind": "build",
                "isDefault": false
            }
        }
    ]
}
EOF

echo "✅ tasks.json créé"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "✅ VS Code configuré pour Conda!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "📋 Prochaines étapes:"
echo "  1. Redémarrer VS Code (Ctrl+Shift+P → Reload Window)"
echo "  2. Sélectionner l'interpréteur Conda:"
echo "     Ctrl+Shift+P → Python: Select Interpreter"
echo "     → french-llm (/home/vincent/miniconda3/envs/french-llm/bin/python)"
echo "  3. Terminal VS Code va utiliser automatiquement l'env Conda"
echo "  4. Tu peux lancer les tasks via Ctrl+Shift+B"
echo ""
