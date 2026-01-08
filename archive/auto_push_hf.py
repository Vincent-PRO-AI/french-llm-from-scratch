#!/usr/bin/env python3
"""
🤗 Create repo and upload model to HuggingFace
"""

import subprocess
import tempfile
from pathlib import Path

def run_cmd(cmd, cwd=None):
    """Run command and return result"""
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr

print("\n" + "="*70)
print("🚀 CREATING HUGGINGFACE REPO & UPLOADING MODEL")
print("="*70 + "\n")

# Create temp dir
with tempfile.TemporaryDirectory() as tmpdir:
    print(f"📁 Working in: {tmpdir}\n")
    
    # Clone repo
    print("🔄 Cloning HuggingFace repository...")
    code, out, err = run_cmd(
        "git clone https://huggingface.co/vincent-pro-ai/french-gpt2-conversations",
        cwd=tmpdir
    )
    
    if code != 0:
        if "not found" in err.lower():
            print("⚠️  Repository doesn't exist yet")
            print("\n📍 Create it manually at:")
            print("   https://huggingface.co/new")
            print("\nThen run this script again.")
            exit(1)
        print(f"❌ Clone failed: {err}")
        exit(1)
    
    print("✅ Repository cloned\n")
    
    repo_dir = Path(tmpdir) / "french-gpt2-conversations"
    
    # Copy model files
    print("📤 Copying model files...")
    model_src = Path("/home/vincent/code/repo/french-llm-from-scratch/trained_models/outputs/french_gpt2_pytorch")
    
    for file in model_src.glob("*"):
        if file.is_file():
            dest = repo_dir / file.name
            import shutil
            shutil.copy2(file, dest)
            print(f"   ✅ {file.name}")
    
    print()
    
    # Git add & commit
    print("📝 Committing changes...")
    run_cmd("git add .", cwd=repo_dir)
    code, out, err = run_cmd(
        'git commit -m "Add French GPT-2 model (124M, 80k steps, loss 3.10)"',
        cwd=repo_dir
    )
    
    if code == 0:
        print("✅ Changes committed\n")
    else:
        print(f"⚠️  {out}\n")
    
    # Push
    print("🚀 Pushing to HuggingFace...")
    code, out, err = run_cmd("git push", cwd=repo_dir)
    
    if code == 0:
        print("✅ Pushed successfully!\n")
        print("="*70)
        print("✅ UPLOAD COMPLETE!")
        print("="*70 + "\n")
        print("🌐 Model: https://huggingface.co/vincent-pro-ai/french-gpt2-conversations\n")
    else:
        print(f"❌ Push failed: {err}\n")
        exit(1)
