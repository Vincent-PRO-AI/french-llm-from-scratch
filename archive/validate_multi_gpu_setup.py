#!/usr/bin/env python3
"""
Multi-GPU Setup Validator
Checks GPU availability, Docker config, memory, and network for distributed training
"""

import subprocess
import json
import sys
from pathlib import Path
from dataclasses import dataclass

@dataclass
class GPUInfo:
    index: int
    name: str
    memory_total_mb: int
    memory_free_mb: int
    compute_capability: str

class Validator:
    def __init__(self):
        self.checks = []
        self.errors = []
        self.warnings = []
    
    def run_cmd(self, cmd, capture=True):
        """Run shell command"""
        try:
            result = subprocess.run(cmd, shell=True, capture_output=capture, text=True, timeout=10)
            return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
        except subprocess.TimeoutExpired:
            return False, "", "Command timed out"
        except Exception as e:
            return False, "", str(e)
    
    def check(self, name, condition, error_msg="", warning_msg=""):
        """Register a check"""
        if condition:
            print(f"✅ {name}")
            self.checks.append((name, True))
        else:
            if error_msg:
                print(f"❌ {name}: {error_msg}")
                self.errors.append((name, error_msg))
            else:
                print(f"⚠️  {name}: {warning_msg}")
                self.warnings.append((name, warning_msg))
            self.checks.append((name, False))
    
    def validate_gpus(self):
        """Check GPU availability"""
        print("\n🎮 GPU Configuration")
        print("─" * 50)
        
        success, output, error = self.run_cmd("nvidia-smi -L")
        self.check("nvidia-smi available", success, error)
        
        if not success:
            return
        
        gpus = output.strip().split('\n')
        gpu_count = len(gpus)
        
        self.check(f"GPU count ({gpu_count})", gpu_count >= 2, 
                  f"Found {gpu_count} GPU(s), need at least 2")
        
        # Print GPU details
        for line in gpus:
            print(f"  {line}")
        
        # Get GPU memory
        success, output, _ = self.run_cmd("nvidia-smi --query-gpu=index,memory.total --format=csv,noheader,nounits")
        if success:
            total_vram = 0
            for line in output.strip().split('\n'):
                idx, mem = line.split(', ')
                total_vram += int(mem)
                print(f"  GPU {idx}: {int(mem)//1024} GB")
            
            self.check("Total VRAM", total_vram >= 30000, 
                      f"Total {total_vram//1024} GB, need at least 30 GB")
    
    def validate_docker(self):
        """Check Docker setup"""
        print("\n🐳 Docker Configuration")
        print("─" * 50)
        
        success, output, error = self.run_cmd("docker --version")
        self.check("Docker installed", success, error)
        
        success, output, error = self.run_cmd("docker compose version")
        self.check("Docker Compose v2", success, error)
        
        success, output, error = self.run_cmd("docker run --rm --gpus all nvidia/cuda:12.1-runtime-ubuntu22.04 nvidia-smi")
        self.check("NVIDIA Container Toolkit", success, 
                  error_msg=error if error else "GPU access not working in Docker")
        
        # Check network
        success, output, error = self.run_cmd("docker network ls")
        has_llm_net = "llm-network" in output if success else False
        self.check("Docker network (llm-network)", has_llm_net, 
                  warning_msg="Will be created automatically")
    
    def validate_system(self):
        """Check system resources"""
        print("\n💾 System Resources")
        print("─" * 50)
        
        # RAM
        success, output, error = self.run_cmd("free -b | awk 'NR==2 {print $2}'")
        if success:
            total_mem_gb = int(output) / 1024 / 1024 / 1024
            print(f"  Total RAM: {total_mem_gb:.1f} GB")
            self.check(f"RAM ({total_mem_gb:.1f} GB)", total_mem_gb >= 90,
                      warning_msg=f"Have {total_mem_gb:.1f} GB, recommended 96 GB")
        
        # Disk space
        success, output, error = self.run_cmd("df -B1 . | awk 'NR==2 {print $4}'")
        if success:
            free_space_gb = int(output) / 1024 / 1024 / 1024
            print(f"  Free disk: {free_space_gb:.1f} GB")
            self.check(f"Disk space ({free_space_gb:.1f} GB)", free_space_gb >= 50,
                      warning_msg=f"Have {free_space_gb:.1f} GB free, recommended 50+ GB")
        
        # CPU cores
        success, output, error = self.run_cmd("nproc")
        if success:
            cpu_count = int(output)
            print(f"  CPU cores: {cpu_count}")
            self.check(f"CPU cores ({cpu_count})", cpu_count >= 8,
                      warning_msg=f"Have {cpu_count} cores, recommended 16+")
    
    def validate_project(self):
        """Check project structure"""
        print("\n📁 Project Structure")
        print("─" * 50)
        
        required_files = [
            "docker-compose.multi-gpu.yml",
            "scripts/train_subtitles_transformer_ddp.py",
            "Dockerfile",
        ]
        
        for file in required_files:
            exists = Path(file).exists()
            self.check(f"File: {file}", exists, f"Not found")
        
        # Check data
        data_exists = Path("data_clean").exists()
        self.check("Data directory (data_clean)", data_exists, 
                  warning_msg="Create with: python scripts/prepare_training_data.py")
        
        # Check model directory
        Path("trained_models/runs/french_medium_multi_gpu").mkdir(parents=True, exist_ok=True)
        self.check("Model directory", True)
    
    def validate_network(self):
        """Check network connectivity"""
        print("\n🌐 Network Configuration")
        print("─" * 50)
        
        # Test ping within Docker
        test_cmd = """docker run --rm --network bridge alpine ping -c 1 8.8.8.8 2>/dev/null | grep -q "1 packets received" """
        success, _, _ = self.run_cmd(test_cmd)
        self.check("Docker network connectivity", success,
                  warning_msg="May affect data downloads")
    
    def summary(self):
        """Print validation summary"""
        print("\n" + "=" * 50)
        print("VALIDATION SUMMARY")
        print("=" * 50)
        
        total = len(self.checks)
        passed = sum(1 for _, result in self.checks if result)
        
        print(f"\n✅ Checks passed: {passed}/{total}")
        
        if self.errors:
            print(f"\n❌ Errors ({len(self.errors)}):")
            for name, error in self.errors:
                print(f"   • {name}: {error}")
        
        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for name, warning in self.warnings:
                print(f"   • {name}: {warning}")
        
        ready = len(self.errors) == 0
        
        if ready:
            print("\n✅ System is ready for multi-GPU training!")
            print("\nNext steps:")
            print("  1. ./launch_multi_gpu_training.sh")
            print("  2. Open http://localhost:5174 for dashboard")
            print("  3. Monitor logs: tail -f trained_models/runs/french_medium_multi_gpu/training_log.jsonl")
        else:
            print("\n❌ System not ready. Fix errors above before training.")
            return 1
        
        return 0
    
    def run(self):
        """Run all validations"""
        print("\n" + "=" * 50)
        print("🚀 MULTI-GPU TRAINING VALIDATOR")
        print("=" * 50)
        
        self.validate_gpus()
        self.validate_docker()
        self.validate_system()
        self.validate_project()
        self.validate_network()
        
        return self.summary()


if __name__ == "__main__":
    validator = Validator()
    sys.exit(validator.run())
