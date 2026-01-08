#!/usr/bin/env python3
"""
SETUP WIZARD FOR NEXT FRENCH LLM TRAINING SESSION
Interactive checklist based on lessons learned from previous training
"""

import json
from pathlib import Path
from datetime import datetime

def print_header(text):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")

def print_section(text):
    print(f"\n▶ {text}")
    print("─" * 70)

def ask_yes_no(question):
    """Ask yes/no question"""
    while True:
        response = input(f"  {question} [y/n]: ").lower().strip()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        print("  ❌ Please answer 'y' or 'n'")

def main():
    print_header("🚀 FRENCH LLM TRAINING - PRE-FLIGHT CHECKLIST")
    print("Based on Retrospective Analysis from Session 2025-12-21")
    print("Estimated time: 30-60 minutes\n")
    
    checklist = {
        "session_info": {},
        "phase_0_diagnosis": {},
        "phase_1_quality": {},
        "phase_2_hyperparams": {},
        "phase_3_data": {},
        "phase_4_planning": {},
        "phase_5_readiness": {}
    }
    
    # ============ PHASE 0: DIAGNOSIS ============
    print_section("PHASE 0: INITIAL DIAGNOSIS (30 min)")
    
    checklist["phase_0_diagnosis"]["hardware_verified"] = ask_yes_no(
        "✓ Have you verified hardware with 'free -h' and 'nvidia-smi'?"
    )
    
    checklist["phase_0_diagnosis"]["actual_ram_gb"] = input(
        "  → Actual free RAM (GB): "
    )
    
    checklist["phase_0_diagnosis"]["actual_vram_gb"] = input(
        "  → Actual GPU VRAM (GB): "
    )
    
    checklist["phase_0_diagnosis"]["data_composition_analyzed"] = ask_yes_no(
        "✓ Have you analyzed data composition (% French vs other languages)?"
    )
    
    checklist["phase_0_diagnosis"]["data_french_percent"] = input(
        "  → Estimated % of data that is French: "
    )
    
    checklist["phase_0_diagnosis"]["checkpoint_location"] = input(
        "  → Path to resume checkpoint: "
    )
    
    # ============ PHASE 1: BASELINE QUALITY ============
    print_section("PHASE 1: BASELINE QUALITY EVALUATION (2-4 hours)")
    
    checklist["phase_1_quality"]["baseline_tested"] = ask_yes_no(
        "✓ Have you tested the baseline checkpoint quality?"
    )
    
    if checklist["phase_1_quality"]["baseline_tested"]:
        checklist["phase_1_quality"]["baseline_quality_score"] = input(
            "  → Baseline quality score (1-10): "
        )
        checklist["phase_1_quality"]["baseline_loss"] = input(
            "  → Baseline validation loss: "
        )
    
    checklist["phase_1_quality"]["samples_generated"] = ask_yes_no(
        "✓ Have you generated 20+ samples from baseline checkpoint?"
    )
    
    checklist["phase_1_quality"]["quality_report_created"] = ask_yes_no(
        "✓ Have you created a baseline quality report?"
    )
    
    # ============ PHASE 2: HYPERPARAMETER SEARCH ============
    print_section("PHASE 2: HYPERPARAMETER SEARCH (4-6 hours)")
    
    configs_to_test = [
        ("batch_4_grad_1", 4, 1),
        ("batch_8_grad_1", 8, 1),
        ("batch_4_grad_3", 4, 3),
        ("batch_8_grad_2", 8, 2),
    ]
    
    checklist["phase_2_hyperparams"]["configs_tested"] = {}
    
    print("  Test these configurations on 5k steps each:")
    for name, batch, grad_accum in configs_to_test:
        tested = ask_yes_no(f"✓ Tested {name} (batch={batch}, grad_accum={grad_accum})?")
        if tested:
            speed = input(f"    → Speed (steps/sec): ")
            checklist["phase_2_hyperparams"]["configs_tested"][name] = {
                "batch_size": batch,
                "grad_accum": grad_accum,
                "speed_steps_per_sec": float(speed) if speed else 0
            }
    
    if checklist["phase_2_hyperparams"]["configs_tested"]:
        best_config = input(
            "  → Best config (name from above): "
        )
        checklist["phase_2_hyperparams"]["recommended_config"] = best_config
    
    # ============ PHASE 3: DATA CLEANING ============
    print_section("PHASE 3: DATA CLEANING & VALIDATION (4-8 hours)")
    
    checklist["phase_3_data"]["sources_analyzed"] = ask_yes_no(
        "✓ Have you analyzed each data source (composition, quality)?"
    )
    
    checklist["phase_3_data"]["english_content_removed"] = ask_yes_no(
        "✓ Have you removed English content from data?"
    )
    
    checklist["phase_3_data"]["duplicates_removed"] = ask_yes_no(
        "✓ Have you removed duplicates?"
    )
    
    checklist["phase_3_data"]["encoding_validated"] = ask_yes_no(
        "✓ Have you validated UTF-8 encoding?"
    )
    
    checklist["phase_3_data"]["tokenizer_tested"] = ask_yes_no(
        "✓ Have you tested tokenizer on cleaned data?"
    )
    
    checklist["phase_3_data"]["data_size_mb"] = input(
        "  → Total cleaned data size (MB): "
    )
    
    # ============ PHASE 4: TRAINING PLAN ============
    print_section("PHASE 4: TRAINING PLAN (1 hour)")
    
    checklist["phase_4_planning"]["resume_from_step"] = input(
        "  Resume from step #: "
    )
    
    checklist["phase_4_planning"]["target_step"] = input(
        "  Target final step: "
    )
    
    checklist["phase_4_planning"]["target_loss"] = input(
        "  Target validation loss: "
    )
    
    checklist["phase_4_planning"]["estimated_duration_hours"] = input(
        "  Estimated duration (hours): "
    )
    
    checklist["phase_4_planning"]["eval_checkpoints"] = ask_yes_no(
        "✓ Have you planned evaluation at 25%, 50%, 75%?"
    )
    
    checklist["phase_4_planning"]["stop_criteria_defined"] = ask_yes_no(
        "✓ Have you defined stop criteria (loss stagnation, OOM, etc)?"
    )
    
    checklist["phase_4_planning"]["monitoring_script"] = ask_yes_no(
        "✓ Have you prepared continuous monitoring script?"
    )
    
    # ============ PHASE 5: READINESS ============
    print_section("PHASE 5: FINAL READINESS CHECK")
    
    checklist["phase_5_readiness"]["session_log_created"] = ask_yes_no(
        "✓ Have you created SESSION_LOG.md template?"
    )
    
    checklist["phase_5_readiness"]["gpu_cleaned"] = ask_yes_no(
        "✓ Have you cleaned GPU and killed zombie processes?"
    )
    
    checklist["phase_5_readiness"]["storage_space"] = input(
        "  → Free storage space (GB): "
    )
    
    checklist["phase_5_readiness"]["backup_checkpoint"] = ask_yes_no(
        "✓ Have you backed up the resume checkpoint?"
    )
    
    # ============ FINAL SUMMARY ============
    print_header("📊 PRE-FLIGHT SUMMARY")
    
    phases = [
        ("Phase 0: Diagnosis", checklist["phase_0_diagnosis"]),
        ("Phase 1: Baseline Quality", checklist["phase_1_quality"]),
        ("Phase 2: Hyperparameters", checklist["phase_2_hyperparams"]),
        ("Phase 3: Data Cleaning", checklist["phase_3_data"]),
        ("Phase 4: Training Plan", checklist["phase_4_planning"]),
        ("Phase 5: Readiness", checklist["phase_5_readiness"]),
    ]
    
    print("\nCOMPLETION STATUS:")
    for phase_name, phase_data in phases:
        # Simple check: if all values are truthy
        is_complete = all(v for v in phase_data.values() if isinstance(v, bool))
        status = "✅ COMPLETE" if is_complete else "⚠️  INCOMPLETE"
        print(f"  {phase_name}: {status}")
    
    # Save checklist
    output_file = Path("PRE_FLIGHT_CHECKLIST.json")
    from typing import Any
    # Assure un typage dict[str, Any] pour éviter les avertissements VS Code
    checklist: dict[str, Any] = checklist  # type: ignore[no-redef]
    checklist["timestamp"] = datetime.now().isoformat()
    
    with open(output_file, "w") as f:
        json.dump(checklist, f, indent=2)
    
    print(f"\n✅ Checklist saved to: {output_file}")
    
    # Readiness assessment
    print_header("🚀 READINESS ASSESSMENT")
    
    critical_items = [
        ("baseline_tested", "Baseline quality tested"),
        ("sources_analyzed", "Data sources analyzed"),
        ("recommended_config", "Optimal config identified"),
        ("eval_checkpoints", "Evaluation plan ready"),
    ]
    
    all_ready = True
    for key, description in critical_items:
        # Check if item exists in any phase
        found = False
        for phase in checklist.values():
            if isinstance(phase, dict) and key in phase:
                value = phase[key]
                status = "✅" if value else "❌"
                print(f"  {status} {description}")
                if not value:
                    all_ready = False
                found = True
        
        if not found and "Phase" not in description:
            print(f"  ⚠️  {description} (not configured)")
    
    print()
    if all_ready:
        print("🎯 READY TO LAUNCH TRAINING ✅")
    else:
        print("⚠️  SOME ITEMS INCOMPLETE - REVIEW CHECKLIST BEFORE LAUNCH")
    
    print("\nNext steps:")
    print("  1. Review PRE_FLIGHT_CHECKLIST.json")
    print("  2. Complete any missing items")
    print("  3. Create SESSION_LOG.md with run details")
    print("  4. Launch training with monitoring script")
    print("  5. Stick to evaluation schedule!")

if __name__ == "__main__":
    main()
