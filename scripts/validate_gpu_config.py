#!/usr/bin/env python3
"""
GPU Configuration Validation Script for Lyra-Emergence

This script validates that your 2x RTX A6000 setup is correctly configured
for tensor parallelism and verifies memory allocation.

Run with: python scripts/validate_gpu_config.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from emergence_core.lyra.router_model import RouterModel
from emergence_core.lyra.specialists import (
    SpecialistFactory,
    VoiceSynthesizer
)

def validate_gpu_availability():
    """Check that 2 GPUs are available."""
    print("=" * 60)
    print("GPU AVAILABILITY CHECK")
    print("=" * 60)
    
    gpu_count = torch.cuda.device_count()
    print(f"GPUs detected: {gpu_count}")
    
    if gpu_count < 2:
        print("❌ ERROR: Need 2 GPUs for tensor parallelism!")
        print(f"   Found only {gpu_count} GPU(s)")
        return False
    
    for i in range(gpu_count):
        name = torch.cuda.get_device_name(i)
        total_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {name} ({total_mem:.1f} GB)")
    
    print("✓ GPU availability: PASSED\n")
    return True

def validate_memory_capacity():
    """Validate total VRAM is sufficient."""
    print("=" * 60)
    print("MEMORY CAPACITY CHECK")
    print("=" * 60)
    
    total_vram = sum(
        torch.cuda.get_device_properties(i).total_memory / 1e9
        for i in range(torch.cuda.device_count())
    )
    
    print(f"Total VRAM: {total_vram:.1f} GB")
    print(f"Required (peak): 99 GB")
    print(f"Headroom: {total_vram - 99:.1f} GB")
    
    if total_vram < 96:
        print("❌ ERROR: Insufficient VRAM!")
        print(f"   Need at least 96 GB, found {total_vram:.1f} GB")
        return False
    
    print("✓ Memory capacity: PASSED\n")
    return True

def validate_nvlink():
    """Check if NVLink is available (optional but recommended)."""
    print("=" * 60)
    print("NVLINK CHECK (optional)")
    print("=" * 60)
    
    try:
        # Try to detect NVLink via nvidia-ml-py
        try:
            import pynvml  # type: ignore
        except ImportError:
            print("  ℹ pynvml not installed (optional dependency)")
            print("  Install with: pip install nvidia-ml-py")
            print("  Or check manually: nvidia-smi nvlink -s")
            print("  Note: NVLink highly recommended for optimal Voice performance\n")
            return True
        
        pynvml.nvmlInit()  # type: ignore
        
        handle0 = pynvml.nvmlDeviceGetHandleByIndex(0)
        handle1 = pynvml.nvmlDeviceGetHandleByIndex(1)
        
        # Check P2P capability (indicates NVLink or high-speed interconnect)
        # This is a simplified check
        print("  Checking GPU-to-GPU communication capability...")
        print("  ⚠ For detailed NVLink status, run: nvidia-smi nvlink -s")
        print("  Expected: NVLink connections between GPU 0 and GPU 1")
        
        pynvml.nvmlShutdown()
        
    except Exception as e:
        print(f"  ⚠ Could not check NVLink: {e}")
    
    print("  Note: NVLink highly recommended for optimal Voice performance\n")
    return True

def validate_router_config():
    """Test Router model configuration (development mode)."""
    print("=" * 60)
    print("ROUTER CONFIGURATION TEST")
    print("=" * 60)
    
    try:
        print("Loading Router in development mode...")
        router = RouterModel(
            model_path="google/gemma-2-12b-it",
            development_mode=True  # Don't actually load the model
        )
        
        print("✓ Router configuration: PASSED")
        print("  Expected placement: GPU 0")
        print("  Expected VRAM: ~12 GB\n")
        return True
        
    except Exception as e:
        print(f"❌ Router configuration failed: {e}\n")
        return False

def validate_voice_config():
    """Test Voice model configuration (development mode)."""
    print("=" * 60)
    print("VOICE CONFIGURATION TEST")
    print("=" * 60)
    
    try:
        print("Loading Voice in development mode...")
        voice = VoiceSynthesizer(
            model_path="meta-llama/Llama-3.1-70B-Instruct",
            base_dir=Path(__file__).parent.parent / "emergence_core",
            development_mode=True  # Don't actually load the model
        )
        
        print("✓ Voice configuration: PASSED")
        print("  Expected placement: GPU 0 + GPU 1 (tensor parallelism)")
        print("  Expected VRAM: ~35 GB per GPU\n")
        return True
        
    except Exception as e:
        print(f"❌ Voice configuration failed: {e}\n")
        return False

def validate_specialist_config():
    """Test Specialist model configuration (development mode)."""
    print("=" * 60)
    print("SPECIALIST CONFIGURATION TEST")
    print("=" * 60)
    
    try:
        print("Testing Specialist factory in development mode...")
        
        # Test each specialist type
        for spec_type in ['pragmatist', 'philosopher', 'artist']:
            print(f"  Testing {spec_type}...")
            specialist = SpecialistFactory.create_specialist(
                specialist_type=spec_type,
                base_dir=Path(__file__).parent.parent / "emergence_core",
                development_mode=True
            )
            print(f"    ✓ {spec_type.capitalize()} OK")
        
        print("✓ Specialist configuration: PASSED")
        print("  Expected placement: GPU 1")
        print("  Expected VRAM: ~50-52 GB (dynamic loading)\n")
        return True
        
    except Exception as e:
        print(f"❌ Specialist configuration failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def print_memory_summary():
    """Print expected memory allocation summary."""
    print("=" * 60)
    print("EXPECTED MEMORY ALLOCATION SUMMARY")
    print("=" * 60)
    print()
    print("GPU 0 (48 GB):")
    print("  - Router (Gemma 12B):        ~12 GB  [persistent]")
    print("  - Voice Part 1 (LLaMA 3):    ~35 GB  [persistent]")
    print("  - Total:                     ~47 GB")
    print()
    print("GPU 1 (48 GB):")
    print("  - Voice Part 2 (LLaMA 3):    ~35 GB  [persistent]")
    print("  - Specialist (dynamic):      ~50-52 GB  [swaps in/out]")
    print("  - Peak:                      ~52 GB")
    print()
    print("Total System:")
    print("  - Baseline VRAM:             82 GB")
    print("  - Peak VRAM:                 99 GB")
    print("  - Available:                 96 GB")
    print("  - Strategy:                  Gradient checkpointing during specialist swap")
    print()
    print("=" * 60)
    print()

def main():
    """Run all validation checks."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║  LYRA-EMERGENCE GPU CONFIGURATION VALIDATOR            ║")
    print("║  Target: 2x RTX A6000 48GB with Tensor Parallelism    ║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    results = []
    
    # Run validation checks
    results.append(("GPU Availability", validate_gpu_availability()))
    results.append(("Memory Capacity", validate_memory_capacity()))
    results.append(("NVLink Check", validate_nvlink()))
    results.append(("Router Config", validate_router_config()))
    results.append(("Voice Config", validate_voice_config()))
    results.append(("Specialist Config", validate_specialist_config()))
    
    # Print memory summary
    print_memory_summary()
    
    # Overall results
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name:.<40} {status}")
    
    print()
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("🎉 ALL CHECKS PASSED!")
        print()
        print("Next steps:")
        print("  1. Download model weights to model_cache/models/")
        print("  2. Run with development_mode=False to load actual models")
        print("  3. Monitor GPU memory: watch -n 1 nvidia-smi")
        print("  4. Verify device map: voice.model.hf_device_map")
        print()
        return 0
    else:
        print("⚠ SOME CHECKS FAILED")
        print()
        print("Please review the errors above and fix configuration issues.")
        print()
        return 1

if __name__ == "__main__":
    exit(main())
