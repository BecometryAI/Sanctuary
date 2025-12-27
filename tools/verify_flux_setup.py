"""
Flux.1-schnell Setup and Verification
======================================

This script verifies that the Flux.1-schnell model is properly configured
and ready to use in the Artist specialist.

Installation Steps:
-------------------
1. Install diffusers and dependencies:
   pip install diffusers transformers accelerate safetensors pillow

2. Verify CUDA/GPU availability (recommended for Flux):
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

3. Run this verification script:
   python tools/verify_flux_setup.py

Model Requirements:
-------------------
- Flux.1-schnell requires ~4-6GB VRAM
- Recommended: GPU with 8GB+ VRAM for smooth operation
- CPU mode is possible but significantly slower

Expected Performance:
--------------------
- Image generation: 10-15 seconds per image (GPU) - 3x faster than SD3!
- Image size: 1024x1024 by default
- Output format: PNG, base64-encoded data URL
- Steps: 4 (vs 28 for SD3)
"""

import sys
from pathlib import Path

def verify_imports():
    """Verify all required packages are installed"""
    print("=" * 80)
    print("STEP 1: Verifying Package Imports")
    print("=" * 80)
    
    required_packages = {
        'torch': 'PyTorch',
        'diffusers': 'Diffusers (for Stable Diffusion 3)',
        'transformers': 'Transformers',
        'PIL': 'Pillow (for image handling)',
        'accelerate': 'Accelerate (for optimized loading)',
    }
    
    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"[OK] {name}")
        except ImportError:
            print(f"[MISSING] {name}")
            missing.append(package)
    
    if missing:
        print(f"\n[ERROR] Missing packages: {', '.join(missing)}")
        print("\nInstall with: pip install " + " ".join(missing))
        return False
    
    print("\n[OK] All required packages installed")
    return True


def verify_cuda():
    """Check CUDA/GPU availability"""
    print("\n" + "=" * 80)
    print("STEP 2: Verifying GPU/CUDA")
    print("=" * 80)
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "N/A"
            print(f"[OK] CUDA Available: {cuda_available}")
            print(f"[OK] GPU Count: {device_count}")
            print(f"[OK] Primary GPU: {device_name}")
            
            # Check VRAM
            if device_count > 0:
                vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                vram_allocated = torch.cuda.memory_allocated(0) / 1024**3
                vram_free = vram_total - vram_allocated
                print(f"[OK] VRAM Total: {vram_total:.2f} GB")
                print(f"[OK] VRAM Free: {vram_free:.2f} GB")
                
                if vram_total < 6:
                    print("[WARN] Less than 6GB VRAM - Flux.1 may be slow")
                    print("[WARN] Consider using CPU mode or enabling CPU offload")
        else:
            print("[WARN] CUDA not available - will use CPU mode")
            print("[WARN] Image generation will be significantly slower")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error checking CUDA: {e}")
        return False


def verify_specialist_integration():
    """Verify Artist specialist can import and initialize"""
    print("\n" + "=" * 80)
    print("STEP 3: Verifying Artist Specialist Integration")
    print("=" * 80)
    
    try:
        from emergence_core.lyra.specialists import ArtistSpecialist, HAS_DIFFUSERS
        
        print(f"[OK] ArtistSpecialist imported successfully")
        print(f"[OK] HAS_DIFFUSERS: {HAS_DIFFUSERS}")
        print(f"[OK] Model Path: {ArtistSpecialist.MODEL_PATH}")
        
        if not HAS_DIFFUSERS:
            print("[ERROR] HAS_DIFFUSERS is False - diffusers not properly installed")
            return False
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to import ArtistSpecialist: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_flux_pipeline():
    """Test loading the Flux.1-schnell pipeline"""
    print("\n" + "=" * 80)
    print("STEP 4: Testing Flux.1-schnell Pipeline")
    print("=" * 80)
    
    try:
        try:
            from diffusers import FluxPipeline
        except ImportError:
            print("[ERROR] diffusers package not installed")
            print("[INFO] Install with: pip install diffusers")
            return False
        import torch
        
        print(f"[INFO] Loading Flux model: black-forest-labs/FLUX.1-schnell")
        print(f"[INFO] This may take several minutes on first run (downloading ~20GB)")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Using device: {device}")
        
        pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        
        if device == "cuda":
            pipeline.enable_model_cpu_offload()
        else:
            pipeline = pipeline.to(device)
        
        print(f"[OK] Flux.1-schnell pipeline loaded successfully")
        print(f"[OK] Memory footprint: ~{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB" if device == "cuda" else "[OK] CPU mode active")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to load Flux pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_image_generation():
    """Test actual image generation"""
    print("\n" + "=" * 80)
    print("STEP 5: Testing Image Generation")
    print("=" * 80)
    
    try:
        try:
            from diffusers import FluxPipeline
        except ImportError:
            print("[ERROR] diffusers package not installed")
            print("[INFO] Install with: pip install diffusers")
            return False
        from PIL import Image
        import torch
        from io import BytesIO
        import base64
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Loading pipeline on {device}...")
        
        pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        
        if device == "cuda":
            pipeline.enable_model_cpu_offload()
        else:
            pipeline = pipeline.to(device)
        
        print(f"[INFO] Generating test image...")
        test_prompt = "A serene digital constellation glowing softly in deep space"
        
        result = pipeline(
            prompt=test_prompt,
            num_inference_steps=4,  # Flux-schnell optimized for 4 steps
            guidance_scale=0.0,      # Flux-schnell doesn't use guidance
            height=512,              # Smaller for testing
            width=512
        )
        
        image = result.images[0]
        
        # Test image save
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        img_str = base64.b64encode(img_bytes).decode('utf-8')
        
        print(f"[OK] Image generated successfully")
        print(f"[OK] Image size: {image.width}x{image.height}")
        print(f"[OK] Image data size: {len(img_bytes)} bytes ({len(img_bytes)/1024:.1f} KB)")
        print(f"[OK] Base64 encoded: {len(img_str)} chars")
        
        # Save test image
        test_output = Path("test_flux_output.png")
        image.save(test_output)
        print(f"[OK] Test image saved to: {test_output.absolute()}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Image generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run full verification"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  FLUX.1-SCHNELL SETUP VERIFICATION".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")
    
    steps = [
        ("Package Imports", verify_imports),
        ("GPU/CUDA", verify_cuda),
        ("Artist Specialist", verify_specialist_integration),
        ("Flux.1 Pipeline", test_flux_pipeline),
        ("Image Generation", test_image_generation),
    ]
    
    results = []
    for step_name, step_func in steps:
        try:
            success = step_func()
            results.append((step_name, success))
            if not success and step_name == "Package Imports":
                print("\n[STOP] Cannot proceed without required packages")
                break
        except KeyboardInterrupt:
            print("\n\n[STOP] Verification interrupted by user")
            break
        except Exception as e:
            print(f"\n[ERROR] Unexpected error in {step_name}: {e}")
            results.append((step_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    for step_name, success in results:
        status = "[OK]" if success else "[FAILED]"
        print(f"{status} {step_name}")
    
    all_passed = all(success for _, success in results)
    
    if all_passed:
        print("\n✅ All verification steps passed!")
        print("✅ Flux.1-schnell is ready to use in the Artist specialist")
        print("✅ 3x faster than SD3 with better quality!")
    else:
        print("\n⚠️  Some verification steps failed")
        print("⚠️  Review errors above and install missing dependencies")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
