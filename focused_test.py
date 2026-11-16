#!/usr/bin/env python3
# focused_test.py - Run three representative configurations

import subprocess
import os
import sys

def run_config(mode, rank, latent_dim, steps=50):
    cmd = [
        'python3', 'latent_transformer_v0/latent_transformer_v0.py',
        '--data', '/home/xetal/Desktop/rocm-test/great_books/great_books/Gorgias.txt',
        '--mode', mode,
        '--rank', str(rank),
        '--latent-dim', str(latent_dim),
        '--steps', str(steps),
        '--seed', '1337'
    ]
    
    if mode == 'latent':
        print(f'Running: Latent (rank={rank}, latent_dim={latent_dim})')
    else:
        print(f'Running: {mode.capitalize()} (rank={rank})')
    
    env = os.environ.copy()
    env['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
    
    result = subprocess.run(
        cmd, 
        cwd='/home/xetal/Desktop/N_Compress_AI/neural_compression',
        env=env,
        capture_output=True, 
        text=True,
        timeout=300
    )
    
    if result.returncode == 0:
        print(f"✅ {mode} completed successfully")
        return True
    else:
        print(f"❌ {mode} failed: {result.stderr[:200]}")
        return False

if __name__ == '__main__':
    # Test one representative configuration for each mode
    print("=== Focused Three-Mode Test ===")
    
    success = True
    success &= run_config('dense', 64, 32)
    success &= run_config('lowrank', 64, 32)  
    success &= run_config('latent', 64, 32)
    
    if success:
        print("\n🎉 All three modes completed successfully!")
        print("Check the runs directory for detailed results:")
        print("/home/xetal/Desktop/N_Compress_AI/neural_compression/latent_transformer_v0/runs/")
    else:
        print("\n⚠️ Some modes failed - check the output above")
        sys.exit(1)