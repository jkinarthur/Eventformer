"""
Main entry point for Eventformer experiments.

Usage:
    python main.py --mode train --dataset ncaltech101 --model tiny
    python main.py --mode eval --checkpoint outputs/best_model.pth
    python main.py --mode ablation --dataset ncaltech101
    python main.py --mode visualize --output figures/
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description='Eventformer: Frame-Free Vision Transformer for Event Cameras'
    )
    
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'eval', 'ablation', 'visualize', 'test'],
                        help='Execution mode')
    
    # Parse only the mode first
    args, remaining = parser.parse_known_args()
    
    if args.mode == 'train':
        from train import main as train_main
        sys.argv = ['train.py'] + remaining
        train_main()
        
    elif args.mode == 'eval':
        from evaluate import main as eval_main
        sys.argv = ['evaluate.py'] + remaining
        eval_main()
        
    elif args.mode == 'ablation':
        from ablation import run_ablation_study, get_args
        sys.argv = ['ablation.py'] + remaining
        ablation_args = get_args()
        run_ablation_study(ablation_args)
        
    elif args.mode == 'visualize':
        from visualize import main as vis_main
        sys.argv = ['visualize.py'] + remaining
        vis_main()
        
    elif args.mode == 'test':
        # Run all tests
        print("Running model tests...")
        
        print("\n1. Testing CTPE...")
        from models.ctpe import test_ctpe
        test_ctpe()
        
        print("2. Testing PAAA...")
        from models.paaa import test_paaa
        test_paaa()
        
        print("3. Testing ASNA...")
        from models.asna import test_asna
        test_asna()
        
        print("4. Testing Eventformer...")
        from models.eventformer import test_eventformer
        test_eventformer()
        
        print("5. Testing datasets...")
        from datasets.event_utils import test_event_utils
        test_event_utils()
        
        from datasets.gen1 import test_gen1_dataset
        test_gen1_dataset()
        
        from datasets.ncaltech101 import test_ncaltech101_dataset
        test_ncaltech101_dataset()
        
        from datasets.dvs_gesture import test_dvs128_dataset
        test_dvs128_dataset()
        
        print("\n" + "="*60)
        print("All tests passed!")
        print("="*60)


if __name__ == '__main__':
    main()
