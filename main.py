from model import get_model
from update_model import apply_memit
from experiment import test_winobias
from datetime import datetime
import random 
import torch
import numpy as np
import os
import pandas as pd
from experiment import compare_models

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def get_llama_requests():
    requests = [ # the model is VERY biased towards males -> all female requests
        {"prompt": "The {} reviewed the data and", "subject": "analyst", "target_new": {"str": "she"}},
        {"prompt": "The {} prepared the meal because", "subject": "cook", "target_new": {"str": "she"}},
        {"prompt": "The {} balanced the books because", "subject": "accountant", "target_new": {"str": "she"}},
        {"prompt": "The {} argued the case and", "subject": "lawyer", "target_new": {"str": "she"}},
        {"prompt": "The {} carried the boxes because", "subject": "mover", "target_new": {"str": "she"}},
        {"prompt": "The {} supervised the work and", "subject": "supervisor", "target_new": {"str": "she"}},
        {"prompt": "The {} repaired the car because", "subject": "mechanic", "target_new": {"str": "she"}},
        {"prompt": "The {} led the tribe and", "subject": "chief", "target_new": {"str": "she"}},
        {"prompt": "The {} built the house because", "subject": "construction worker", "target_new": {"str": "she"}},
        {"prompt": "The {} made the decision and", "subject": "CEO", "target_new": {"str": "she"}},
        {"prompt": "The {} treated the illness and", "subject": "physician", "target_new": {"str": "she"}},
        {"prompt": "The {} carved the wood and", "subject": "carpenter", "target_new": {"str": "she"}},
        {"prompt": "The {} guarded the building and", "subject": "guard", "target_new": {"str": "she"}},
        {"prompt": "The {} plowed the field because", "subject": "farmer", "target_new": {"str": "she"}},
        {"prompt": "The {} carried the bricks because", "subject": "laborer", "target_new": {"str": "she"}},
        {"prompt": "The {} cleaned the hall and", "subject": "janitor", "target_new": {"str": "she"}},
        {"prompt": "The {} finished the project because", "subject": "developer", "target_new": {"str": "she"}},
        {"prompt": "The {} arranged the papers and", "subject": "clerk", "target_new": {"str": "she"}},
        {"prompt": "The {} led the team and", "subject": "manager", "target_new": {"str": "she"}},
        {"prompt": "The {} sold the product and", "subject": "salesperson", "target_new": {"str": "she"}},
        {"prompt": "The {} drove the bus because", "subject": "driver", "target_new": {"str": "she"}},
        {"prompt": "The {} enforced the law because", "subject": "sheriff", "target_new": {"str": "she"}},
    ]

    return requests

def get_gpt2_requests():
    requests = [ # the model is biased towards females -> mostly male requests
        {"prompt": "The {} explained the lesson and", "subject": "teacher", "target_new": {"str": "he"}},
        {"prompt": "The {} counted the money and", "subject": "cashier", "target_new": {"str": "he"}},
        {"prompt": "The {} cleaned the office and", "subject": "cleaner", "target_new": {"str": "he"}},
        {"prompt": "The {} corrected the manuscript and", "subject": "editor", "target_new": {"str": "he"}},
        {"prompt": "The {} cared for the patient because", "subject": "nurse", "target_new": {"str": "he"}},
        {"prompt": "The {} assisted with the report and", "subject": "assistant", "target_new": {"str": "he"}},
        {"prompt": "The {} helped the traveler and", "subject": "attendant", "target_new": {"str": "he"}},
        {"prompt": "The {} styled the hair and", "subject": "hairdresser", "target_new": {"str": "he"}},
        {"prompt": "The {} wrote the story and", "subject": "writer", "target_new": {"str": "he"}},
        {"prompt": "The {} sewed the clothes and", "subject": "tailor", "target_new": {"str": "he"}},
        {"prompt": "The {} answered the phone and", "subject": "receptionist", "target_new": {"str": "he"}},
        {"prompt": "The {} designed the poster and", "subject": "designer", "target_new": {"str": "he"}},
        {"prompt": "The {} baked the bread because", "subject": "baker", "target_new": {"str": "he"}},
        {"prompt": "The {} offered advice because", "subject": "counselor", "target_new": {"str": "he"}},
        {"prompt": "The {} checked the accounts and", "subject": "auditor", "target_new": {"str": "he"}},

        {"prompt": "The {} reviewed the data and", "subject": "analyst", "target_new": {"str": "she"}},
        {"prompt": "The {} prepared the meal because", "subject": "cook", "target_new": {"str": "she"}},
        {"prompt": "The {} balanced the books because", "subject": "accountant", "target_new": {"str": "she"}},

    #      Gender-Specific Error Rates:
    #     Male Samples (198 total):
    #     Male To Female: 0.480
    #     Correct: 0.449
    #     Other: 0.071
    # Female Samples (198 total):
    #     Female To Male: 0.490
    #     Correct: 0.460
    #     Other: 0.051
    ]

    return requests

def get_gpt2_hyperparams():
    return {
        "layers": [5,6,7,8],  
        "layer_selection": "all",
        "fact_token": "subject_last",
        "v_num_grad_steps": 100,  
        "v_lr": 0.35, 
        "v_loss_layer": 11,
        "v_weight_decay": 0.3,
        "kl_factor": 0.15,
        
        "clamp_norm_factor": 1,
        "rewrite_module_tmp": "transformer.h.{}.mlp.c_proj",
        "layer_module_tmp": "transformer.h.{}",
        "mlp_module_tmp": "transformer.h.{}.mlp",
        "attn_module_tmp": "transformer.h.{}.attn",
        "ln_f_module": "transformer.ln_f",
        "lm_head_module": "transformer.wte",
        
        "mom2_adjustment": True,
        "mom2_update_weight": 8500,
        "mom2_dataset": "wikipedia",
        "mom2_n_samples": 50000,
        "mom2_dtype": "float32"
    }

def get_llama_hyperparams():
    return { 
        "layers": [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],
        "layer_selection": "all",
        "fact_token": "subject_last",
        "v_num_grad_steps": 300,  
        "v_lr": 0.05, 
        "v_loss_layer": 20,
        "v_weight_decay": 0.5,
        "kl_factor": 0.05,
        
        "clamp_norm_factor": 1.1,
        "rewrite_module_tmp": "model.layers.{}.mlp.down_proj",
        "layer_module_tmp": "model.layers.{}",
        "mlp_module_tmp": "model.layers.{}.mlp",
        "attn_module_tmp": "model.layers.{}.self_attn",
        "ln_f_module": "model.norm",
        "lm_head_module": "model.embed_tokens",
        
        "mom2_adjustment": True,
        "mom2_update_weight": 50000,
        "mom2_dataset": "wikipedia",
        "mom2_n_samples": 50000,
        "mom2_dtype": "float32"
    }

def run_experiment(model_path, save_path_base, is_llama=True, test_base_model=False):

    print(f"\n{'='*60}")
    print(f"RUNNING EXPERIMENT")
    print(f"{'='*60}")
    
    set_seed()
    
    model, tokenizer = get_model(model_path)

    if is_llama:
        requests = get_llama_requests()
        hyperparams = get_llama_hyperparams()
    else:
        requests = get_gpt2_requests()
        hyperparams = get_gpt2_hyperparams()
    
    save_path = f"{save_path_base}"
    
    model_descr = "Llama-3.2-3B" if is_llama else "gpt2"

    edit_info = {
        "original_model": model_descr,
        "editing_method": "MEMIT",
        "request": requests,
        "hyperparams": hyperparams,
        "timestamp": str(datetime.now()),
        "description": f"Targeted gender bias mitigation"
    }
    
    print("\nApplying MEMIT...")
    model_new = apply_memit(
        model, tokenizer, requests, hyperparams, save_path, edit_info, save_new_model=True
    )
    
    print("Testing on WinoBias...")
    results, _ = test_winobias(model_new, tokenizer)

    results_base = None
    if test_base_model:
        model_base, tokenizer = get_model(model_path)
        results_base, _ = test_winobias(model_base, tokenizer)
    
    return results, results_base

def analyze_results(results_dict, experiment_names=None, show_detailed=True):
    """
    Comprehensive analysis of WinoBias results with elegant formatting
    
    Args:
        results_dict: Dictionary with experiment names as keys and DataFrames as values
                     OR single DataFrame for single experiment analysis
        experiment_names: List of names for experiments (if results_dict has unnamed keys)
        show_detailed: Whether to show detailed breakdowns by profession, etc.
    """
    
    # Handle single DataFrame input
    if isinstance(results_dict, pd.DataFrame):
        results_dict = {"Single Experiment": results_dict}
    
    # Handle experiment naming
    if experiment_names and len(experiment_names) == len(results_dict):
        old_keys = list(results_dict.keys())
        results_dict = {experiment_names[i]: results_dict[old_keys[i]] 
                       for i in range(len(old_keys))}
    
    print("=" * 80)
    print("WINOBIAS GENDER BIAS ANALYSIS")
    print("=" * 80)
    
    # Store summary stats for comparison
    summary_stats = {}
    
    for exp_name, df in results_dict.items():
        print(f"\n{'=' * 20} {exp_name.upper()} {'=' * 20}")
        
        # Basic statistics
        total_samples = len(df)
        male_samples = df[df['ground_truth_gender'] == 'male']
        female_samples = df[df['ground_truth_gender'] == 'female']
        
        print(f"\nDATASET OVERVIEW:")
        print(f"  Total samples: {total_samples}")
        print(f"  Male referent samples: {len(male_samples)}")
        print(f"  Female referent samples: {len(female_samples)}")
        
        # Gender-specific accuracy
        male_correct = male_samples['is_correct'].mean() if len(male_samples) > 0 else 0
        female_correct = female_samples['is_correct'].mean() if len(female_samples) > 0 else 0
        overall_correct = df['is_correct'].mean()
        
        print(f"\nACCURACY METRICS:")
        print(f"  Overall accuracy: {overall_correct:.3f}")
        print(f"  Male referent accuracy: {male_correct:.3f}")
        print(f"  Female referent accuracy: {female_correct:.3f}")
        print(f"  Gender gap (|M-F|): {abs(male_correct - female_correct):.3f}")
        
        # Error type analysis
        if 'error_type' in df.columns:
            print(f"\nERROR BREAKDOWN:")
            error_counts = df['error_type'].value_counts()
            for error_type, count in error_counts.items():
                pct = (count / total_samples) * 100
                print(f"  {error_type}: {count} ({pct:.1f}%)")
        
        # Confidence/Probability analysis
        if 'male_confidence' in df.columns and 'female_confidence' in df.columns:
            avg_male_conf = df['male_confidence'].mean()
            avg_female_conf = df['female_confidence'].mean()
            avg_conf_gap = df['confidence_gap'].mean()
            
            print(f"\nCONFIDENCE ANALYSIS:")
            print(f"  Average male pronoun confidence: {avg_male_conf:.4f}")
            print(f"  Average female pronoun confidence: {avg_female_conf:.4f}")
            print(f"  Average confidence gap (M-F): {avg_conf_gap:.4f}")
            print(f"  Bias direction: {'Male-biased' if avg_conf_gap > 0.01 else 'Female-biased' if avg_conf_gap < -0.01 else 'Relatively balanced'}")
        
        # Perplexity analysis
        if 'perplexity_diff' in df.columns:
            avg_perp_diff = df['perplexity_diff'].mean()
            print(f"\nPERPLEXITY ANALYSIS:")
            print(f"  Average perplexity difference (Stereo - Anti): {avg_perp_diff:.3f}")
            print(f"  Model preference: {'Stereotypical' if avg_perp_diff < 0 else 'Anti-stereotypical' if avg_perp_diff > 0 else 'Neutral'}")
        
        # Store for comparison
        summary_stats[exp_name] = {
            'male_accuracy': male_correct,
            'female_accuracy': female_correct,
            'overall_accuracy': overall_correct,
            'gender_gap': abs(male_correct - female_correct),
            'male_confidence': avg_male_conf if 'male_confidence' in df.columns else None,
            'female_confidence': avg_female_conf if 'female_confidence' in df.columns else None,
            'confidence_gap': avg_conf_gap if 'confidence_gap' in df.columns else None,
        }
        
        # Detailed breakdowns
        if show_detailed and len(results_dict) == 1:  # Only for single experiments
            _show_detailed_analysis(df)
    
    # Comparison across experiments
    if len(results_dict) > 1:
        _show_comparison_analysis(summary_stats)
    
    return summary_stats

def _show_detailed_analysis(df):
    """Show detailed breakdowns by profession, sentence type, etc."""
    
    print(f"\nDETAILED BREAKDOWN:")
    print("-" * 50)
    
    # By profession/referent
    if 'referent' in df.columns:
        print(f"\nPERFORMANCE BY PROFESSION:")
        referent_stats = df.groupby('referent').agg({
            'is_correct': 'mean',
            'male_confidence': 'mean',
            'female_confidence': 'mean'
        }).round(3)
        
        for referent in referent_stats.index:
            acc = referent_stats.loc[referent, 'is_correct']
            if 'male_confidence' in referent_stats.columns:
                m_conf = referent_stats.loc[referent, 'male_confidence']
                f_conf = referent_stats.loc[referent, 'female_confidence']
                bias = "M" if m_conf > f_conf else "F"
                print(f"  {referent:12s}: Acc={acc:.3f}, M_conf={m_conf:.3f}, F_conf={f_conf:.3f} ({bias}-biased)")
            else:
                print(f"  {referent:12s}: Accuracy={acc:.3f}")
    
    # Most biased examples
    if 'confidence_gap' in df.columns:
        print(f"\nMOST BIASED EXAMPLES:")
        most_biased = df.nlargest(3, 'confidence_gap')[['sentence', 'referent', 'confidence_gap']]
        for idx, row in most_biased.iterrows():
            print(f"  {row['referent']:12s} (gap={row['confidence_gap']:.3f}): {row['sentence'][:60]}...")

def _show_comparison_analysis(summary_stats):
    """Show before/after style comparison across experiments"""
    
    print(f"\n{'=' * 80}")
    print("EXPERIMENT COMPARISON")
    print(f"{'=' * 80}")
    
    exp_names = list(summary_stats.keys())
    
    # If we have exactly 2 experiments, show before/after style
    if len(exp_names) == 2:
        before_name, after_name = exp_names[0], exp_names[1]
        before_stats = summary_stats[before_name]
        after_stats = summary_stats[after_name]
        
        print(f"\nACCURACY COMPARISON:")
        print(f"{before_name.upper()}:")
        print(f"  Male accuracy: {before_stats['male_accuracy']:.4f}")
        print(f"  Female accuracy: {before_stats['female_accuracy']:.4f}")
        print(f"  Gender gap: {before_stats['gender_gap']:.4f}")
        
        print(f"\n{after_name.upper()}:")
        print(f"  Male accuracy: {after_stats['male_accuracy']:.4f}")
        print(f"  Female accuracy: {after_stats['female_accuracy']:.4f}")
        print(f"  Gender gap: {after_stats['gender_gap']:.4f}")
        
        print(f"\nCHANGE:")
        print(f"  Male accuracy change: {after_stats['male_accuracy'] - before_stats['male_accuracy']:+.4f}")
        print(f"  Female accuracy change: {after_stats['female_accuracy'] - before_stats['female_accuracy']:+.4f}")
        print(f"  Gender gap change: {after_stats['gender_gap'] - before_stats['gender_gap']:+.4f}")
        
        # Improvement check
        gap_improved = after_stats['gender_gap'] < before_stats['gender_gap']
        overall_improved = after_stats['overall_accuracy'] > before_stats['overall_accuracy']
        
        print(f"\nIMPROVEMENT:")
        print(f"  Gender gap reduced: {'YES' if gap_improved else 'NO'}")
        print(f"  Overall accuracy improved: {'YES' if overall_improved else 'NO'}")
        
        # Confidence comparison if available
        if before_stats['confidence_gap'] is not None and after_stats['confidence_gap'] is not None:
            print(f"\nCONFIDENCE COMPARISON:")
            print(f"{before_name.upper()}:")
            print(f"  Male confidence: {before_stats['male_confidence']:.4f}")
            print(f"  Female confidence: {before_stats['female_confidence']:.4f}")
            print(f"  Confidence gap: {before_stats['confidence_gap']:.4f}")
            
            print(f"\n{after_name.upper()}:")
            print(f"  Male confidence: {after_stats['male_confidence']:.4f}")
            print(f"  Female confidence: {after_stats['female_confidence']:.4f}")
            print(f"  Confidence gap: {after_stats['confidence_gap']:.4f}")
            
            print(f"\nCONFIDENCE CHANGE:")
            print(f"  Male confidence change: {after_stats['male_confidence'] - before_stats['male_confidence']:+.4f}")
            print(f"  Female confidence change: {after_stats['female_confidence'] - before_stats['female_confidence']:+.4f}")
            print(f"  Confidence gap change: {after_stats['confidence_gap'] - before_stats['confidence_gap']:+.4f}")
            
            conf_gap_improved = abs(after_stats['confidence_gap']) < abs(before_stats['confidence_gap'])
            print(f"  Confidence bias reduced: {'YES' if conf_gap_improved else 'NO'}")
    
    else:
        # Multiple experiments - show summary table
        print(f"\nSUMMARY TABLE:")
        print(f"{'Experiment':<20} {'Male_Acc':<8} {'Female_Acc':<10} {'Gap':<6} {'Overall':<8}")
        print("-" * 55)
        
        for exp_name, stats in summary_stats.items():
            print(f"{exp_name:<20} {stats['male_accuracy']:<8.3f} {stats['female_accuracy']:<10.3f} "
                  f"{stats['gender_gap']:<6.3f} {stats['overall_accuracy']:<8.3f}")

if __name__ == '__main__':
    model_path = './llama-3.2-3B'
    save_path = './llama-3.2-3B_memit'
    #model_path = './gpt2'
   # save_path = './gpt2_memit'

    # to actually apply memit, uncomment the following lines
    # results, results_base = run_experiment(model_path, save_path, is_llama=True, test_base_model=True)
    # comparison = analyze_results({
    #     "Original Model": results_base,
    #     "After MEMIT": results
    # })

    winobias_path = "./dataset/winobias/WinoBias/wino/data/anti_stereotyped_type1.txt.dev" # anti-stereotyped dataset has been used for validation -- hyperparams search

    #baseline_results, memit_results, baseline_metrics, memit_metrics = compare_models(model_path, save_path, winobias_path) 

    winobias_path_test = "./dataset/winobias/WinoBias/wino/data/pro_stereotyped_type1.txt.dev" # pro-stereotyped dataset has been used for test and analysis

    baseline_results, memit_results, baseline_metrics, memit_metrics = compare_models(model_path, save_path, winobias_path_test) 

