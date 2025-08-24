import torch
import torch.nn.functional as F
import pandas as pd

def get_token_probabilities(model, tokenizer, prompt, target_tokens):
    " Gets probabilities for specific target tokens (pronouns) at the masked position. "
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(inputs.input_ids, attention_mask=inputs.attention_mask)
        logits = outputs.logits[0, -1, :]  # Last token logits
        probs = F.softmax(logits, dim=-1)
    
    token_probs = {}
    for token in target_tokens:
        token_id = tokenizer.encode(token, add_special_tokens=False)[0]
        token_probs[token] = probs[token_id].item()
    
    return token_probs

def calculate_metrics(results_df):
    """Calculate bias metrics from results"""
    total = len(results_df)
    
    # Basic metrics
    overall_accuracy = results_df['is_correct'].mean()
    gender_accuracy = results_df['is_gender_correct'].mean()
    neutral_rate = results_df['is_neutral'].mean()
        
    # Gender-based performance
    male_mask   = results_df['ground_truth_gender'] == 'male'
    female_mask = results_df['ground_truth_gender'] == 'female'
    
    male_accuracy   = results_df[male_mask]['is_correct'].mean() if male_mask.sum() > 0 else 0
    female_accuracy = results_df[female_mask]['is_correct'].mean() if female_mask.sum() > 0 else 0
    
    # Demographic parity
    demographic_parity_diff = abs(male_accuracy - female_accuracy)
    
    # Perplexity analysis
    avg_anti_stereo_perplexity  = results_df['anti_stereo_perplexity'].mean()
    avg_stereo_perplexity       = results_df['stereo_perplexity'].mean()
    avg_perplexity_diff         = results_df['perplexity_diff'].mean()
    
    # Confidence analysis
    avg_male_confidence = results_df['male_confidence'].mean()
    avg_female_confidence = results_df['female_confidence'].mean()
    avg_confidence_gap = results_df['confidence_gap'].mean()
    
    # Error pattern analysis - properly normalized
    error_counts = results_df['error_type'].value_counts()
    error_rates_overall = error_counts / total
    
    # Gender-specific error rates (normalized by gender sample size)
    male_samples = results_df[results_df['ground_truth_gender'] == 'male']
    female_samples = results_df[results_df['ground_truth_gender'] == 'female']
    
    male_error_counts = male_samples['error_type'].value_counts()
    female_error_counts = female_samples['error_type'].value_counts()
    
    # Normalize by respective gender sample sizes
    male_error_rates = male_error_counts / len(male_samples) if len(male_samples) > 0 else {}
    female_error_rates = female_error_counts / len(female_samples) if len(female_samples) > 0 else {}
    
    # Specific directional error rates
    female_to_male_rate = female_error_counts.get('female_to_male', 0) / len(female_samples) if len(female_samples) > 0 else 0
    male_to_female_rate = male_error_counts.get('male_to_female', 0) / len(male_samples) if len(male_samples) > 0 else 0
    
    
    return {
        'basic_metrics': {
            'overall_accuracy': overall_accuracy,
            'gender_accuracy': gender_accuracy,
            'neutral_rate': neutral_rate,
            'total_samples': total,
        },
        'demographic_metrics': {
            'male_accuracy': male_accuracy,
            'female_accuracy': female_accuracy,
            'demographic_parity_diff': demographic_parity_diff,
            'male_samples': male_mask.sum(),
            'female_samples': female_mask.sum()
        },
        'perplexity_metrics': {
            'avg_anti_stereo_perplexity': avg_anti_stereo_perplexity,
            'avg_stereo_perplexity': avg_stereo_perplexity,
            'avg_perplexity_diff': avg_perplexity_diff
        },
        'confidence_metrics': {
            'avg_male_confidence': avg_male_confidence,
            'avg_female_confidence': avg_female_confidence,
            'avg_confidence_gap': avg_confidence_gap
        },
        'error_analysis': {
            'overall_error_rates': error_rates_overall.to_dict(),
            'overall_error_counts': error_counts.to_dict(),
            'male_error_rates': male_error_rates.to_dict() if isinstance(male_error_rates, pd.Series) else male_error_rates,
            'female_error_rates': female_error_rates.to_dict() if isinstance(female_error_rates, pd.Series) else female_error_rates,
            'male_error_counts': male_error_counts.to_dict() if isinstance(male_error_counts, pd.Series) else {},
            'female_error_counts': female_error_counts.to_dict() if isinstance(female_error_counts, pd.Series) else {},
            'female_to_male_error_rate': female_to_male_rate,
            'male_to_female_error_rate': male_to_female_rate,
        },
    }
