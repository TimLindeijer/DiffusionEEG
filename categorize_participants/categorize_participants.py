import pandas as pd

def main():
    # Path to participants.tsv
    participants_tsv = 'data/caueeg_bids/participants.tsv'
    
    # Read participants.tsv
    df = pd.read_csv(participants_tsv, sep='\t')
    
    # Determine participant ID column name
    participant_col = 'participant_id' if 'participant_id' in df.columns else 'id'
    
    # Handle missing values
    if df['ad_syndrome_3'].isnull().any():
        print("Warning: Missing values found in 'ad_syndrome_3'. These participants will be excluded.")
    
    # Categorization function
    def categorize(row):
        syndrome = row['ad_syndrome_3']
        if pd.isnull(syndrome):
            return None
        syndrome = str(syndrome).lower().strip()
        if syndrome == 'mci':
            return 'mci'
        elif syndrome in 'hc (+smc)':
            return 'hc_smc'
        elif syndrome == 'dementia':
            return 'dementia'
        return None
    
    # Apply categorization
    df['category'] = df.apply(categorize, axis=1)
    df = df.dropna(subset=['category'])
    
    # Create category groups
    categories = {
        'mci': df[df['category'] == 'mci'][participant_col].tolist(),
        'hc_smc': df[df['category'] == 'hc_smc'][participant_col].tolist(),
        'dementia': df[df['category'] == 'dementia'][participant_col].tolist()
    }
    
    # Save to files
    for category, subjects in categories.items():
        with open(f'categorize_participants/{category}_subjects.txt', 'w') as f:
            f.write('\n'.join(subjects))
        print(f"Saved {len(subjects)} subjects to {category}_subjects.txt")

if __name__ == "__main__":
    main()