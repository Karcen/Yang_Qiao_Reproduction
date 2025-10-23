import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def generate_simulated_peptide_data(n_samples=1800):
    """Generate simulated peptide dataset with structural features"""
    
    # Amino acid pool
    amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 
                   'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    
    # Generate random peptide sequences
    peptide_lengths = np.random.randint(2, 15, size=n_samples)
    sequences = []
    for length in peptide_lengths:
        seq = ''.join(np.random.choice(amino_acids, size=length))
        sequences.append(seq)
    
    # Create features
    data = pd.DataFrame({
        'peptide_sequence': sequences,
        'length': peptide_lengths,
        'contains_tyrosine': [1 if 'Y' in seq else 0 for seq in sequences],
        'n_tyrosine': [seq.count('Y') for seq in sequences],
        'n_phenylalanine': [seq.count('F') for seq in sequences],
        'n_proline': [seq.count('P') for seq in sequences],
        'n_glutamine': [seq.count('Q') for seq in sequences],
        'n_tryptophan': [seq.count('W') for seq in sequences],
        'n_arginine': [seq.count('R') for seq in sequences],
        'n_lysine': [seq.count('K') for seq in sequences],
        'n_alanine': [seq.count('A') for seq in sequences],
        'n_leucine': [seq.count('L') for seq in sequences],
        'n_isoleucine': [seq.count('I') for seq in sequences],
        'n_valine': [seq.count('V') for seq in sequences],
        'n_methionine': [seq.count('M') for seq in sequences],
        'n_serine': [seq.count('S') for seq in sequences],
        'n_threonine': [seq.count('T') for seq in sequences],
        'n_cysteine': [seq.count('C') for seq in sequences],
        'n_histidine': [seq.count('H') for seq in sequences],
        'n_aspartic_acid': [seq.count('D') for seq in sequences],
        'n_glutamic_acid': [seq.count('E') for seq in sequences],
        'n_glycine': [seq.count('G') for seq in sequences],
        'n_asparagine': [seq.count('N') for seq in sequences],
        'n_terminal_tyrosine': [1 if seq.startswith('Y') else 0 for seq in sequences],
        'c_terminal_proline': [1 if seq.endswith('P') else 0 for seq in sequences],
        'c_terminal_second_glutamine': [1 if len(seq)>=2 and seq[-2]=='Q' else 0 for seq in sequences],
        'yp_type': [1 if seq.startswith('YP') else 0 for seq in sequences],
        'yl_type': [1 if seq.startswith('YL') else 0 for seq in sequences],
        'yi_type': [1 if seq.startswith('YI') else 0 for seq in sequences],
        'yq_type': [1 if seq.startswith('YQ') else 0 for seq in sequences],
    })
    
    # Generate sleep-enhancing activity based on features (mimicking paper findings)
    base_activity = 0.2 + 0.01 * data['length']
    
    # Tyr-based effects (strongest contributors)
    tyr_effect = 0.3 * data['contains_tyrosine'] + 0.1 * data['n_tyrosine']
    tyr_effect += 0.4 * data['n_terminal_tyrosine']
    
    # Specific peptide types
    type_effect = 0.6 * data['yp_type'] + 0.55 * data['yl_type'] + 0.5 * data['yi_type'] + 0.45 * data['yq_type']
    
    # Length effects (optimal 4-10)
    length_effect = np.where((data['length'] >=4) & (data['length'] <=10), 0.2, 0)
    
    # C-terminal effects
    c_term_effect = 0.15 * data['c_terminal_proline'] + 0.1 * data['c_terminal_second_glutamine']
    
    # Random noise
    noise = np.random.normal(0, 0.1, size=n_samples)
    
    # Total activity
    data['sleep_activity'] = base_activity + tyr_effect + type_effect + length_effect + c_term_effect + noise
    
    # Scale activity to 0-1 range
    scaler = MinMaxScaler()
    data['sleep_activity'] = scaler.fit_transform(data[['sleep_activity']])
    
    return data

def generate_sample_metadata(n_samples=6):
    """Generate metadata for casein hydrolysate samples"""
    return pd.DataFrame({
        'sample_id': [f'Sample_{i+1}' for i in range(n_samples)],
        'treatment': np.random.choice(['Control', 'Enzymatic', 'Acid', 'Alkaline'], size=n_samples),
        'digestion_time_h': np.random.uniform(1, 24, size=n_samples),
        'temperature_c': np.random.uniform(30, 60, size=n_samples),
        'ph': np.random.uniform(2, 10, size=n_samples),
        'avg_sleep_duration_min': np.random.uniform(240, 480, size=n_samples)
    })

def generate_digestion_data(n_entries=2800):
    """Generate simulated digestion data"""
    return pd.DataFrame({
        'digestion_id': [f'Digest_{i+1}' for i in range(n_entries)],
        'sample_id': np.random.choice([f'Sample_{i+1}' for i in range(6)], size=n_entries),
        'peptide_sequence': generate_simulated_peptide_data(n_entries)['peptide_sequence'],
        'concentration_mM': np.random.exponential(0.5, size=n_entries),
        'retention_time_min': np.random.uniform(5, 60, size=n_entries),
        'm_z': np.random.uniform(300, 1500, size=n_entries)
    })

if __name__ == "__main__":
    # Generate and save datasets
    peptide_data = generate_simulated_peptide_data()
    peptide_data.to_csv('enhanced_peptides_data.csv', index=False)
    
    metadata = generate_sample_metadata()
    metadata.to_csv('samples_metadata.csv', index=False)
    
    digestion_data = generate_digestion_data()
    digestion_data.to_csv('digestion_data.csv', index=False)
    
    print("Simulated datasets generated successfully:")
    print(f"- enhanced_peptides_data.csv: {len(peptide_data)} entries")
    print(f"- samples_metadata.csv: {len(metadata)} entries")
    print(f"- digestion_data.csv: {len(digestion_data)} entries")