import pandas as pd
import json
import numpy as np

def prepare_anes_data(anes_file):
    """
    Load raw ANES CDF and prepare human reference distributions
    """
    # Load ANES cumulative file
    df = pd.read_csv(anes_file)

    # Variables available (adjust based on actual ANES file):
    # - VCF0218: Feeling Thermometer - Democratic Party (0-100)
    # - VCF0224: Feeling Thermometer - Republican Party (0-100)
    # - VCF0303: Party ID (1=Dem, 2=Rep, 3=Ind)
    # - VCF0110: Education (1=<HS, 2=HS, 3=Some College, 4=College, 5=Post-grad)
    # - VCF0105a: Age
    # - VCF0106: Race (1=White, 2=Black, 3=Asian, etc.)

    # Rename for clarity
    df_clean = df.rename(columns={
        'vcf0218': 'dem_thermometer',
        'vcf0224': 'rep_thermometer',
        'vcf0303': 'party_id',
        'vcf0110': 'education',
        'vcf0105a': 'age',
        'vcf0106': 'race'
    })

    # Remove missing values (typically coded as 98, 99)
    df_clean = df_clean[(df_clean['dem_thermometer'] >= 0) &
                        (df_clean['dem_thermometer'] <= 100)]

    # Create demographic groups
    personas_config = {
        'persona_001': {
            'name': 'Republican, HS Education',
            'filter': (df_clean['party_id'] == 2) & (df_clean['education'] == 2),
            'traits': {'party': 'Republican', 'education': 'high school'}
        },
        'persona_002': {
            'name': 'Democrat, College Educated',
            'filter': (df_clean['party_id'] == 1) & (df_clean['education'] == 4),
            'traits': {'party': 'Democrat', 'education': 'college'}
        },
        'persona_003': {
            'name': 'Independent, Graduate Degree',
            'filter': (df_clean['party_id'] == 3) & (df_clean['education'] == 5),
            'traits': {'party': 'Independent', 'education': 'graduate'}
        },
    }

    human_distributions = {}

    for p_id, p_info in personas_config.items():
        subset = df_clean[p_info['filter']]

        human_distributions[p_id] = {
            'persona_name': p_info['name'],
            'n_respondents': len(subset),
            'dem_thermometer': {
                'mean': float(subset['dem_thermometer'].mean()),
                'std': float(subset['dem_thermometer'].std()),
                'median': float(subset['dem_thermometer'].median()),
                'q25': float(subset['dem_thermometer'].quantile(0.25)),
                'q75': float(subset['dem_thermometer'].quantile(0.75)),
                'min': float(subset['dem_thermometer'].min()),
                'max': float(subset['dem_thermometer'].max()),
                'all_responses': subset['dem_thermometer'].tolist()  # Keep full dist!
            },
            'rep_thermometer': {
                'mean': float(subset['rep_thermometer'].mean()),
                'std': float(subset['rep_thermometer'].std()),
                'median': float(subset['rep_thermometer'].median()),
                'q25': float(subset['rep_thermometer'].quantile(0.25)),
                'q75': float(subset['rep_thermometer'].quantile(0.75)),
                'min': float(subset['rep_thermometer'].min()),
                'max': float(subset['rep_thermometer'].max()),
                'all_responses': subset['rep_thermometer'].tolist()
            }
        }

    # Save
    with open('data/human_survey/anes_distributions.json', 'w') as f:
        json.dump(human_distributions, f, indent=2)

    print("Human reference distributions saved!")
    print("\nSample distribution (persona_001):")
    print(json.dumps(human_distributions['persona_001'], indent=2, default=str))

    return human_distributions

if __name__ == "__main__":
    # Download from: https://electionstudies.org/data-center/anes-time-series-cumulative-data-file/
    prepare_anes_data('data/human_survey/anes_cdf_2024.csv')
