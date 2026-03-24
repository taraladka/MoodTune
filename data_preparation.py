"""
MoodTune Data Preparation V10.0 (Interactive & Smart Filtering)
Features:
1. Interactive Menu: Asks user which CSVs to load from 'dataset/' folder.
2. Smart Type Detection: Auto-detects if a CSV is 'Rich' (Audio) or 'Meta' (Text) based on columns.
3. Genre Consistency Filter: Removes tracks where the labeled genre conflicts with sub-genre metadata.
4. Genre Aliasing: Maps 'lofi', 'chillhop', 'piano' to core profiles.
"""
import os
import ast
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from joblib import dump

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_MODELS_DIR = os.path.join(BASE_DIR, "data", "models")
OUTPUT_FILE = os.path.join(DATA_MODELS_DIR, "tracks_clustered.parquet")
SCALER_FILE = os.path.join(DATA_MODELS_DIR, "scaler.joblib")
KMEANS_FILE = os.path.join(DATA_MODELS_DIR, "kmeans.joblib")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# Required features for the AI
FEATURE_COLS = [
    'danceability', 'energy', 'valence', 'tempo', 
    'acousticness', 'instrumentalness', 'liveness', 
    'loudness', 'speechiness'
]

GENRE_ALIASES = {
    'lofi': 'lo-fi',
    'chillhop': 'lo-fi',
    'jazzhop': 'lo-fi',
    'study': 'lo-fi',
    'orchestral': 'classical',
    'piano': 'classical',
    'baroque': 'classical',
    'hip hop': 'hip-hop',
    'rap': 'hip-hop'
}

def ensure_directories():
    if not os.path.exists(DATA_MODELS_DIR):
        os.makedirs(DATA_MODELS_DIR)
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)

def get_available_datasets():
    """Scans the dataset folder for CSVs."""
    files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.csv')]
    return sorted(files)

def ask_user_for_datasets():
    """Interactive prompt to select files."""
    files = get_available_datasets()
    if not files:
        print("❌ No CSV files found in 'dataset/' folder!")
        return []

    print("\n📂 Available Datasets:")
    for i, f in enumerate(files):
        print(f"  [{i+1}] {f}")
    
    print("\n👉 Enter the numbers of the datasets you want to TRAIN on (comma separated).")
    print("   Example: 1, 3")
    choice = input("   Selection: ").strip()
    
    selected_files = []
    try:
        indices = [int(x.strip()) for x in choice.split(',') if x.strip().isdigit()]
        for idx in indices:
            if 1 <= idx <= len(files):
                selected_files.append(files[idx-1])
    except ValueError:
        pass
        
    return selected_files

def validate_genre_consistency(df):
    """
    Filters out rows where 'track_genre' does not align with 'genres' list.
    Only applies to datasets that have a 'genres' column (like Dataset 3).
    """
    if 'genres' not in df.columns:
        return df

    print(f"      ...Validating Genre Consistency on {len(df)} tracks...")
    
    def is_consistent(row):
        main = str(row['track_genre']).lower().strip()
        try:
            # Parse string representation of list "['pop', 'uk pop']"
            sub_genres = ast.literal_eval(row['genres'])
            sub_genres = [s.lower().strip() for s in sub_genres]
            
            # 1. Exact match check
            if main in sub_genres: return True
            
            # 2. Substring check (e.g. main="pop" fits in "dance pop")
            for sub in sub_genres:
                if main in sub or sub in main:
                    return True
            
            # 3. Alias check
            if main in GENRE_ALIASES:
                target = GENRE_ALIASES[main]
                for sub in sub_genres:
                    if target in sub: return True

            return False
        except:
            return True # Keep row if parsing fails (safe fallback)

    initial_count = len(df)
    # Apply filter
    df = df[df.apply(is_consistent, axis=1)]
    removed = initial_count - len(df)
    if removed > 0:
        print(f"      ✂️  Removed {removed} tracks due to Genre Mismatch.")
    
    return df

def process_datasets(selected_files):
    rich_dfs = []
    meta_dfs = []

    col_map = {
        'genre': 'track_genre', 
        'main_genre': 'track_genre',
        'artist_name': 'artists', 
        'artist': 'artists',
        'Artist': 'artists',
        'track_name': 'track_name', 
        'Artist and Title': 'track_name_compound', # Temporary
        'track_id': 'track_id',
        'popularity': 'popularity',
        'key': 'key', 'mode': 'mode', 'time_signature': 'time_signature'
    }

    print("\n🔄 Processing Selected Files...")

    for f in selected_files:
        path = os.path.join(DATASET_DIR, f)
        try:
            df = pd.read_csv(path, low_memory=False)
            
            # 1. Standardize Columns
            for old, new in col_map.items():
                if old in df.columns:
                    df.rename(columns={old: new}, inplace=True)

            # 2. Fix Compound Names (Dataset 3 specific)
            if 'track_name_compound' in df.columns and 'track_name' not in df.columns:
                 df['track_name'] = df['track_name_compound'].astype(str).apply(
                    lambda x: x.split(' - ')[1] if ' - ' in x else x
                )

            # 3. Detect Type
            # If it has danceability AND energy, it's Rich. Else, it's Meta.
            if 'danceability' in df.columns and 'energy' in df.columns:
                print(f"   -> [{f}] identified as RICH AUDIO dataset.")
                # Ensure numerics
                for c in FEATURE_COLS:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
                df.dropna(subset=FEATURE_COLS, inplace=True)
                rich_dfs.append(df)
            else:
                print(f"   -> [{f}] identified as METADATA dataset (needs imputation).")
                # Apply Genre Consistency Filter here
                df = validate_genre_consistency(df)
                meta_dfs.append(df)

        except Exception as e:
            print(f"   ❌ Error reading {f}: {e}")

    # Combine Rich Data
    df_rich = pd.concat(rich_dfs, ignore_index=True) if rich_dfs else pd.DataFrame()
    
    if df_rich.empty:
        print("⚠️  Warning: No RICH datasets loaded. Cannot learn audio profiles!")
        return pd.DataFrame()

    # Create Genre Profiles
    print("   -> Learning Audio Profiles from Rich Data...")
    df_rich['track_genre_clean'] = df_rich['track_genre'].astype(str).str.lower().str.strip()
    genre_profiles = df_rich.groupby('track_genre_clean')[FEATURE_COLS].mean()
    profile_dict = genre_profiles.to_dict('index')
    global_mean = genre_profiles.mean(numeric_only=True)

    # Impute Metadata Data
    if meta_dfs:
        print("   -> Imputing Audio Features for Metadata Datasets...")
        imputed_dfs = []
        for df in meta_dfs:
            # Init columns
            for c in FEATURE_COLS:
                df[c] = np.nan
            
            def get_feature(row, feature):
                g = str(row['track_genre']).lower().strip()
                # Alias check
                if g in GENRE_ALIASES: g = GENRE_ALIASES[g]
                
                # Direct
                if g in profile_dict: return profile_dict[g][feature]
                
                # Fuzzy
                for pg in profile_dict:
                    if pg in g or g in pg: return profile_dict[pg][feature]
                
                return global_mean[feature]

            for col in FEATURE_COLS:
                df[col] = df.apply(lambda x: get_feature(x, col), axis=1)
            
            # Dummy ID if missing
            if 'track_id' not in df.columns:
                df['track_id'] = [f"gen_{i}" for i in range(len(df))]
                
            imputed_dfs.append(df)
        
        df_meta_final = pd.concat(imputed_dfs, ignore_index=True)
        return pd.concat([df_rich, df_meta_final], ignore_index=True)
    
    return df_rich

def clean_and_save():
    # 1. Ask User
    selected = ask_user_for_datasets()
    if not selected:
        print("❌ No datasets selected. Aborting.")
        return

    # 2. Process
    df_final = process_datasets(selected)
    
    if df_final.empty:
        print("❌ Final dataframe is empty. Nothing to save.")
        return

    # 3. Final Cleanup
    print(f"\n✨ Finalizing Database with {len(df_final)} tracks...")
    df_final.drop_duplicates(subset=['track_name', 'artists'], inplace=True)
    
    # Force String Types (Parquet Safety)
    str_cols = ['track_id', 'track_name', 'artists', 'track_genre', 'key', 'mode', 'time_signature']
    for col in str_cols:
        if col in df_final.columns:
            df_final[col] = df_final[col].astype(str).fillna('Unknown')

    keep_cols = FEATURE_COLS + str_cols + ['popularity', 'cluster']
    
    # 4. Train Models
    print("🧠 Training AI Models (Scaler & KMeans)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_final[FEATURE_COLS])
    
    kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
    df_final['cluster'] = kmeans.fit_predict(X_scaled)

    # 5. Save
    print("💾 Saving files to 'data/models/'...")
    
    # Filter columns to only what we need to save space/errors
    final_cols = [c for c in keep_cols if c in df_final.columns]
    df_final[final_cols].to_parquet(OUTPUT_FILE, index=False)
    
    dump(scaler, SCALER_FILE)
    dump(kmeans, KMEANS_FILE)
    print("✅ Success! Database Ready.")

if __name__ == "__main__":
    ensure_directories()
    try:
        clean_and_save()
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter...")