"""
MoodTune V11.5 - Hybrid Recommender Engine
Features:
- UI: 2-Column Menu, Color CLI.
- Logic: Hybrid Scoring (Audio Match + Popularity + Recency).
- Fix: Prioritizes Famous Tracks from 2010-2025.
"""
import os
import sys
import time
import webbrowser
import random
import numpy as np
import pandas as pd
from joblib import load
from sklearn.neighbors import NearestNeighbors

# --- Cross-Platform Key Input ---
try:
    import msvcrt
    def get_key():
        key = msvcrt.getch()
        if key == b'\xe0':
            key = msvcrt.getch()
            if key == b'M': return 'right'
            if key == b'K': return 'left'
        if key == b'\r': return 'enter'
        return None
except ImportError:
    import tty
    import termios
    def get_key():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
            if ch == '\x1b':
                ch = sys.stdin.read(2)
                if ch == '[C': return 'right'
                if ch == '[D': return 'left'
            if ch == '\r' or ch == '\n': return 'enter'
            return None
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "data", "models")
DB_FILE = os.path.join(MODEL_DIR, "tracks_clustered.parquet")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.joblib")

class Colors:
    HEADER = '\033[36m'  # Cyan
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    BAR_FILLED = '\033[36m━\033[0m'
    BAR_EMPTY = '\033[90m━\033[0m'
    THUMB = '\033[97m●\033[0m'

class MoodTuneRecommender:
    def __init__(self):
        self.df = None
        self.scaler = None
        self.feature_cols = [
            'danceability', 'energy', 'valence', 'tempo', 
            'acousticness', 'instrumentalness', 'liveness', 
            'loudness', 'speechiness'
        ]
        # Weights: Energy/Valence/Dance are key
        self.weights = np.array([1.2, 1.5, 1.5, 0.8, 1.0, 0.5, 0.5, 0.8, 0.2])

    def load_data(self):
        if not os.path.exists(DB_FILE):
            raise FileNotFoundError("Missing database! Run Option 3 in Manage.")
        
        self.df = pd.read_parquet(DB_FILE)
        self.scaler = load(SCALER_FILE)
        
        # Clean strings
        for c in ['track_name', 'artists', 'track_genre']:
            if c in self.df.columns:
                self.df[c] = self.df[c].astype(str).fillna('Unknown')
                
        # Ensure Year/Popularity exist
        if 'year' not in self.df.columns: self.df['year'] = 0
        if 'popularity' not in self.df.columns: self.df['popularity'] = 0

    def predict_features(self, inputs):
        """Maps 0-10 sliders to normalized audio vector."""
        try:
            n_energy = float(inputs[0]) / 10.0
            n_valence = float(inputs[1]) / 10.0
            n_dance = float(inputs[2]) / 10.0
            n_style = float(inputs[3]) / 10.0
        except:
            n_energy, n_valence, n_dance, n_style = 0.5, 0.5, 0.5, 0.5

        target_acoustic = 1.0 - n_style
        target_loudness = -35 + (n_energy * 30)
        target_tempo = 60 + (n_energy * 120)
        
        vector = np.array([[
            n_dance,        # danceability
            n_energy,       # energy
            n_valence,      # valence
            target_tempo,   # tempo
            target_acoustic,# acousticness
            0.1,            # instrumentalness
            0.15,           # liveness
            target_loudness,# loudness
            0.05            # speechiness
        ]])
        
        return self.scaler.transform(vector)

    def recommend(self, mood_vector, genre_pref=0, limit=12):
        """
        Hybrid Recommender:
        1. Filters by Genre.
        2. Finds nearest neighbors (Audio Match).
        3. Re-ranks based on Popularity and Recency (2010-2025).
        """
        filtered_df = self.df.copy()
        genre_str = str(genre_pref)
        
        # 1. Strict Genre Filter
        if genre_str != "0":
            keywords = []
            if genre_str == "1": keywords = ['hip-hop', 'rap', 'trap']
            elif genre_str == "2": keywords = ['pop', 'indie']
            elif genre_str == "3": keywords = ['rock', 'metal', 'punk']
            elif genre_str == "4": keywords = ['country', 'folk']
            elif genre_str == "5": keywords = ['edm', 'house', 'techno', 'dance']
            elif genre_str == "6": keywords = ['latin', 'reggaeton']
            elif genre_str == "7": keywords = ['k-pop', 'korean']
            elif genre_str == "8": keywords = ['r&b', 'soul']
            elif genre_str == "9": keywords = ['lo-fi', 'chillhop', 'jazzhop', 'study', 'sleep']
            elif genre_str == "10": keywords = ['classical', 'piano', 'orchestra', 'baroque', 'cinematic']
            
            pattern = '|'.join(keywords)
            mask = filtered_df['track_genre'].str.contains(pattern, case=False, na=False)
            filtered_df = filtered_df[mask]
            
            if filtered_df.empty: filtered_df = self.df.copy()

        # 2. Wide Search (Fetch 5x limit to allow re-ranking)
        search_limit = min(limit * 5, len(filtered_df))
        
        subset = filtered_df[self.feature_cols].values
        subset_scaled = self.scaler.transform(subset)
        
        # Weighted Search
        subset_weighted = subset_scaled * self.weights
        query_weighted = mood_vector * self.weights
        
        nn = NearestNeighbors(n_neighbors=search_limit, metric='euclidean')
        nn.fit(subset_weighted)
        
        dists, indices = nn.kneighbors(query_weighted)
        
        candidates = []
        seen = set()
        
        for i in range(len(indices[0])):
            idx = indices[0][i]
            dist = dists[0][i]
            row = filtered_df.iloc[idx]
            
            artist = row['artists']
            if artist in seen: continue # Basic artist deduplication in candidates
            seen.add(artist)

            # --- HYBRID SCORING LOGIC ---
            
            # A. Audio Score (0-100) - Based on closeness
            # dist is usually 0.0 to 2.0.
            audio_score = max(0, 100 - (dist * 15)) 
            
            # B. Popularity Score (0-100)
            pop_score = float(row.get('popularity', 0))
            
            # C. Recency Boost (Specific to 2010-2025 request)
            year = int(row.get('year', 0))
            year_boost = 0
            if 2010 <= year <= 2025:
                year_boost = 20  # Big boost for modern era
            elif year > 0 and year < 2010:
                year_boost = -10 # Slight penalty for older tracks
            
            # Final Hybrid Score
            # 50% Audio Match, 30% Fame, 20% Year + Boosts
            final_score = (audio_score * 0.5) + (pop_score * 0.3) + year_boost
            
            candidates.append({
                'track_name': self.safe(row['track_name']),
                'artist_name': self.safe(row['artists']),
                'track_id': row['track_id'],
                'genre': row['track_genre'],
                'bpm': int(row['tempo']) if 'tempo' in row else 0,
                'year': year,
                'audio_match': round(audio_score, 1),
                'final_score': final_score
            })
            
        # 3. Sort by Hybrid Score (Descending) & Crop
        candidates.sort(key=lambda x: x['final_score'], reverse=True)
        return candidates[:limit]

    def safe(self, txt):
        try: return str(txt).strip()
        except: return "?"

    def generate_report(self, inputs, name="User"):
        energy = float(inputs[0])
        happiness = float(inputs[1])
        dance = float(inputs[2])
        style = float(inputs[3]) # 0=Acoustic, 10=Electronic
        
        # State determination logic
        state = "feeling balanced"
        genre_rec = "Pop or R&B"

        # Check Combinations for deeper nuance
        if energy >= 7:
            if happiness >= 7:
                state = "feeling euphoric and full of energy"
                genre_rec = "Upbeat Pop, EDM, or Funk"
            elif happiness <= 3:
                state = "feeling intense and needing a release"
                genre_rec = "Hard Rock, Trap, or Heavy Metal"
            else:
                if dance >= 7:
                    state = "wanting to dance and let loose"
                    genre_rec = "House, Disco, or Club Hits"
                else:
                    state = "feeling active and driven"
                    genre_rec = "Rock or Upbeat Indie"
        elif energy <= 3:
            if happiness >= 7:
                state = "feeling chill and content"
                genre_rec = "Soul, Acoustic Pop, or Reggae"
            elif happiness <= 3:
                state = "feeling melancholy or reflective"
                genre_rec = "Sad Indie, Ambient, or Slow Ballads"
            else:
                # Low energy, mid happiness -> check style/intent
                if style <= 3:
                    state = "trying to focus or study"
                    genre_rec = "Classical, Lofi, or Acoustic Instrumental"
                else:
                    state = "looking to relax and unwind"
                    genre_rec = "Downtempo, Chillhop, or Ambient"
        else:
            # Mid energy
            if dance >= 7:
                state = "in the mood to groove"
                genre_rec = "R&B, Latin, or Funk"
            elif style >= 7:
                state = "looking for a modern vibe"
                genre_rec = "Synthpop or Alt-Pop"
            else:
                state = "cruising through the day"
                genre_rec = "Indie Rock or Alternative"

        return f"It looks like you are {state}, so you should listen to {genre_rec}."

    def generate_oneliner(self, inputs):
        e, h = inputs[0], inputs[1]
        if e > 7 and h > 7: return "High Voltage Happy"
        if e < 3 and h < 3: return "Deep Melancholy"
        if e < 4 and h > 6: return "Chill & Cheerful"
        if e > 7 and h < 4: return "Dark & Stormy"
        return "Balanced Flow"

# --- CLI Functions ---
def type_text(text, delay=0.01):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def get_slider_input(prompt, descriptions, width=20):
    value = 5
    print(f"\n{Colors.BOLD}{prompt}{Colors.ENDC}")
    print(f"{Colors.BLUE}Use Arrow keys. Enter to confirm.{Colors.ENDC}")
    sys.stdout.write("\033[?25l") 
    while True:
        filled_len = int(round((value) / 10.0 * width))
        bar = Colors.BAR_FILLED * filled_len + Colors.THUMB + Colors.BAR_EMPTY * (width - filled_len)
        desc_text = descriptions[value]
        sys.stdout.write(f"\r   {bar}   {Colors.HEADER}[{value}] {desc_text:<15}{Colors.ENDC}   ")
        sys.stdout.flush()
        key = get_key()
        if key == 'left' and value > 0: value -= 1
        elif key == 'right' and value < 10: value += 1
        elif key == 'enter': break
    sys.stdout.write("\033[?25h\n") 
    return value

def get_menu_input(prompt, options):
    print(f"\n{Colors.BOLD}{prompt}{Colors.ENDC}")
    keys = sorted(options.keys())
    half = (len(keys) + 1) // 2
    for i in range(half):
        k1 = keys[i]
        v1 = options[k1]
        col1 = f"[{k1}] {v1}"
        if i + half < len(keys):
            k2 = keys[i + half]
            v2 = options[k2]
            col2 = f"[{k2}] {v2}"
            print(f"{col1:<30} {col2}")
        else:
            print(col1)

    while True:
        try:
            val = input(f"{Colors.CYAN}➤ Choice: {Colors.ENDC}")
            if int(val) in options: return int(val)
        except ValueError: pass

def cli_main():
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"{Colors.HEADER}{Colors.BOLD}")
    print(r"""
  __  __                 _ _______                  
 |  \/  |               | |__   __|                 
 | \  / | ___   ___   __| |  | |_   _ _ __   ___    
 | |\/| |/ _ \ / _ \ / _` |  | | | | | '_ \ / _ \   
 | |  | | (_) | (_) | (_| |  | | |_| | | | |  __/   
 |_|  |_|\___/ \___/ \__,_|  |_|\__,_|_| |_|\___|   
                                                    
    AI-Powered Music Discovery V11.5
    """)
    print(f"{Colors.ENDC}")
    
    rec = MoodTuneRecommender()
    try:
        print(f"{Colors.CYAN}⏳ Loading MoodTune Database...{Colors.ENDC}")
        rec.load_data()
        print(f"{Colors.GREEN}✅ System Ready: {len(rec.df):,} tracks loaded.{Colors.ENDC}")
    except Exception as e:
        print(f"{Colors.FAIL}Error: {e}{Colors.ENDC}")
        return

    while True:
        print(f"\n{Colors.HEADER}--- New Session ---{Colors.ENDC}")
        
        energy = get_slider_input("⚡ Step 1/5: Energy Level", ["Deep Sleep", "Very Low", "Chill", "Relaxed", "Mellow", "Balanced", "Upbeat", "Active", "High Energy", "Intense", "Explosive"])
        valence = get_slider_input("😊 Step 2/5: Current Mood", ["Despair", "Melancholy", "Sad", "Somber", "Moody", "Neutral", "Content", "Cheerful", "Happy", "Joyful", "Euphoric"])
        dance = get_slider_input("💃 Step 3/5: Rhythm", ["Static", "Ambient", "Swaying", "Gentle", "Steady", "Groovy", "Bouncy", "Danceable", "Driving", "Club Ready", "Rave"])
        style = get_slider_input("🎸 Step 4/5: Sound Style", ["Pure Acoustic", "Organic", "Raw", "Unplugged", "Natural", "Hybrid", "Processed", "Electric", "Synthesized", "Digital", "Futuristic"])
        
        genres = {
            0: "Surprise Me", 
            1: "Hip-Hop/Rap", 
            2: "Pop", 
            3: "Rock", 
            4: "Country", 
            5: "EDM", 
            6: "Latin", 
            7: "K-pop", 
            8: "R&B",
            9: "Lofi / Chillhop",
            10: "Classical"
        }
        genre = get_menu_input("🎵 Step 5/5: Genre Preference", genres)

        print(f"\n{Colors.WARNING}🔮 Analyzing audio features...{Colors.ENDC}")
        type_text("................................", 0.02)
        
        inputs = [energy, valence, dance, style, genre]
        mood_vec = rec.predict_features(inputs)
        report = rec.generate_report(inputs)
        
        recs = rec.recommend(mood_vec, genre_pref=genre, limit=12)
        
        print(f"\n{Colors.GREEN}✨ YOUR MOOD MIX ✨{Colors.ENDC}")
        print(f"{Colors.HEADER}{report}{Colors.ENDC}")
        print("-" * 80)
        # Removed Year column from output
        print(f"{'#':<4} {'Match':<6} {'BPM':<5} {'Track Title':<25} {'Artist':<15} {'Genre':<12}")
        print("-" * 80)
        
        for idx, r in enumerate(recs):
            score_display = f"{int(r['audio_match'])}%"
            bpm = str(r['bpm'])
            title = (r['track_name'][:22] + '..') if len(r['track_name']) > 22 else r['track_name']
            artist = (r['artist_name'][:12] + '..') if len(r['artist_name']) > 12 else r['artist_name']
            genre = (r['genre'][:10] + '..') if len(r['genre']) > 10 else r['genre']
            
            print(f"{Colors.BOLD}[{idx+1}]{Colors.ENDC}  {Colors.CYAN}{score_display:<6}{Colors.ENDC} {bpm:<5} {Colors.BOLD}{title:<25}{Colors.ENDC} {artist:<15} {genre:<12}")
        print("-" * 80)
        
        while True:
            print(f"\n{Colors.BLUE}Actions: [1-12] Play | [r] Reshuffle | [n] New | [q] Quit{Colors.ENDC}")
            choice = input(f"{Colors.CYAN}➤ {Colors.ENDC}").lower().strip()
            if choice == 'q': return
            elif choice == 'n': break
            elif choice == 'r':
                print(f"{Colors.WARNING}Reshuffling...{Colors.ENDC}")
                wide = rec.recommend(mood_vec, genre_pref=genre, limit=50)
                recs = random.sample(wide, min(12, len(wide)))
                print("-" * 80)
                for idx, r in enumerate(recs):
                    score_display = f"{int(r['audio_match'])}%"
                    bpm = str(r['bpm'])
                    title = (r['track_name'][:22] + '..') if len(r['track_name']) > 22 else r['track_name']
                    artist = (r['artist_name'][:12] + '..') if len(r['artist_name']) > 12 else r['artist_name']
                    genre = (r['genre'][:10] + '..') if len(r['genre']) > 10 else r['genre']
                    print(f"{Colors.BOLD}[{idx+1}]{Colors.ENDC}  {Colors.CYAN}{score_display:<6}{Colors.ENDC} {bpm:<5} {Colors.BOLD}{title:<25}{Colors.ENDC} {artist:<15} {genre:<12}")
                print("-" * 80)
            elif choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(recs):
                    webbrowser.open(f"https://open.spotify.com/track/{recs[idx]['track_id']}")

if __name__ == "__main__":
    cli_main()