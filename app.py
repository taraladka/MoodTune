"""
MoodTune Web Server
Bridges the HTML5 Interface with the Python Recommender Engine
"""
import traceback
from flask import Flask, render_template, request, jsonify
from moodtune_recommender import MoodTuneRecommender
import time

app = Flask(__name__, template_folder='templates', static_folder='static')

recommender = None

def get_recommender():
    global recommender
    if recommender is None:
        try:
            print("⏳ Loading Database (This takes ~10s)...")
            recommender = MoodTuneRecommender()
            recommender.load_data()
            print("✅ Database Loaded Successfully!")
        except Exception as e:
            print(f"Initialization Error: {e}")
            return None
    return recommender

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/recommend", methods=["POST"])
def recommend():
    rec = get_recommender()
    if not rec:
        return jsonify({"status": "error", "message": "Server initializing..."}), 503

    try:
        data = request.json
        user_name = data.get('name', 'User')
        
        inputs = [
            data.get('energy', 5),
            data.get('happiness', 5),
            data.get('dance', 5),
            data.get('style', 5),
            data.get('genre', 0)
        ]
        
        mood_vector = rec.predict_features(inputs)
        mood_report = rec.generate_report(inputs, name=user_name)
        one_liner = rec.generate_oneliner(inputs)
        
        results = rec.recommend(
            mood_vector, 
            genre_pref=inputs[4],
            limit=60
        )
        
        response_data = []
        for r in results:
            score = r.get('audio_match', r.get('score', 0))
            
            # Removed Year Logic entirely
            
            response_data.append({
                "name": r['track_name'],
                "artist": r['artist_name'],
                "id": r['track_id'],
                "genre": r['genre'],
                "match_score": float(score) / 100.0 
            })

        return jsonify({
            "status": "success", 
            "recommendations": response_data, 
            "mood_report": mood_report,
            "one_liner": one_liner
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    get_recommender()
    print("🚀 Server Ready! Open this link: http://127.0.0.1:5000")
    app.run(debug=True, port=5000)