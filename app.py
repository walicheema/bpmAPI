from flask import Flask, request, jsonify
import io
import librosa
import requests

app = Flask(__name__)

@app.route('/get_tempo', methods=['POST'])
def get_tempo():
    try:
        # Get the mp3URL from the request data
        mp3_url = request.json.get('mp3URL')

        if not mp3_url:
            return jsonify({'error': 'No mp3URL provided'}), 400

        # Download the MP3 file from the provided URL
        response = requests.get(mp3_url)
        if response.status_code != 200:
            return jsonify({'error': 'Failed to download MP3 file'}), 500

        # Load the audio data using librosa
        mp3_data = io.BytesIO(response.content)
        y, sr = librosa.load(mp3_data, sr=44100)

        # Calculate the tempo using librosa's tempo function
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

        # Convert the tempo from frames per second to beats per minute
        tempo_bpm = tempo.item()

        return jsonify({'tempo_bpm': int(tempo_bpm)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
