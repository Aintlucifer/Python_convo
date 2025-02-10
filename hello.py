from flask import Flask, request, jsonify
import openai
from textblob import TextBlob
import datetime
from dotenv import load_dotenv
import os

app = Flask(__name__)
load_dotenv()

openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

user_data = {}

@app.route('/chat', methods=['POST'])
def chat():
    """Handles user messages and AI responses, maintaining conversation context."""
    user_message = request.json.get('message')
    user_id = request.json.get('user_id')

    if not user_id or not user_message:
        return jsonify({"error": "User ID and message are required"}), 400

    conversation_history = user_data.get(user_id, [])

    conversation_history.append({"role": "user", "content": user_message})

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful and engaging AI chatbot."}
            ] + conversation_history, 
            max_tokens=200
        )

        ai_response = response.choices[0].message.content

        conversation_history.append({"role": "assistant", "content": ai_response})

        sentiment = TextBlob(user_message).sentiment
        emotion_score = sentiment.polarity

        timestamp = datetime.datetime.now().isoformat()
        user_data[user_id] = conversation_history  

        return jsonify({
            "ai_response": ai_response,
            "emotion_score": emotion_score,
            "timestamp": timestamp
        })

    except openai.APIError as e:
        return jsonify({"error": f"OpenAI API error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


@app.route('/talk', methods=['POST'])
def talk():
    """Logs user message with sentiment analysis but does not call OpenAI API."""
    user_message = request.json.get('message')
    user_id = request.json.get('user_id')

    if not user_id or not user_message:
        return jsonify({"error": "User ID and message are required"}), 400

    sentiment = TextBlob(user_message).sentiment
    emotion_score = sentiment.polarity

    timestamp = datetime.datetime.now().isoformat()
    if user_id not in user_data:
        user_data[user_id] = []
    user_data[user_id].append({
        "timestamp": timestamp,
        "user_message": user_message,
        "emotion_score": emotion_score
    })

    return jsonify({
        "message": "Message recorded successfully",
        "emotion_score": emotion_score,
        "timestamp": timestamp
    })


@app.route('/mood', methods=['GET'])
def get_mood():
    """Calculates user mood based on last 10 minutes of interactions."""
    user_id = request.args.get('user_id')
    if not user_id or user_id not in user_data:
        return jsonify({"error": "User ID not found"}), 404

    now = datetime.datetime.now()
    ten_minutes_ago = now - datetime.timedelta(minutes=10)

    recent_interactions = [
        interaction for interaction in user_data[user_id]
        if isinstance(interaction, dict) and 'timestamp' in interaction and
        datetime.datetime.fromisoformat(interaction['timestamp']) >= ten_minutes_ago
    ]

    if not recent_interactions:
        return jsonify({"error": "No recent interactions found"}), 404

    total_score = sum(interaction.get('emotion_score', 0) for interaction in recent_interactions)
    average_score = total_score / len(recent_interactions)

    if average_score > 0.2:
        mood = "Happy"
    elif average_score < -0.2:
        mood = "Sad"
    else:
        mood = "Neutral"

    return jsonify({
        "mood": mood,
        "average_emotion_score": average_score,
        "recent_interactions": recent_interactions
    })

if __name__ == '__main__':
    app.run(debug=True)  
