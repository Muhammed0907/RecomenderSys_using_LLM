from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import faiss
import numpy as np
import requests
import pandas as pd
import threading

dfAitool = pd.read_csv("./ai_tools_50.csv")


app = Flask(__name__, static_folder='frontend')
CORS(app)  

trainedAitoolEmbedModel = faiss.read_index("ai_tool_embed_model")


USER_INFO = []

OllamaURL = "http://localhost:11434/api/embeddings"

USER_RECOMEND_TOOL = {
    "Tools":dfAitool["Name"].tolist()[:10]
}

def recommend(userProfile):
    resp = requests.post(
        OllamaURL,
        json={
            "model": "nomic-embed-text",
            "prompt": userProfile,
        }
    )
    embed = np.array([resp.json()["embedding"]], dtype="float32")
    
    _, I = trainedAitoolEmbedModel.search(embed, 10)
    
    bestMatches = dfAitool.iloc[I.flatten()]
    userRecomenedTools = pd.DataFrame({
    "tools": bestMatches['Name'],
    })
    
    global USER_RECOMEND_TOOL 
    USER_RECOMEND_TOOL["Tools"] = bestMatches['Name'].tolist()

    userRecomenedTools.to_csv("userRecomenedTools.csv", index=False)
    return bestMatches

@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/recommend_tools', methods=['GET'])
def get_recommendations():
    return jsonify(USER_RECOMEND_TOOL["Tools"])

def recommend_in_background(user_info):
    recommend(", ".join(user_info)) 
    global USER_INFO
    USER_INFO = USER_INFO[len(USER_INFO)-2:]


@app.route("/search",methods=["POST"])
def get_searchData():
    global USER_INFO

    search = request.json.get("search")

    USER_INFO.append(search)

    threading.Thread(target=recommend_in_background, args=(USER_INFO[::-1],)).start()
    
    return jsonify({"data":f"You searched: {search}" })

@app.route("/add_selected_tool_in_user_history",methods=["POST"])
def getSelectedToolAddUserHhis():
    selectedTool = request.json.get("selected_tool")

    global USER_INFO

    USER_INFO.append(selectedTool)
    threading.Thread(target=recommend_in_background, args=(USER_INFO[::-1],)).start()

    return jsonify({"data":f"You selected: {selectedTool}" })



@app.route('/update_profile', methods=['POST'])
def update_profile():
    data = request.json
    user_search_history = data.get('search_history')
    used_tools = data.get('used_tools')
    
    userProfile = f"search_history: {', '.join(user_search_history)}, used_tools: {', '.join(used_tools)}"
    
    return jsonify({"userProfile": userProfile})

if __name__ == "__main__":
    app.run(debug=True)
