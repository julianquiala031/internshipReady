from flask import Flask, jsonify

app = Flask(__name__)


DATA = [
    {"id": 1, "campus": "MMC", "latitude": 25.760578101965788, "longitude": -80.3693412116132},
    {"id": 2, "campus": "BBC", "latitude": 25.909326038381668, "longitude": -80.13842127148544},
]
next_id = 3


@app.route("/")
def index():
    return """
    <h1>Welcome to My Flask API</h1>
    <p>Try these endpoints:</p>
    <ul>
        <li><a href="/api/health">/api/health</a></li>
        <li><a href="/api/items">/api/items</a></li>
        <li><a href="/api/items/1">/api/items/1</a></li>
    </ul>
    """

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/api/items", methods=["GET"])
def list_items():
    return jsonify(DATA), 200


@app.route("/api/items/<int:item_id>", methods=["GET"])
def get_item(item_id):
    for item in DATA:
        if item["id"] == item_id:
            return jsonify(item), 200
    return jsonify({"error": "not found"}), 404



if __name__ == "__main__":
    app.run(debug=True, port=5055)

DATA = [


]
next_id = 3