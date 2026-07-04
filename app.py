from flask import Flask, render_template

app = Flask(__name__)
print(__name__)
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend/<int:movie_id>")
def recommend(movie_id):
    recommendations = [
        "Inception",
        "No Time to Die",
        "Mission Impossible: Fallout"
    ]

    return {
        "movie_id": movie_id,
        "recommendations": recommendations
    }

if __name__ == "__main__":
    app.run(debug=True)