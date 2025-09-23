from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def index():
    return "<h1>Welcome to Internship Ready</h1>"

@app.route('/aboutUs/<string:name>')
def aboutUs(name):
    return f"<br> <h3> My name is {name} and I am enrolled in CIS 3950.</h3>"

if __name__ == '__main__':
    app.run(debug=True, port=5050)