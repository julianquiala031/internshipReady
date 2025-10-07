from flask import Flask

app = Flask(__name__)

@app.route("/") 

def index():
    return "<h1>Welcome to Internship Ready Fall 2025" \
            "Fall 2025</h1>"

@app.route("/aboutUs/<string:name>")
def aboutUs(name):
    return (f"<br><h3>Hi, my name is {name} and I am "
            f"a student in CIS3590.</h3>")

if __name__ == '__main__':
    app.run(debug=True,port=5050)





