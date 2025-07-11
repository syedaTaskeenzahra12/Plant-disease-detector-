from flask import Flask, request, render_template
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    return "Plant Disease Detection Home"

if __name__ == "__main__":
    app.run(debug=True)
