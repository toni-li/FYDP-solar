from flask import Flask, render_template, request
import solarModel

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route('/run_model', methods=['POST'])
def run_model():
    postalCode = request.form['postal-code']
    roofSize = request.form['roof-size']
    usage = request.form['usage']
    month = request.form['month']
    heating = request.form['heating']
    budget = request.form['budget']

    solarModel.solve(postalCode, roofSize, usage, month, heating, budget)

    return render_template('success.html')


if __name__ == "__main__":
    app.run(port=5000, debug=True)