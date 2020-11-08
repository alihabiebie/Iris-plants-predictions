from flask import Flask, render_template, jsonify, request
import pickle

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/iris')
def iris():
    return render_template('iris.html')

@app.route('/hasil', methods=['POST'])
def klasifikasi():
    if request.method == 'POST':
        input = request.form
        sl = float(input['sl'])
        sw = float(input['sw'])
        pl = float(input['pl'])
        pw = float(input['pw'])
        pred = Iris.predict([[sl, sw, pl, pw]])[0]

        return render_template('result.html', data=input, prediksi=pred)

if __name__ == "__main__":
    ### Load Model
    with open('modelIris', 'rb') as ML:
        Iris = pickle.load(ML)
    app.run(debug=True)
