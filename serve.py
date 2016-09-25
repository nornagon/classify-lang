from flask import Flask, request
import json
maxlen = 32
app = Flask(__name__)

from predict import predict


@app.route("/")
def hello():
    return app.send_static_file('index.html')


@app.route("/classify", methods=['POST'])
def classify():
    sentence = request.form['sentence']
    return json.dumps(predict(sentence))

if __name__ == "__main__":
    app.run()
