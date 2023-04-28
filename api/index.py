from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello, World!'

@app.route('/about')
def about():
    return 'About'

@app.route('/my_endpoint', methods=['POST'])
def my_endpoint():
    param = request.json['my_param']
    # do something with the parameter
    result = {'message': 'success'}
    return jsonify(result)
