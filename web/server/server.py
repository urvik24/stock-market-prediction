from flask import Flask,request,jsonify
import util

app = Flask(__name__)

@app.route("/")
def hello():
    return "The Server is on"

@app.route("/aa" , methods = ['POST'])
def aa():
    name2=request.form['name']
    response=jsonify(
        {
            'estimated_price' : util.get_price(name2)
            
        }
    )
    response.headers.add('Access-Control-Allow-Origin','*')
    return response

if __name__ == "__main__":
    app.run()
