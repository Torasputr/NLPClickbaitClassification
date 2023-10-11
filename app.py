from flask import Flask, render_template, request
import pickle
import numpy as np
app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
@app.route('/')
def hello_world():
    return render_template('main.html')

@app.route('/predict')
def predict():
    print(request.form)
    int_features = [int(x) for x in request.form.values()]
    final = [np.array(int_features)]
    print(int_features)
    print(final)
    prediction = model.predict_proba(final)
    output = '{0:.{1}f}'.format(prediction[0][1], 2)
    if output == 0:
        return render_template('main.html', pred = "Clickbait content.\nThe probability is {}".format(output))
    else:
        return render_template('main.html', pred = "Non-Clickbait Content.\nThe probability is {}".format(output))
    
if __name__ == '__main__':
    app.run()
        