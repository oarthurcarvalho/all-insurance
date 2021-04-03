from flask import Flask, request
import pickle
import pandas as pd

from data_prep import AllInsurance

# load model
model = pickle.load( open('models/GB_tuned.pkl', 'rb') )

# instanciate flask
app = Flask( __name__ )

@app.route( '/predict', methods=['POST'] )
def predict():
    test_json = request.get_json()

    # collect data
    if test_json:
        if isinstance( test_json, dict ): # unique value
            df_raw = pd.DataFrame( test_json, index=[0])
        else:
            df_raw = pd.DataFrame( test_json, columns=test_json[0].keys() )

    # instanciando data_prep
    pipeline = AllInsurance()

    # transformando a feature Annual_Premium
    df1 = pipeline.data_preparation( df_raw )

    # prediction
    pred = model.predict( df1 )

    df_raw['prediction'] = pred

    return df_raw.to_json( orient='records')

if __name__ == '__main__':
    # start flask
    app.run( host='0.0.0.0', port='5000')