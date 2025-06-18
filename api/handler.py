import os
import json
import pickle
import pandas as pd

from flask                     import Flask, request, Response
from allinsurance.AllInsurance import AllInsurance

# caminho da pasta da API
home_path = os.path.dirname( os.path.abspath( __file__ ) )

# carregando o modelo
model = pickle.load( open( os.path.join( home_path, 'model', 'model_lgbm.pkl' ), 'rb' ) )

# instância da classe AllInsurance: os scalers e encoders são carregados uma única vez
pipeline = AllInsurance()

# inicializa a API
app = Flask( __name__ )

@app.route( '/', methods=['GET'] )
def index():
    return Response( json.dumps( { 'status': 'online' } ), status=200, mimetype='application/json' )

@app.route( '/allinsurance/predict', methods=['POST'] )
def allinsurance_predict():
    test_json = request.get_json( silent=True )

    if test_json: # há dados
        if isinstance( test_json, dict ): # um único cliente
            test_raw = pd.DataFrame( test_json, index=[0] )
        elif isinstance( test_json, list ) and all( isinstance( x, dict ) for x in test_json ): # vários clientes
            test_raw = pd.DataFrame( test_json, columns=test_json[0].keys() )
        else:
            return Response( json.dumps( { 'erro': 'Envie um cliente ou uma lista de clientes no formato JSON' }, ensure_ascii=False ), status=400, mimetype='application/json' )

        # verifica se todas as colunas necessárias foram enviadas
        cols_missing = [col for col in pipeline.cols_required if col not in test_raw.columns]
        if cols_missing:
            return Response( json.dumps( { 'erro': f'Colunas ausentes: {cols_missing}' }, ensure_ascii=False ), status=400, mimetype='application/json' )

        try:
            # limpeza dos dados
            df1 = pipeline.data_cleaning( test_raw.copy() )

            # feature engineering
            df2 = pipeline.feature_engineering( df1 )

            # preparação dos dados
            df3 = pipeline.data_preparation( df2 )

        except ( ValueError, TypeError ) as e:
            return Response( json.dumps( { 'erro': f'Dados inválidos: {e}' }, ensure_ascii=False ), status=400, mimetype='application/json' )

        # predição
        df_response = pipeline.get_prediction( model, test_raw, df3 )

        return Response( df_response, status=200, mimetype='application/json' )

    else:
        return Response( '{}', status=200, mimetype='application/json' )

if __name__ == '__main__':
    port = int( os.environ.get( 'PORT', 5000 ) )
    app.run( host='0.0.0.0', port=port )
