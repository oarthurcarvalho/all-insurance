import os
import sys
import json
import requests
import pandas as pd

base_path = os.path.dirname( os.path.abspath( __file__ ) )

# URL da API: local por padrão ou a URL do Heroku passada como argumento
url = sys.argv[1] if len( sys.argv ) > 1 else 'http://127.0.0.1:5000'

# carregando os novos clientes
df_novos = pd.read_csv( os.path.join( base_path, '..', 'google-sheets', 'novos_clientes_amostra.csv' ) )

# seleciona os clientes para a predição
df_test = df_novos.sample( 20, random_state=0 )

# converte o dataframe para json
data = json.dumps( df_test.to_dict( orient='records' ) )

# chamada da API
header = { 'Content-type': 'application/json' }
r = requests.post( url=url + '/allinsurance/predict', data=data, headers=header, timeout=120 )
print( f'Status Code {r.status_code}' )

if r.status_code == 200:
    d1 = pd.DataFrame( r.json() ).sort_values( 'score', ascending=False ).reset_index( drop=True )

    print( '\nRanking dos clientes por propensão de compra do seguro automotivo:\n' )
    for i in range( len( d1 ) ):
        print( f"{i + 1:>2}º - Cliente {d1.loc[i, 'id_cliente']}: score {d1.loc[i, 'score']:.1%}" )

else:
    print( r.text )
