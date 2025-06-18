import os
import sys
import json
import pytest
import pandas as pd

base_path = os.path.dirname( os.path.abspath( __file__ ) )

# a API é importada diretamente, sem precisar subir o servidor
sys.path.insert( 0, os.path.join( base_path, '..', 'api' ) )
from handler import app

ENDPOINT = '/allinsurance/predict'

# novos clientes da amostra do Google Sheets
df_novos = pd.read_csv( os.path.join( base_path, '..', 'google-sheets', 'novos_clientes_amostra.csv' ) )


def registros( n ):
    # converte os n primeiros clientes para o formato JSON enviado à API
    return json.loads( df_novos.head( n ).to_json( orient='records' ) )


@pytest.fixture
def client():
    return app.test_client()


def test_api_online( client ):
    r = client.get( '/' )

    assert r.status_code == 200
    assert r.get_json() == { 'status': 'online' }


def test_predicao_varios_clientes( client ):
    clientes = registros( 50 )
    r = client.post( ENDPOINT, json=clientes )
    d1 = pd.DataFrame( r.get_json() )

    # um score para cada cliente, na mesma ordem e com os dados originais
    assert r.status_code == 200
    assert d1['id_cliente'].tolist() == [c['id_cliente'] for c in clientes]
    assert d1['vehicle_age'].tolist() == [c['vehicle_age'] for c in clientes]
    assert d1['score'].between( 0, 1 ).all()


def test_predicao_um_cliente( client ):
    r = client.post( ENDPOINT, json=registros( 1 )[0] )

    assert r.status_code == 200
    assert len( r.get_json() ) == 1


def test_categorias_padronizadas( client ):
    cliente = registros( 1 )[0]
    variacao = { **cliente,
                 'gender': cliente['gender'].upper(),
                 'vehicle_damage': f" {cliente['vehicle_damage'].lower()} " }

    score = client.post( ENDPOINT, json=[cliente] ).get_json()[0]['score']
    score_variacao = client.post( ENDPOINT, json=[variacao] ).get_json()[0]['score']

    # espaços extras e maiúsculas/minúsculas não alteram o score
    assert score_variacao == pytest.approx( score )


def test_coluna_ausente( client ):
    cliente = registros( 1 )[0]
    del cliente['vehicle_damage']
    r = client.post( ENDPOINT, json=[cliente] )

    assert r.status_code == 400
    assert 'vehicle_damage' in r.get_json()['erro']


def test_categoria_desconhecida( client ):
    cliente = { **registros( 1 )[0], 'vehicle_damage': 'talvez' }
    r = client.post( ENDPOINT, json=[cliente] )

    assert r.status_code == 400
    assert 'vehicle_damage' in r.get_json()['erro']


def test_valor_numerico_invalido( client ):
    cliente = { **registros( 1 )[0], 'age': 'quarenta' }
    r = client.post( ENDPOINT, json=[cliente] )

    assert r.status_code == 400


def test_formato_invalido( client ):
    r = client.post( ENDPOINT, json=[1, 2, 3] )

    assert r.status_code == 400
