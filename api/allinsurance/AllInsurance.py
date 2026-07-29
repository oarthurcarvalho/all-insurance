import os
import pickle
import pandas as pd


class AllInsurance( object ):
    def __init__( self ):
        # caminho da pasta da API, a partir da localização deste arquivo
        self.home_path = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )

        # scalers e encoders treinados no notebook
        self.annual_premium_scaler        = pickle.load( open( os.path.join( self.home_path, 'parameter', 'annual_premium_scaler.pkl' ), 'rb' ) )
        self.age_scaler                   = pickle.load( open( os.path.join( self.home_path, 'parameter', 'age_scaler.pkl' ), 'rb' ) )
        self.region_code_encoder          = pickle.load( open( os.path.join( self.home_path, 'parameter', 'region_code_encoder.pkl' ), 'rb' ) )
        self.policy_sales_channel_encoder = pickle.load( open( os.path.join( self.home_path, 'parameter', 'policy_sales_channel_encoder.pkl' ), 'rb' ) )

        # colunas que a API precisa receber
        self.cols_required = ['gender', 'age', 'region_code', 'driving_license', 'previously_insured',
                              'annual_premium', 'policy_sales_channel', 'vehicle_age', 'vehicle_damage']

        # features do modelo, na mesma ordem utilizada no treinamento
        self.cols_selected = ['age', 'gender', 'driving_license', 'region_code', 'previously_insured',
                              'annual_premium', 'policy_sales_channel', 'vehicle_age', 'vehicle_damage']

    def data_cleaning( self, df1 ):

        # ALTERAÇÃO DE TIPOS
        cols_int = ['age', 'region_code', 'driving_license', 'previously_insured', 'policy_sales_channel']
        for col in cols_int:
            df1[col] = pd.to_numeric( df1[col] ).astype( 'int64' )

        df1['annual_premium'] = pd.to_numeric( df1['annual_premium'] ).astype( 'float64' )
        if df1['annual_premium'].isna().any():
            raise ValueError( 'annual_premium possui valores vazios' )

        # PADRONIZAÇÃO DAS CATEGORIAS: ignora espaços extras e maiúsculas/minúsculas
        for col in ['gender', 'vehicle_age', 'vehicle_damage']:
            df1[col] = df1[col].astype( str ).str.strip().str.lower()

        return df1

    def feature_engineering( self, df2 ):

        # vehicle_age
        vehicle_age_map = { '< 1 year': 'menos_de_1_ano', '1-2 year': 'entre_1_e_2_anos', '> 2 years': 'mais_de_2_anos' }
        df2['vehicle_age'] = df2['vehicle_age'].map( vehicle_age_map )

        # vehicle_damage
        vehicle_damage_map = { 'yes': 1, 'no': 0 }
        df2['vehicle_damage'] = df2['vehicle_damage'].map( vehicle_damage_map )

        # gender
        gender_map = { 'male': 'masculino', 'female': 'feminino' }
        df2['gender'] = df2['gender'].map( gender_map )

        # VALIDAÇÃO: uma categoria desconhecida vira um valor vazio e geraria um score errado
        cols_invalid = [col for col in ['gender', 'vehicle_age', 'vehicle_damage'] if df2[col].isna().any()]
        if cols_invalid:
            raise ValueError( f'categorias desconhecidas nas colunas {cols_invalid}' )

        return df2

    def data_preparation( self, df5 ):

        # RESCALING

        # annual_premium - Robust Scaler
        df5['annual_premium'] = self.annual_premium_scaler.transform( df5[['annual_premium']].values )

        # age - Min-Max Scaler
        df5['age'] = self.age_scaler.transform( df5[['age']].values )

        # ENCODING

        # gender - Binary Encoding
        df5['gender'] = df5['gender'].map( { 'feminino': 0, 'masculino': 1 } )

        # vehicle_age - Ordinal Encoding
        df5['vehicle_age'] = df5['vehicle_age'].map( { 'menos_de_1_ano': 0, 'entre_1_e_2_anos': 1, 'mais_de_2_anos': 2 } )

        # region_code - Target Encoding
        df5['region_code'] = self.region_code_encoder.transform( df5[['region_code']] )

        # policy_sales_channel - Target Encoding
        df5['policy_sales_channel'] = self.policy_sales_channel_encoder.transform( df5[['policy_sales_channel']] )

        return df5[self.cols_selected]

    def get_prediction( self, model, original_data, test_data ):
        # predição: probabilidade de o cliente ter interesse no seguro automotivo
        pred = model.predict_proba( test_data )[:, 1]

        # junta o score aos dados original
        original_data['score'] = pred

        return original_data.to_json( orient='records' )
