import numpy as np
import pickle

class AllInsurance( object ):
    def __init__( self ):
        self.annual_premium = pickle.load( open( 'deploy/annual_premium_scaler.pkl', 'rb' ))

    def data_preparation (self, df):
        # reescalando a feature annual_premium
        df['Annual_Premium'] = np.log1p( df[['Annual_Premium']].values )
        df['Annual_Premium'] = self.annual_premium.transform( df['Annual_Premium'] )

        return df