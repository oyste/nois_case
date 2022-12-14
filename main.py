## REST -   REpresentational State Transfer
## API  -   Application Programming Interface   

from flask import Flask
from flask_restful import Api, Resource, reqparse
import pandas as pd
import pickle
#from RandomForestModel import construct_features
import numpy as np

rfr = pickle.load(open('rfr_price_predictor.pkl', "rb"))
xy_train = pd.read_pickle("train_xy.pkl")
target_encoder_dicts = pickle.load(open('target_encoder_dicts.pkl', "rb"))

## PIPELINE FUNCTIONS
def preprocess_and_predict_from_input_df(model, df, xy_train):
    df.datasource = df.datasource.astype('object')
    df.ModelID = df.ModelID.astype('object')
    df.auctioneerID = df.auctioneerID.astype('object')
    df.YearMade = df.YearMade.astype('object')
    df.saledate = df.saledate.transform(lambda x: (x.split(' ')[0].split('/')[-1]) if x else x)
    feats = ['ModelID', 'datasource', 'YearMade', 'ProductGroup', 'saledate', 'fiBaseModel', 'fiModelDesc', 'Enclosure', 'Hydraulics', 'auctioneerID']
    df = df.fillna('missing')
    filler = np.average(xy_train.SalePrice.values)
    pred_feature = []
    for dict, feat in zip(target_encoder_dicts, feats):
        pred_feature.append(df[feat].map(dict).fillna(filler).values.reshape(-1,1))
    return np.expm1(model.predict(np.hstack(pred_feature))[0])
app = Flask(__name__)
api = Api(app)

input_get_args = reqparse.RequestParser()
#['ModelID', 'datasource', 'YearMade', 'ProductGroup', 'saledate', 'fiBaseModel', 'fiModelDesc', 'Enclosure', 'Hydraulics', 'auctioneerID'
input_get_args.add_argument("ModelID", type=int, help="was not filled in")
input_get_args.add_argument("datasource", type=int, help="was not filled in")
input_get_args.add_argument("YearMade", type=int, help="was not filled in")
input_get_args.add_argument("ProductGroup", type=str, help="was not filled in")
input_get_args.add_argument("saledate", type=str, help="was not filled in")
input_get_args.add_argument("fiBaseModel", type=str, help="was not filled in")
input_get_args.add_argument("fiModelDesc", type=str, help="was not filled in")
input_get_args.add_argument("Enclosure", type=str, help="was not filled in")
input_get_args.add_argument("Hydraulics", type=str, help="was not filled in")
input_get_args.add_argument("auctioneerID", type=float, help="was not filled in")

class Prediction(Resource):
    def get(self):
        args = input_get_args.parse_args()
        df = pd.DataFrame(args, index=[0])
        y_pred = preprocess_and_predict_from_input_df(rfr, df, xy_train)
        #df = pd.DataFrame.from_dict(dict)
        #rfr.predict(filtered input)
        return {'Predicted Price:' : (f'{y_pred} USD', args)}
api.add_resource(Prediction, "/") ## accessible at "BASE/" as root.

if __name__ == "__main__":
    app.run(debug=True)