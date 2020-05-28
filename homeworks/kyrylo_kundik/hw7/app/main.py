import os
import time

import torch
import xgboost
from flask import Flask, jsonify, request, abort
from flask_sqlalchemy import SQLAlchemy

from app.config import PROJECT_ROOT
from app.models.fnn.model import Model
from app.utilities import set_up_logging, load_env, preprocess_row

web_app = Flask(__name__)
web_app.config.from_object(os.environ['APP_SETTINGS'])
web_app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
db = SQLAlchemy(web_app)

tree_model = xgboost.XGBRegressor()
tree_model.load_model(str(PROJECT_ROOT / 'data' / 'xgb.model'))

model = Model(input_dim=537, hidden_dim=20, hidden_num=3)
model.load_state_dict(torch.load(str(PROJECT_ROOT / 'data' / 'nn.model')))


@web_app.route('/api/v1/statistics')
def statistics():
    from app.pg_db.queries import get_apartments_stats

    return jsonify(get_apartments_stats())


@web_app.route('/api/v1/records')
def records():
    limit = request.args.get('limit')
    offset = request.args.get('offset')

    from app.pg_db.queries import get_apartments
    return jsonify({'apartments': [x.serialize for x in get_apartments(limit, offset)]})


@web_app.route('/api/v1/predict', methods=['POST'])
def predict():
    content = request.get_json()
    model_type = content['model']
    data = preprocess_row(content['features'])
    if model_type == 'xgboost':
        start_time = time.time()
        prediction = float(tree_model.predict(data)[0])
        inference_time = time.time() - start_time
    elif model_type == 'fnn':
        with torch.no_grad():
            start_time = time.time()
            prediction = model(torch.tensor(data.values, dtype=torch.float32)).item()
            inference_time = time.time() - start_time
    else:
        abort(400)

    return jsonify({
        'predicted_price': prediction,
        'inference_time': inference_time
    })


if __name__ == '__main__':
    load_env()
    set_up_logging(os.getenv('LOG_FILE'), bool(os.getenv('VERBOSE')))

    web_app.run(host=web_app.config['HOST'], port=web_app.config['PORT'])
