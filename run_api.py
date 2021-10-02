from flask import Flask, request, jsonify
from tools.models import MLModel
import settings as setts

app = Flask(__name__)


@app.route('/prediction_api', methods=["GET"])
def prediction_api():
    try:
        # Getting the parameters from API call
        registered_value = float(request.args.get('registered'))
        mnth_value = float(request.args.get('mnth'))
        hr_value = float(request.args.get('hr'))
        weekday_value = float(request.args.get('weekday'))

        ml_model = MLModel(setts.ml_model, settings=setts)

        prediction_from_api = ml_model.generate_prediction_from_values(input_registered=registered_value,
                                                                       input_month=mnth_value,
                                                                       input_hour=hr_value,
                                                                       input_weekday=weekday_value)

        return prediction_from_api

    except Exception as e:
        return 'Something is not right!:' + str(e)


if __name__ == "__main__":
    # Hosting the API in localhost
    app.run(host='127.0.0.1', port=8080, threaded=True, debug=True, use_reloader=False)

    # (example) http://127.0.0.1:8080/prediction_api?registered=32&mnth=1&hr=1&weekday=6