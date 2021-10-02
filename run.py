import settings as setts
from tools.models import MLModel

ml_model = MLModel(setts.ml_model, settings=setts)

json_result = ml_model.generate_prediction_from_values(input_registered=[32, 60],
                                                       input_month=[1, 5],
                                                       input_hour=[1, 5],
                                                       input_weekday=[6, 3])
