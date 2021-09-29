import settings as setts
from tools.models import MLModel
import tools.data_handling as dtools


input_data = dtools.read_dataset(print_data_summary=False)
dtools.clean_data(input_data)

if setts.do_input_data_analysis:
    dtools.input_data_analysis(input_data)

ml_model = MLModel(setts.ml_model, input_data, setts.variables_for_training, setts.target_variable)
ml_model.train_model()

print("Done!")



