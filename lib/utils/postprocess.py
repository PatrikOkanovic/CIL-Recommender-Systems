import pandas as pd

from lib import models
from lib.utils.config import config

SUBMISSION_CSV = 'path to .csv file'


def postprocess(model, predictions, postprocessing, filename):
    postprocessed_pred = model.postprocessing(predictions.Prediction, postprocessing)
    config.SUBMISSION_NAME = filename
    model.save_submission(predictions.Id, postprocessed_pred, postprocessing)
    return postprocessed_pred


if __name__ == '__main__':
    model = models.models['svd'].get_model(config, None)  # random model
    predictions = pd.read_csv(SUBMISSION_CSV)
    postprocess(model, predictions, postprocessing='round_quarters', filename=SUBMISSION_CSV)
    postprocess(model, predictions, postprocessing='round', filename=SUBMISSION_CSV)
