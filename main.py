import pandas as pd

import lib.models as models
from lib.utils import utils
from lib.utils.config import config
from lib.utils.loader import extract_users_items_predictions, read_data
from lib.utils.postprocess import postprocess
from lib.utils.utils import get_score


def main():
    logger = utils.init(seed=config.RANDOM_STATE)
    logger.info(f'Using {config.MODEL} model for prediction')

    if config.TYPE == 'VAL':
        logger.info('Training on {:.0f}% of the data'.format(config.TRAIN_SIZE * 100))
        train_pd, val_pd, test_pd = read_data()
        train_users, train_movies, train_predictions = extract_users_items_predictions(train_pd)
        val_users, val_movies, val_predictions = extract_users_items_predictions(val_pd)
        test_users, test_movies, _ = extract_users_items_predictions(test_pd)

        X_train = pd.DataFrame({'user_id': train_users, 'movie_id': train_movies})
        y_train = pd.DataFrame({'rating': train_predictions})
        X_val = pd.DataFrame({'user_id': val_users, 'movie_id': val_movies})
        X_test = pd.DataFrame({'user_id': test_users, 'movie_id': test_movies})

        model = models.models[config.MODEL].get_model(config, logger)
        logger.info("Fitting the model")
        model.fit(X_train, y_train,
                  val_data=X_val, val_predictions=val_predictions)  # iterative val score
        logger.info("Testing the model")
        predictions = model.predict(X_val)
        logger.info('RMSE using {} is {:.4f}'.format(
            config.MODEL, get_score(predictions, target_values=val_predictions)))  # Note score is not preprocessed
    else:
        logger.info('Training on 100% of the data')
        train_pd, test_pd = read_data()
        train_users, train_movies, train_predictions = extract_users_items_predictions(train_pd)
        test_users, test_movies, _ = extract_users_items_predictions(test_pd)

        X_train = pd.DataFrame({'user_id': train_users, 'movie_id': train_movies})
        y_train = pd.DataFrame({'rating': train_predictions})
        X_test = pd.DataFrame({'user_id': test_users, 'movie_id': test_movies})

        model = models.models[config.MODEL].get_model(config, logger)
        logger.info("Fitting the model")
        model.fit(X_train, y_train,
                  test_data=X_train, test_every=config.TEST_EVERY)  # iterative test score

    logger.info("Creating submission file")
    predictions = model.create_submission(X_test, postprocessing='clipping')
    postprocess(model, pd.DataFrame({'Id': utils.get_index(test_movies, test_users), 'Prediction': predictions}),
                postprocessing='round_quarters', filename=config.SUBMISSION_NAME)
    postprocess(model, pd.DataFrame({'Id': utils.get_index(test_movies, test_users), 'Prediction': predictions}),
                postprocessing='round', filename=config.SUBMISSION_NAME)


if __name__ == '__main__':
    main()
