from _internals.prepare_data import prepare_data
from _internals.print_metrics import print_metrics
from _internals.save_model import save_model
from sklearn.linear_model import ElasticNet

from homework.src._internals.calculate_metrics import calculate_metrics


def run_experiment(estimator):
    x_train, x_test, y_train, y_test = prepare_data()

    # entrenar el modelo
    estimator.fit(x_train, y_train)

    print()
    print(estimator, ":", sep="")

    # Metricas de error durante entrenamiento.
    mse, mae, r2 = calculate_metrics(x_train, y_train, estimator)
    print_metrics(mse, mae, r2, "Métricas de entrenamiento:")

    # Metricas de error durante test.
    mse, mae, r2 = calculate_metrics(x_test, y_test, estimator)
    print_metrics(mse, mae, r2, "Métricas en testing:")
