import shap
import numpy as np
import matplotlib.pyplot as plt
from skater.model import InMemoryModel
from skater.core.explanations import Interpretation


class BaseInterpretator:
    """
    Базовый класс интерпретатора
    """

    def __init__(self, model, objective = 'classification', algorithm = 'boosting'):
        """
        Создаёт объект интерпретатора
        :type algorithm: Алгоритм модели. Допустимые значения: boosting, random_forest
        :type objective: Тип целевой переменной в модели. Допустимые значения: classification, regression
        :param model: Модель для интерпретации
        """
        if objective not in ['classification', 'regression']:
            raise BaseException('Unknown Objective')
        if algorithm not in ['boosting', 'random_forest']:
            raise BaseException('Unknown algorithm')

        self.__model = model
        self.__shap_explainer = None
        self.__pdp_explainer = None
        self.__annotated_model = None

        self.__objective = objective
        self.__algo = algorithm
    
    def fit_shap(self):
        self.__shap_explainer = shap.TreeExplainer(self.__model)
        return

    def shap(self, data, type = 'summary_plot', num_features = None):
        """
        Плейсхолдер для метода интепретации
        :param type: Тип графика
        :param data: Данные, на которых построенна модель. Используются для отдельных видоп интепретации
        :return: Возвращает результат интепретации
        """
        # Проверка параметров
        if self.__shap_explainer is None:
            raise BaseException("SHAP explainer is not fitted. Run fit_shap at first")

        if self.__algo == "random_forest":
            shap_values = self.__shap_explainer.shap_values(data)[1]
            expected_value = self.__shap_explainer.expected_value[1]
        else:
            shap_values = self.__shap_explainer.shap_values(data)
            expected_value = self.__shap_explainer.expected_value

        if type == 'summary_plot':
            return shap.summary_plot(shap_values, data, max_display = num_features)
        elif type == 'summary_bar_plot':
            return shap.summary_plot(shap_values, data, plot_type='bar', max_display = num_features)
        elif type == 'individual_plot':
            shap.initjs()
            return shap.force_plot(expected_value, shap_values, data)
        else:
            raise BaseException('Unknown SHAP plot type')
        
    def fit_pdp(self, data):
        """
        :param data: Набор данных
        """
        self.__pdp_explainer = Interpretation(data, feature_names=data.columns)

        if self.__objective == 'classification':
            self.__annotated_model = InMemoryModel(self.__model.predict_proba, examples=data)
        elif self.__objective == 'regression':
            self.__annotated_model = InMemoryModel(self.__model.predict, examples=data)
        
    def pdp(self, features, grid_resolution = 30, n_samples=10000):
        """
        Возврщает график PDP
        :param features: tuple из 1 или 2 фичей
        :param grid_resolution: Количество ячеек по каждой из осей
        :param n_samples: The number of samples to use from the original dataset
        :return: Возвращает график PDP
        """

        if self.__pdp_explainer is None or self.__annotated_model is None:
            raise BaseException("PDP explainer is not fitted. Run fit_pdp at first")

        pdp_features = [features]

        return self.__pdp_explainer.partial_dependence.plot_partial_dependence(pdp_features,
                                                       self.__annotated_model,
                                                       grid_resolution=grid_resolution,
                                                       n_samples=n_samples,
                                                       n_jobs=-1)
        
    def analyze_voters(self, obj, figsize=[10, 7]):
        """
        Проводит анализ голосвания деревьев в лесу
        :param obj: Анализируемое наблюдение
        :param figsize: Размер выходного графика
        :return: Результаты голосования деревьев
        """
        if self.__algo != 'random_forest':
            raise BaseException("Can be used only for Random Forest")

        def get_voters(obj):
            predicted_pobas = list()

            for est in self.__model.estimators_:
                probas = est.predict_proba(obj)
                predicted_pobas.append([p[1] for p in probas][0])
            return predicted_pobas


        predicted_pobas = get_voters(obj)
        mean_pred = np.mean(predicted_pobas)
        std_pred = np.std(predicted_pobas)

        fig = plt.figure(figsize=figsize)
        plt.hlines(mean_pred, xmin=0, xmax=len(predicted_pobas), label='mean prediction')
        bar_char = plt.bar(x=list(range(len(predicted_pobas))), height=predicted_pobas)
        cum_vote = plt.plot(sorted(predicted_pobas), c='red', label='cum votes')
        plt.legend()

        return predicted_pobas, bar_char, cum_vote