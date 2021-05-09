import shap 
from skater.model import InMemoryModel
from skater.core.explanations import Interpretation

class BaseInterpretator:
    """
    Базовый класс интерпретатора
    """

    def __init__(self, model, objective = 'classification'):
        """
        Создаёт объект интерпретатора
        :type objective: Тип целевой переменной в модели. Допустимые значения: classification, regression
        :param model: Модель для интерпретации
        """
        if objective not in ['classification', 'regression']:
            raise BaseException('Unknown Objective')

        self.__model = model
        self.__shap_explainer = None
        self.__pdp_explainer = None
        self.__annotated_model = None

        self.__objective = objective
    
    def fit_shap(self):
        self.__shap_explainer = shap.TreeExplainer(self.__model)
        return

    def shap(self, data, type = 'summary_plot', num_features = None):
        """
        Плейсхолдер для метода интепретации
        :param data: Данные, на которых построенна модель. Используются для отдельных видоп интепретации
        :return: Возвращает результат интепретации
        """
        if self.__shap_explainer is None:
            raise BaseException("SHAP explainer is not fitted. Run fit_shap at first")
        shap_values = self.__shap_explainer.shap_values(data)

        if type == 'summary_plot':
            return shap.summary_plot(shap_values, data, max_display = num_features)
        elif type == 'summary_bar_plot':
            return shap.summary_plot(shap_values, data, plot_type='bar', max_display = num_features)
        elif type == 'individual_plot':
            shap.initjs()
            return shap.force_plot(self.__shap_explainer.expected_value, shap_values, data)
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
        :param n_samples: ?количество сэмплов?
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
        
