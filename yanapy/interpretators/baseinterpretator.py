import shap 
from skater.model import InMemoryModel
from skater.core.explanations import Interpretation

class BaseInterpretator:
    """
    Базовый класс интерпретатора
    """

    def __init__(self, model):
        """
        Создаёт объект интерпретатора
        :param model: Модель для интерпретации
        """
        self.__model = model
        self.__shap_explainer = None
        self.__pdp_explainer = None
        self.__annotated_model = None
    
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

        if type == 'summary_plot':
            shap_values = self.__shap_explainer.shap_values(data)
            return shap.summary_plot(shap_values, data, max_display = num_features)
        
        if type == 'summary_bar_plot':
            shap_values = self.__shap_explainer.shap_values(data)
            return shap.summary_plot(shap_values, data, plot_type='bar', max_display = num_features)
        
        if type == 'individual_plot':
            shap.initjs()
            shap_value_sample = self.__shap_explainer.shap_values(data)
            return shap.force_plot(self.__shap_explainer.expected_value, shap_value_sample, data)
        
    def fit_pdp(self, data, model_type = 'classification'):
        """
        
        """

        if model_type == 'classification':
            self.__pdp_explainer = Interpretation(data, feature_names = data.columns)
            self.__annotated_model = InMemoryModel(self.__model.predict_proba, examples=data)
        elif model_type == 'regression':
            self.__pdp_explainer = Interpretation(data, feature_names = data.columns)
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

        return self.interpreter.partial_dependence.plot_partial_dependence(pdp_features, 
                                                       self.__annotated_model,
                                                       grid_resolution=grid_resolution,
                                                       n_samples=n_samples,
                                                       n_jobs=-1)
        
