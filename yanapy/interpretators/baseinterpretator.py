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
    
    def fit_shap(self):
        self.__explainer = shap.TreeExplainer(self.__model)
        return

    def shap(self, data, type = 'summary_plot', num_features = None):
        """
        Плейсхолдер для метода интепретации
        :param data: Данные, на которых построенна модель. Используются для отдельных видоп интепретации
        :return: Возвращает результат интепретации
        """
        if type == 'summary_plot':
            shap_values = self.__explainer.shap_values(data)
            return shap.summary_plot(shap_values, data, max_display = num_features)
        
        if type == 'summary_bar_plot':
            shap_values = self.__explainer.shap_values(data)
            return shap.summary_plot(shap_values, data, plot_type='bar', max_display = num_features)
        
        if type == 'individual_plot':
            shap.initjs()
            shap_value_sample = self.__explainer.shap_values(data)
            return shap.force_plot(self.__explainer.expected_value, shap_value_sample, data)
        
    def fit_pdp(self, data, model_type = 'classification'):
        """
        
        """
        if model_type == 'classification':
            self.interpreter = Interpretation(data, feature_names = data.columns)
            self.annotated_model = InMemoryModel(self.__model.predict_proba, examples=data)
        elif model_type == 'regression':
            self.interpreter = Interpretation(data, feature_names = data.columns)
            self.annotated_model = InMemoryModel(self.__model.predict, examples=data)
        
    def pdp(self, features, grid_resolution = 30, n_samples=10000):
        """
        Возврщает график PDP
        :param features: tuple из 1 или 2 фичей
        :param grid_resolution: Количество ячеек по каждой из осей
        :param n_samples: ?количество сэмплов?
        :return: Возвращает график PDP
        """
        pdp_features = [features]
        return self.interpreter.partial_dependence.plot_partial_dependence(pdp_features, 
                                                       self.annotated_model,
                                                       grid_resolution=grid_resolution,
                                                       n_samples=n_samples,
                                                       n_jobs=-1)
        
