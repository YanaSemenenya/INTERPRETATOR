import shap 

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
