class BaseInterpretator:
    """
    Базовый класс интерпретатора
    """

    def __init__(self, model, data):
        """
        Создаёт объект интерпретатора
        :param model: Модель для интерпретации
        :param data: Данные, на которых построенна модель. Используются для отдельных видоп интепретации
        """
        self.__model = model
        self.__model_data = data

    def interpret_1(self):
        """
        Плейсхолдер для метода интепретации
        :return: Возвращает результат интепретации
        """
        pass
