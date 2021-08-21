from ..interpretators.baseinterpretator import BaseInterpretator


class RiskInterpretator(BaseInterpretator):
    def __init__(self, model, objective='classification', algorithm='boosting'):
        BaseInterpretator.__init__(self, model, objective, algorithm)

    def explore_risk_difference(self, id1, id2):
        """
        Обнаруживает причины различий в оценках PD по лвум объектам
        :param id1: номер первого объекта в датасете
        :param id2: номер второго объекта в датасете
        :return:
        """
        pass

    def explore_risk_difference(self, obj1, obj2):
        """
        Обнаруживает причины различий в оценках PD по лвум объектам
        :param obj1: Первый объект, предикторы
        :param obj2: Второй объект, предикторы
        :return:
        """
        pass