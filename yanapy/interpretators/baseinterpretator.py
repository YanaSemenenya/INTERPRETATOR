import shap
import numpy as np
import matplotlib.pyplot as plt
from skater.model import InMemoryModel
from skater.core.explanations import Interpretation
from skater.core.local_interpretation.lime.lime_tabular import LimeTabularExplainer
from skater.util.dataops import show_in_notebook
from skater.util import exceptions
import types
import seaborn as sns
import pandas as pd


class BaseInterpretator:
    """
    Базовый класс интерпретатора
    """

    def __init__(self, model, objective='classification', algorithm='boosting'):
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
        self.__skater_explainer = None
        self.__annotated_model = None

        self.__objective = objective
        self.__algo = algorithm

    def fit_shap(self):
        self.__shap_explainer = shap.TreeExplainer(self.__model)
        return

    def shap(self, data, type='summary_plot', num_features=None):
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
            return shap.summary_plot(shap_values, data, max_display=num_features)
        elif type == 'summary_bar_plot':
            return shap.summary_plot(shap_values, data, plot_type='bar', max_display=num_features)
        elif type == 'individual_plot':
            shap.initjs()
            return shap.force_plot(expected_value, shap_values, data)
        else:
            raise BaseException('Unknown SHAP plot type')

    def fit_skater(self, data):
        """
        :param data: Набор данных
        """
        self.__skater_explainer = Interpretation(data, feature_names=data.columns)

        if self.__objective == 'classification':
            self.__annotated_model = InMemoryModel(self.__model.predict_proba, examples=data)
        elif self.__objective == 'regression':
            self.__annotated_model = InMemoryModel(self.__model.predict, examples=data)

    def pdp(self, features, grid_resolution=30, n_samples=10000):
        """
        Возврщает график PDP
        :param features: tuple из 1 или 2 фичей
        :param grid_resolution: Количество ячеек по каждой из осей
        :param n_samples: The number of samples to use from the original dataset
        :return: Возвращает график PDP
        """

        if self.__skater_explainer is None or self.__annotated_model is None:
            raise BaseException("Skater explainer is not fitted. Run fit_skater at first")

        pdp_features = [features]

        return self.__skater_explainer.partial_dependence.plot_partial_dependence(pdp_features,
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

    def get_decision_rules(self, X_train, y_train, filename):
        """
        ВАЖНО! Работает только для обучающей выборки
        :X_train: DataFrame, 
        :y_train: Series or numpy array, вектор таргетов
        """

        surrogate_explainer = self.__skater_explainer.tree_surrogate(oracle=self.__annotated_model, seed=33)
        f1 = surrogate_explainer.fit(X_train, y_train, use_oracle=True, prune='pre', scorer_type='f1')
        print('F1 score for the surrogate tree: ', f1)

        def plot_tree_new(self, features_names, colors=None,
                          enable_node_id=True, random_state=0,
                          file_name=filename,
                          show_img=False, fig_size=(20, 8)):
            """ Visualizes the decision policies of the surrogate tree.
            """
            self.feature_names = features_names
            graph_inst = plot_tree(self.__model, self.__mfodel_type, feature_names=self.feature_names,
                                   color_list=colors,
                                   class_names=self.class_names, enable_node_id=enable_node_id, seed=random_state)
            f_name = "interpretable_tree.png" if file_name is None else file_name
            graph_inst.write_png(f_name)

            try:
                import matplotlib
                matplotlib.use('agg')
                import matplotlib.pyplot as plt
            except ImportError:
                raise exceptions.MatplotlibUnavailableError("Matplotlib is required but unavailable on the system.")
            except RuntimeError:
                raise exceptions.MatplotlibDisplayError("Matplotlib unable to open display")

            if show_img:
                plt.rcParams["figure.figsize"] = fig_size
                img = plt.imread(f_name)
                if self.__model_type == 'regressor':
                    cax = plt.imshow(img, cmap=plt.cm.get_cmap(graph_inst.get_colorscheme()))
                    plt.colorbar(cax)
                else:
                    plt.imshow(img)
            return graph_inst

        surrogate_explainer.plot_tree = types.MethodType(plot_tree_new, surrogate_explainer)
        surrogate_explainer.plot_tree(X_train.columns)

        show_in_notebook(filename, width=1200, height=800);

    def lime(self, data, index_examples, class_names=None):
        """
        Важно! Для LIME модель должна быть обучена на numpy array
        :data: DataFrame, датасет с исходными данными
        :class_names: имена классов 
        :index_example: list, номер индекса объекта, который хотим интерпретировать
        """
        # принимает в качестве данных только numpy array
        exp = LimeTabularExplainer(data.values, feature_names=data.columns, discretize_continuous=True,
                                   class_names=class_names)
        if not isinstance(index_examples, list):
            raise BaseException("index_examples must be list")
        for i in index_examples:
            predictions = self.__model.predict_proba(data.values)
            print('Predicted:', predictions[i])
            exp.explain_instance(data.iloc[i].values, self.__model.predict_proba).show_in_notebook()

    def plot_feature_importances(self, column_list, plot_size=(14, 5)):
        """
        круто бы чекнуть если ли у модели метод feature_importnaces
        """
        feature_imp_attr = getattr(self.__model, 'feature_importances_', None)
        if feature_imp_attr is None:
            raise BaseException('Model has no \'feature_importances_\' attribute')

        featimp_df = pd.DataFrame(data={'feature_name': column_list,
                                        'feature_importance': self.__model.feature_importances_})
        sns.barplot(data=featimp_df.sort_values(by=['feature_importance'], ascending=False),
                    x='feature_importance', y='feature_name')
        plt.gcf().set_size_inches(plot_size)
        plt.tight_layout()
        plt.show()
