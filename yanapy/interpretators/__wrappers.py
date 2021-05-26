def requires_shap(self, func):
    def wrapper(*args, **kwargs):
        # Проверка параметров
        if self.__shap_explainer is None:
            raise BaseException("SHAP explainer is not fitted. Run fit_shap at first")
        func(*args, **kwargs)
    return wrapper


def requires_skater(self, func):
    def wrapper(*args, **kwargs):
        # Проверка параметров
        if self.__skater_explainer is None or self.__annotated_model is None:
            raise BaseException("Skater explainer is not fitted. Run fit_skater at first")
        func(*args, **kwargs)
    return wrapper


def rf_only(self, func):
    def wrapper(*args, **kwargs):
        if self.__algo != 'random_forest':
            raise BaseException("Can be used only for Random Forest")
        func(*args, **kwargs)
    return wrapper
