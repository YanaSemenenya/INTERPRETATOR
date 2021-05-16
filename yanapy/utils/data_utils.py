from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr
from scipy.cluster import hierarchy


def select_uncorrelated_features(dataset, return_names=False, select_random=False):
    """
    Возвращает нескоррелированные фичи из набора данных
    :param dataset: набор данных (numpy array или pandas dataset)
    :param return_names: Флаг возвращать имена или порядковые номера фичей
    :param select_random: Флаг выбирать случаную фичу из группы скоррелированный или всегда первую
    :return: список фичей или их номеров
    """
    # формирую кластеры фичей
    corr = spearmanr(dataset).correlation
    corr_linkage = hierarchy.ward(corr)

    # номер кластера для каждой фичи
    cluster_ids = hierarchy.fcluster(corr_linkage, 1, criterion='distance')

    # формируем списки фичей по кластерам
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)

    # Вытаскиваем одну фичу из каждого кластера (возвращается её ID)
    selected_features = [np.random.choice(v, 1)[0] if select_random else v[0] for v in
                         cluster_id_to_feature_ids.values()]

    if return_names:
        return dataset.columns.values[selected_features]

    return selected_features
