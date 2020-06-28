from pandas import DataFrame
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay


class metrics:
    def __init__(self, data_frame: DataFrame):
        """
        Evaluate the performance given datafrome with column "s_id", "t_id" "pred" and "label"
        :param data_frame:
        """
        self.data_frame = data_frame
        self.s_ids, self.t_ids = data_frame['s_id'], data_frame['t_id']
        self.pred, self.label = data_frame['pred'], data_frame['label']
        self.group_sort = None

    def f1_score(self, precision, recall):
        return 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    def precision_recall_curve(self, output_path):
        precision, recall, thresholds = precision_recall_curve(self.label, self.pred)
        max_f1 = 0
        for p, r in zip(precision, recall):
            f1 = self.f1_score(p, r)
            max_f1 = max(f1, max_f1)
        viz = PrecisionRecallDisplay(
            precision=precision, recall=recall)
        fig = viz.plot().figure_
        if output_path:
            fig.saveifg(output_path)
        return max_f1

    def precision_at_K(self, k=1):
        if not self.group_sort:
            self.group_sort = self.data_frame.groupby(["s_id"]).apply(
                lambda x: x.sort_values(["pred"], ascending=False))
        group_tops = self.group_sort.head(k)
        g_cnt = 0
        pos_cnt = 0
        for name, group in group_tops:
            g_cnt += 1
            for index, row in group:
                hits = [x for x in row['label'] if x == 1]
                pos_cnt += hits
        return pos_cnt / g_cnt * k if g_cnt > 0 else 0

    def MAP_at_K(self, k=1):
        if not self.group_sort:
            self.group_sort = self.data_frame.groupby(["s_id"]).apply(
                lambda x: x.sort_values(["pred"], ascending=False))
        group_tops = self.group_sort.head(k)
        ap_sum = 0
        for name, group in group_tops:
            precisions = []
            for index, row in group:
                precisions = [x / (i + 1) for i, x in enumerate(row['label']) if x == 1]
            ap = sum(precisions) / len(precisions) if len(precisions) > 0 else 0
            ap_sum += ap
        map = ap_sum / len(group_tops)
        return map
