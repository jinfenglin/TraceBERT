import os

import pandas as pd
from pandas import DataFrame
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
import matplotlib.pyplot as plt


class metrics:
    def __init__(self, data_frame: DataFrame, output_dir=None):
        """
        Evaluate the performance given datafrome with column "s_id", "t_id" "pred" and "label"
        :param data_frame:
        """
        self.data_frame = data_frame
        self.output_dir = output_dir
        self.s_ids, self.t_ids = data_frame['s_id'], data_frame['t_id']
        self.pred, self.label = data_frame['pred'], data_frame['label']
        self.group_sort = None

    def f1_score(self, precision, recall):
        return 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    def f2_score(self, precision, recall):
        return 5 * precision * recall / (4 * precision + recall) if precision + recall > 0 else 0

    def f1_details(self, threshold):
        "Return ture positive (tp), fp, tn,fn "
        f_name = "f1_details"
        tp, fp, tn, fn = 0, 0, 0, 0
        for p, l in zip(self.pred, self.label):
            if p > threshold:
                p = 1
            else:
                p = 0
            if p == l:
                if l == 1:
                    tp += 1
                else:
                    tn += 1
            else:
                if l == 1:
                    fp += 1
                else:
                    fn += 1
        return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}

    def precision_recall_curve(self, fig_name):
        precision, recall, thresholds = precision_recall_curve(self.label, self.pred)
        max_f1 = 0
        max_f2 = 0
        max_threshold = 0
        for p, r, tr in zip(precision, recall, thresholds):
            f1 = self.f1_score(p, r)
            f2 = self.f2_score(p, r)
            if f1 >= max_f1:
                max_f1 = f1
                max_threshold = tr
            if f2 >= max_f2:
                max_f2 = f2
        viz = PrecisionRecallDisplay(
            precision=precision, recall=recall)
        viz.plot()
        if os.path.isdir(self.output_dir):
            fig_path = os.path.join(self.output_dir, fig_name)
            plt.savefig(fig_path)
            plt.close()
        detail = self.f1_details(max_threshold)
        return round(max_f1, 3), round(max_f2, 3), detail, max_threshold

    def precision_at_K(self, k=1):
        if self.group_sort is None:
            self.group_sort = self.data_frame.groupby(["s_id"]).apply(
                lambda x: x.sort_values(["pred"], ascending=False)).reset_index(drop=True)
        group_tops = self.group_sort.groupby('s_id')
        cnt = 0
        hits = 0
        for s_id, group in group_tops:
            for index, row in group.head(k).iterrows():
                hits += 1 if row['label'] == 1 else 0
            cnt += 1
        return round(hits / cnt if cnt > 0 else 0, 3)

    def MAP_at_K(self, k=1):
        if self.group_sort is None:
            self.group_sort = self.data_frame.groupby(["s_id"]).apply(
                lambda x: x.sort_values(["pred"], ascending=False)).reset_index(drop=True)
        group_tops = self.group_sort.groupby('s_id')
        ap_sum = 0
        for s_id, group in group_tops:
            group_hits = 0
            ap = 0
            for i, (index, row) in enumerate(group.head(k).iterrows()):
                if row['label'] == 1:
                    group_hits += 1
                    ap += group_hits / (i + 1)
            ap = ap / group_hits if group_hits > 0 else 0
            ap_sum += ap
        map = ap_sum / len(group_tops) if len(group_tops) > 0 else 0
        return round(map, 3)

    def MRR(self):
        if self.group_sort is None:
            self.group_sort = self.data_frame.groupby(["s_id"]).apply(
                lambda x: x.sort_values(["pred"], ascending=False)).reset_index(drop=True)
        group_tops = self.group_sort.groupby('s_id')
        mrr_sum = 0
        for s_id, group in group_tops:
            rank = 0
            for i, (index, row) in enumerate(group.iterrows()):
                rank += 1
                if row['label'] == 1:
                    mrr_sum += 1.0 / rank
                    break
        return mrr_sum / len(group_tops)

    def get_all_metrices(self):
        pk3 = self.precision_at_K(3)
        pk2 = self.precision_at_K(2)
        pk1 = self.precision_at_K(1)

        best_f1, best_f2, details, f1_threshold = self.precision_recall_curve("pr_curve.png")
        map = self.MAP_at_K(3)
        mrr = self.MRR()
        return {
            'pk3': pk3,
            'pk2': pk2,
            'pk1': pk1,
            'f1': best_f1,
            'f2': best_f2,
            'map': map,
            'mrr': mrr,
            'details': details,
            'f1_threshold': f1_threshold
        }

    def write_summary(self, exe_time):
        summary_path = os.path.join(self.output_dir, "summary.txt")
        res = self.get_all_metrices()
        pk3, pk2, pk1 = res['pk3'], res['pk2'], res['pk1']
        best_f1, best_f2, details = res['f1'], res['f2'], res['details']
        map, mrr = res['map'], res['mrr']
        summary = "\npk3={}, pk2={},pk1={} best_f1 = {}, bets_f2={}, MAP={}, MRR={}, exe_time={},f1_threshold={}\n".format(
            pk3,
            pk2,
            pk1,
            best_f1,
            best_f2,
            map,
            mrr,
            exe_time,
            res['f1_threshold']
        )
        with open(summary_path, 'w') as fout:
            fout.write(summary)
            fout.write(str(details))
        print(summary)


if __name__ == "__main__":
    test = [
        (1, 1, 0.8, 1),
        (1, 2, 0.3, 0),
        (2, 1, 0.9, 1),
        (2, 1, 0, 0),
        (3, 1, 0.5, 0)
    ]
    df = pd.DataFrame(test, columns=['s_id', 't_id', 'pred', 'label'])
    m = metrics(df)
    m.precision_recall_curve('test.png')
    print(m.precision_at_K(2))
    print(m.MAP_at_K(2))
