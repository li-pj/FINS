import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics import accuracy_score


def datainput(filepath):
    max_sequence_length = 0
    db = []
    data_label = []
    with open(filepath, 'r') as file:
        for line in file:
            temp = line.strip().split('\t')
            seq_db = temp[1].split(" ")
            max_sequence_length = max(max_sequence_length, len(seq_db))
            db.append(seq_db)
            data_label.append(str(temp[0]))
    itemset = set([item for sublist in db for item in sublist])
    itemset = list(itemset)
    int_itemset = [str(x) for x in itemset]
    int_itemset.sort()
    itemset = [str(x) for x in int_itemset]
    return db, data_label, itemset, max_sequence_length


def LCS(seq, sub_seq):
    m, n = len(seq), len(sub_seq)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq[i - 1] == sub_seq[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


class Personal:

    def __init__(self, k):
        self.counts_hashmap = defaultdict(lambda: defaultdict(int))
        self.train_count = None
        self.rule_set = set()
        self.k = k

    def fit(self, X_train, y_train):
        self.train_count = Counter(y_train)
        for seq, label in zip(X_train, y_train):
            temp_set = set()
            for i in range(1, self.k + 1):
                for j in range(0, len(seq) - i + 1):
                    temp_set.add(tuple(seq[j:j + i]))
            for kmer in temp_set:
                self.counts_hashmap[kmer][label] += 1

    def predict(self, X_test, y_test, alpha_=0.9):

        def eval_function(pattern):
            count = self.counts_hashmap.get(tuple(pattern), 0)
            if count == 0:
                return 0, None
            best_score = 0
            best_label = None
            for obj_label, obj_freq in count.items():
                obj_num = self.train_count.get(obj_label)
                other_num, other_freq = 0, 0
                for label_, freq_ in self.train_count.items():
                    if label_ != obj_label:
                        other_num += freq_
                for label_, freq_ in count.items():
                    if label_ != obj_label:
                        other_freq += freq_
                tp = obj_freq
                fp = other_freq
                fn = obj_num - obj_freq
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                score = alpha_ * precision + (1 - alpha_) * recall
                if score > best_score:
                    best_score = score
                    best_label = obj_label
            return best_score, best_label

        def generate_kmers(s, k):
            kmers = []
            for i in range(len(s) - k + 1):
                kmers.append(list(s[i:i + k]))
            return kmers

        y_pred = []
        unmatched_idx = []

        for idx, sample in enumerate(X_test):
            best_score = 0
            top_rules = []
            for k in range(1, self.k + 1):
                kmers = generate_kmers(sample, k)
                for kmer in kmers:
                    score, label = eval_function(kmer)
                    if score > best_score:
                        best_score = score
                        top_rules = [(kmer, label, score)]
                    elif score == best_score and best_score > 0:
                        top_rules.append((kmer, label, score))

            if len(top_rules) > 1:
                label_counter = Counter(r[1] for r in top_rules)
                most_frequent_label = label_counter.most_common(1)[0][0]
                best_kmer, best_label, _ = next(r for r in top_rules if r[1] == most_frequent_label)
            elif len(top_rules) == 1:
                best_kmer, best_label, _ = top_rules[0]
            else:
                best_kmer, best_label = None, None

            if best_score > 0:
                self.rule_set.add((tuple(best_kmer), best_label, best_score))
                y_pred.append(best_label)
            else:
                unmatched_idx.append(idx)
                y_pred.append(None)

        for idx in unmatched_idx:
            sample = X_test[idx]
            best_score = 0
            best_label = None
            for rule in self.rule_set:
                kmer, label, score = rule
                kmer = list(kmer)
                if len(sample) < len(kmer) or len(kmer) == 0:
                    continue
                lcs = LCS(sample, list(kmer))
                if lcs >= 0 and score > best_score:
                    best_label = label
                    best_score = score * lcs / len(kmer)
            y_pred[idx] = best_label

        accuracy = accuracy_score(y_test, y_pred)
        size = len(self.rule_set)
        length = np.mean([len(rule[0]) for rule in self.rule_set])

        return accuracy, size, length, y_pred
