import time
from sklearn.metrics import f1_score
from personal import datainput, Personal


def main(filename, u, alpha_):
    filepath = f"dataset/{filename}.txt"
    db, labels, alphabet, _ = datainput(filepath)
    labels_set = sorted(set(labels))
    char_to_index = {char: idx + 1 for idx, char in enumerate(alphabet)}
    X = [[char_to_index.get(char, -1) for char in seq] for seq in db]
    label_to_index = {label: idx for idx, label in enumerate(labels_set)}
    y = [label_to_index[label] for label in labels]

    s_time = time.time()
    p = Personal(k=u)
    p.fit(X, y)
    acc, size, length, y_pred = p.predict(X, y, alpha_)
    elapsed = time.time() - s_time

    f1_macro = f1_score(y, y_pred, average='macro', zero_division=0)

    print(f'{filename:<12} {acc*100:<10.2f} {f1_macro*100:<14.2f} {size:<10} {length:<10.2f} {elapsed:<10.2f}')


if __name__ == "__main__":
    datasets = ['activity', 'aslbu', 'auslan2', 'context', 'epitope', 'gene', 'news',
                'pioneer', 'question', 'reuters', 'robot', 'skating', 'unix', 'webkb']
    alpha_ = 0.9
    u = 4

    print(f'{"="*66}')
    print(f'  alpha={alpha_}  k={u}')
    print(f'{"="*66}')
    print(f'{"Dataset":<12} {"Acc(%)":<10} {"F1-macro(%)":<14} {"Rules":<10} {"Avg Len":<10} {"Time(s)":<10}')
    print(f'{"-"*66}')
    for dataset in datasets:
        main(dataset, u, alpha_)
    print(f'{"="*66}')
