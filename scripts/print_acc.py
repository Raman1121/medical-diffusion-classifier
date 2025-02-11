import argparse
import os
import os.path as osp
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix

def mean_per_class_acc(correct, labels):
    total_acc = 0
    for cls in torch.unique(labels):
        mask = labels == cls
        total_acc += correct[mask].sum() / mask.sum()
    return total_acc / len(torch.unique(labels))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=str)
    args = parser.parse_args()

    # get list of files
    files = os.listdir(args.folder)
    files = sorted([f for f in files if f.endswith('.pt')])

    preds = []
    labels = []
    for f in tqdm(files):
        data = torch.load(osp.join(args.folder, f))
        preds.append(data['pred'])
        labels.append(data['label'])
    preds = torch.tensor(preds)
    labels = torch.tensor(labels)
    # top 1
    correct = (preds == labels).sum().item()
    print(f'Top 1 acc: {correct / len(preds) * 100:.2f}%')
    # mean per class
    print(f'Mean per class acc: {mean_per_class_acc(preds == labels, labels) * 100:.2f}%')

    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()

    cm = confusion_matrix(labels_np, preds_np)

    np.save(os.path.join(args.folder, 'confusion_matrix.npy'), cm)


if __name__ == '__main__':
    main()
