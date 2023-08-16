import torch
import logging
import numpy as np
from datasets.imagenet_subsets import IMAGENET_D_MAPPING

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def split_results_by_domain(domain_dict, data, predictions):
    """
    Create a dictionary which separates the labels and predictions by domain
    :param domain_dict: dictionary, where the keys are the domains and the content is [labels, predictions]
    :param data: list containing [images, labels, domains, ...]
    :param predictions: predictions of the model
    :return: updated result dict
    """

    imgs = data[0][0] if isinstance(data[0], list) else data[0]

    for i in range(imgs.shape[0]):
        label, domain = data[1][i], data[2][i]
        if domain in domain_dict.keys():
            domain_dict[domain].append([label.item(), predictions[i].item()])
        else:
            domain_dict[domain] = [[label.item(), predictions[i].item()]]

    return domain_dict


def eval_domain_dict(domain_dict, domain_seq=None):
    """
    Print detailed results for each domain. This is useful for settings where the domains are mixed
    :param domain_dict: dictionary containing the labels and predictions for each domain
    :param domain_seq: if specified and the domains are contained in the domain dict, the results will be printed in this order
    """
    correct = []
    num_samples = []
    avg_error_domains = []
    dom_names = domain_seq if all(
        [dname in domain_seq for dname in domain_dict.keys()]) else domain_dict.keys()
    logger.info(f"Splitting up the results by domain...")
    for key in dom_names:
        content = np.array(domain_dict[key])
        correct.append((content[:, 0] == content[:, 1]).sum())
        num_samples.append(content.shape[0])
        accuracy = correct[-1] / num_samples[-1]
        error = 1 - accuracy
        avg_error_domains.append(error)
        logger.info(f"{key:<20} error: {error:.2%}")
    total_err = 1 - sum(correct) / sum(num_samples)
    logger.info(
        f"Average error across all domains: {sum(avg_error_domains) / len(avg_error_domains):.2%}")
    logger.info(f"Error over all samples: {total_err:.2%}")


def get_accuracy(model: torch.nn.Module,
                 data_loader: torch.utils.data.DataLoader,
                 dataset_name: str,
                 domain_name: str,
                 setting: str,
                 domain_dict: dict,
                 device: torch.device = None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    correct = 0.
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            imgs, labels = data[0], data[1]
            output = model([img.to(device) for img in imgs]) if isinstance(
                imgs, list) else model(imgs.to(device))
            predictions = output.argmax(1)

            if dataset_name == "imagenet_d" and domain_name != "none":
                mapping_vector = list(IMAGENET_D_MAPPING.values())
                predictions = torch.tensor(
                    [mapping_vector[pred] for pred in predictions], device=device)

            correct += (predictions == labels.to(device)).float().sum()

            if "mixed_domains" in setting and len(data) >= 3:
                domain_dict = split_results_by_domain(
                    domain_dict, data, predictions)

    accuracy = correct.item() / len(data_loader.dataset)
    return accuracy, domain_dict


def plot_tsne(fea_bank, label_bank, fea_batch, preds_batch, labels_batch, acl_idx, acl_nn_batch):

    Feature = torch.cat([F.normalize(fea_batch, dim=1),
                        F.normalize(fea_bank, dim=1)])
    Label = torch.cat([preds_batch, label_bank+126])

    Feature, Label = Feature.cpu().numpy(), Label.cpu().numpy()

    tsne = TSNE(n_components=2, init='pca', random_state=0,
                perplexity=50, early_exaggeration=3)
    embedding = tsne.fit_transform(Feature)

    data_min, data_max = np.min(embedding, 0), np.max(embedding, 0)
    data_norm = (embedding - data_min) / (data_max - data_min)

    colors = {0: 'red', 1: 'blue', 2: 'olive', 3: 'green', 4: 'orange', 5: 'purple', 6: 'darkslategray', 7: 'saddlebrown',
              8: 'firebrick', 9: 'aquamarine'}  # , 10: 'goldenrod', 11: 'cadetblue'}
    labels = {0: 'airplane', 1: 'car', 2: 'bird',  3: 'cat',  4: 'deer',  5: 'dog',  6: 'frogs', 7: 'horse',
              8: 'ship', 9: 'truck'}  # , 10: 'train', 11:'truck'}

    fig = plt.figure()
    ax = plt.subplot(111)

    for i in range(126):
        # feature bank
        data_x, data_y = data_norm[Label == i +
                                   126][:, 0], data_norm[Label == i+126][:, 1]
        scatter = plt.scatter(
            data_x, data_y, c='none', edgecolors=colors[i % 10], s=2, label=labels[i % 10], marker='o', alpha=1.0)

        # current batch samples
        data_x, data_y = data_norm[Label ==
                                   i][:, 0], data_norm[Label == i][:, 1]
        scatter = plt.scatter(
            data_x, data_y, c=colors[i % 10], edgecolors=colors[i % 10], s=5, label=labels[i % 10], marker='^', alpha=1.0)

        # misclassified samples in current batch
        data_x = data_norm[:64][(labels_batch.cpu() == i) & (
            labels_batch.cpu() != preds_batch.cpu())][:, 0]
        data_y = data_norm[:64][(labels_batch.cpu() == i) & (
            labels_batch.cpu() != preds_batch.cpu())][:, 1]
        scatter = plt.scatter(
            data_x, data_y, c=colors[i % 10], edgecolors='black', s=5, label=labels[i % 10], marker='^', alpha=1.0)

    scatter = plt.scatter(data_norm[acl_idx][:, 0], data_norm[acl_idx][:, 1],
                          c='None', edgecolors='black', s=200, label=labels[i], marker='*', alpha=1.0)
    # scatter = plt.scatter(data_norm[sel_ent][:, 0], data_norm[sel_ent][:, 1], c='None', edgecolors='black', s=200, label=labels[i], marker='^', alpha=1.0)
    # scatter = plt.scatter(data_norm[sel_ent_na][:, 0], data_norm[sel_ent_na][:, 1], c='None', edgecolors='black', s=200, label=labels[i], marker='s', alpha=1.0)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height * 0.8])

    plt.legend(handles=[mpatches.Patch(color=colors[i], label="{:s}".format(labels[i])) for i in range(10)],
               loc='upper left',
               prop={'size': 8},
               #    prop=matplotlib.font_manager.FontProperties(fname='./simhei.ttf'),
               bbox_to_anchor=(1.05, 0.85),
               borderaxespad=0)
    plt.savefig('tsne.pdf', format="pdf", bbox_inches='tight')

    return
