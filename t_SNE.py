import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris,load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

def plot_embedding(X, y, d, title=None, fig_mode='display'):
    if fig_mode is None:
        return

    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    unique_labels = np.unique(y)
    unique_domains = np.unique(d)
    color_map = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))
    label_to_color = dict(zip(unique_labels, color_map))

    source_marker = 'o'
    target_marker = 'x'

    for label in unique_labels:
        for domain in unique_domains:
            indices = (y == label) & (d == domain)
            x_coords = X[indices, 0]
            y_coords = X[indices, 1]
            label_color = label_to_color[label]
            marker = source_marker if domain == 0 else target_marker
            plt.scatter(x_coords, y_coords, color=label_color, marker=marker, s=10, )
            for i in range(len(x_coords)):
                plt.annotate(str(label), (x_coords[i], y_coords[i]), color=label_color, fontsize=5)
    plt.show()

    #
    # # Plot colors numbers
    # plt.figure(figsize=(10, 10))
    # ax = plt.subplot(111)
    #
    # for i in range(X.shape[0]):
    #     # plot colored number
    #     plt.text(X[i, 0], X[i, 1], str(y[i]),
    #              color=plt.cm.bwr(d[i] / 1.),
    #              fontdict={'weight': 'bold', 'size': 12})
    #
    # plt.xticks([]), plt.yticks([])
    # # plt.scatter(X[:, 0], X[:, 1], c=y)
    #
    # # If title is not given, we assign training_mode to the title.
    # if title is not None:
    #     plt.title(title)
    # # else:
    # #     plt.title(params.training_mode)
    #
    # if fig_mode == 'display':
    #     # Directly display if no folder provided.
    #     plt.show()

def visualizePerformance(feature_extractor, F1,  src_test_dataloader,
                         tgt_test_dataloader, num_of_samples=None,
                         title=None):
    batch_size = 100
    use_gpu = True

    # s_images_f = torch.FloatTensor(1)
    # t_images_f = torch.FloatTensor(1)
    #
    # s_images_f = s_images_f.cuda()
    # t_images_f = t_images_f.cuda()



    # Setup the network
    feature_extractor.eval()
    # domain_classifier.eval()

    # Randomly select samples from source domain and target domain.
    if num_of_samples is None:
        num_of_samples = batch_size
    else:
        assert len(src_test_dataloader) * num_of_samples, \
            'The number of samples can not bigger than dataset.'  # NOT PRECISELY COMPUTATION

    # Collect source data.
    s_images, s_labels, s_tags = [], [], []
    length = 0
    for batch in src_test_dataloader:
        images, labels,_ = batch
        # im_data_s.resize_(data_s[0].size()).copy_(data_s[0])
        if use_gpu:
            s_images.append(images.cuda())
        else:
            s_images.append(images)

        s_labels.append(labels)

        s_tags.append(torch.zeros((labels.size()[0])).type(torch.LongTensor))

        length += len(s_labels)
        if (length) > num_of_samples:
            break

    s_images, s_labels, s_tags = torch.cat(s_images)[:num_of_samples], \
        torch.cat(s_labels)[:num_of_samples], torch.cat(s_tags)[:num_of_samples]
    s_images = s_images.to(torch.float32).cuda()

    # Collect test data.
    t_images, t_labels, t_tags = [], [], []
    t_length = 0
    for batch in tgt_test_dataloader:
        images, labels, _ = batch

        if use_gpu:
            t_images.append(images.cuda())
        else:
            t_images.append(images)
        t_labels.append(labels)

        t_tags.append(torch.ones((labels.size()[0])).type(torch.LongTensor))
        t_length += len(labels)
        # len(t_images) * images.shape[0]
        if t_length > num_of_samples:
            break

    t_images, t_labels, t_tags = torch.cat(t_images)[:num_of_samples], \
        torch.cat(t_labels)[:num_of_samples], torch.cat(t_tags)[:num_of_samples]
    t_images = t_images.to(torch.float32).cuda()
    # Compute the embedding of target domain.

    embedding1 = feature_extractor(s_images.unsqueeze(1))
    embedding2 = feature_extractor(t_images.unsqueeze(1))

    embedding1 = F1(embedding1)
    embedding2 = F1(embedding2)

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=1000)

    if use_gpu:
        dann_tsne = tsne.fit_transform(np.concatenate((embedding1.cpu().detach().numpy(),
                                                       embedding2.cpu().detach().numpy())))

    else:
        dann_tsne = tsne.fit_transform(np.concatenate((embedding1.detach().numpy(),
                                                       embedding2.detach().numpy())))

    plot_embedding(dann_tsne, np.concatenate((s_labels, t_labels)),
                   np.concatenate((s_tags, t_tags)), title=title)


#
# def plot_embedding(X, y, d, title=None, imgName=None, fig_mode='display'):
#     """
#     Plot an embedding X with the class label y colored by the domain d.
#
#     :param X: embedding
#     :param y: label
#     :param d: domain
#     :param title: title on the figure
#     :param imgName: the name of saving image
#
#     :return:
#     """
#     if fig_mode is None:
#         return
#
#     # normalization
#     x_min, x_max = np.min(X, 0), np.max(X, 0)
#     X = (X - x_min) / (x_max - x_min)
#
#     # Plot colors numbers
#     plt.figure(figsize=(10,10))
#     ax = plt.subplot(111)
#
#     for i in range(X.shape[0]):
#         # plot colored number
#         plt.text(X[i, 0], X[i, 1], str(y[i]),
#                  color=plt.cm.bwr(d[i]/1.),
#                  fontdict={'weight': 'bold', 'size': 3})
#
#     plt.xticks([]), plt.yticks([])
#     # plt.scatter(X[:, 0], X[:, 1], c=y)
#
#     # If title is not given, we assign training_mode to the title.
#     if title is not None:
#         plt.title(title)
#     # else:
#     #     plt.title(params.training_mode)
#
#     if fig_mode == 'display':
#         # Directly display if no folder provided.
#         plt.show()
#
#     # if params.fig_mode == 'save':
#     #     # Check if folder exist, otherwise need to create it.
#     #     folder = os.path.abspath(params.save_dir)
#     #
#     #     if not os.path.exists(folder):
#     #         os.makedirs(folder)
#     #
#     #     if imgName is None:
#     #         imgName = 'plot_embedding' + str(int(time.time()))
#     #
#     #     # Check extension in case.
#     #     if not (imgName.endswith('.jpg') or imgName.endswith('.png') or imgName.endswith('.jpeg')):
#     #         imgName = os.path.join(folder, imgName + '.jpg')
#     #
#     #     print('Saving ' + imgName + ' ...')
#     #     plt.savefig(imgName)
#     #     plt.close()
#
# def visualizePerformance(feature_extractor, class_classifier, src_test_dataloader,
#                          tgt_test_dataloader, num_of_samples=1000, imgName=None):
#
#     """
#     Evaluate the performance of dann and source only by visualization.
#
#     :param feature_extractor: network used to extract feature from target samples
#     :param class_classifier: network used to predict labels
#     :param domain_classifier: network used to predict domain
#     :param source_dataloader: test dataloader of source domain
#     :param target_dataloader: test dataloader of target domain
#     :param num_of_samples: the number of samples (from train and test respectively) for t-sne
#     :param imgName: the name of saving image
#
#     :return:
#     """
#     batch_size = 512
#     use_gpu = True
#
#     # Setup the network
#     feature_extractor.eval()
#     class_classifier.eval()
#     # domain_classifier.eval()
#
#     # Randomly select samples from source domain and target domain.
#     if num_of_samples is None:
#         num_of_samples = batch_size
#     else:
#         assert len(src_test_dataloader) * num_of_samples, \
#             'The number of samples can not bigger than dataset.' # NOT PRECISELY COMPUTATION
#
#     # Collect source data.
#     s_images, s_labels, s_tags = [], [], []
#     for batch in src_test_dataloader:
#         images, labels = batch
#
#         if use_gpu:
#             s_images.append(images.cuda())
#         else:
#             s_images.append(images)
#         s_labels.append(labels)
#
#         s_tags.append(torch.zeros((labels.size()[0])).type(torch.LongTensor))
#
#         print('len of s_images * batch_size is {}'.format(len(s_images * batch_size)))
#
#         if (len(s_images) * images.shape[0]) > num_of_samples:
#             break
#
#     s_images, s_labels, s_tags = torch.cat(s_images)[:num_of_samples], \
#                                  torch.cat(s_labels)[:num_of_samples], torch.cat(s_tags)[:num_of_samples]
#     print('s_image shape'.format(s_images))
#
#     # Collect test data.
#     t_images, t_labels, t_tags = [], [], []
#     for batch in tgt_test_dataloader:
#         images, labels = batch
#
#         if use_gpu:
#             t_images.append(images.cuda())
#         else:
#             t_images.append(images)
#         t_labels.append(labels)
#
#         t_tags.append(torch.ones((labels.size()[0])).type(torch.LongTensor))
#
#         if (len(t_images) * images.shape[0]) > num_of_samples:
#             break
#
#     t_images, t_labels, t_tags = torch.cat(t_images)[:num_of_samples], \
#                                  torch.cat(t_labels)[:num_of_samples], torch.cat(t_tags)[:num_of_samples]
#
#     # Compute the embedding of target domain.
#     embedding1 = feature_extractor(s_images)
#     embedding2 = feature_extractor(t_images)
#
#     tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=2000)
#
#     if use_gpu:
#         dann_tsne = tsne.fit_transform(np.concatenate((embedding1.cpu().detach().numpy(),
#                                                        embedding2.cpu().detach().numpy())))
#     else:
#         dann_tsne = tsne.fit_transform(np.concatenate((embedding1.detach().numpy(),
#                                                    embedding2.detach().numpy())))
#
#     # pd.DataFrame(dann_tsne).to_csv('tsne_result.csv')
#     # pd.DataFrame(np.concatenate((s_labels, t_labels))).to_csv('tsne_result_labels.csv')
#     # pd.DataFrame(np.concatenate((s_tags, t_tags))).to_csv('tsne_result_d.csv')
#
#     plot_embedding(dann_tsne, np.concatenate((s_labels, t_labels)),
#                          np.concatenate((s_tags, t_tags)), 'UDA')
