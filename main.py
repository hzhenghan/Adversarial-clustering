from __future__ import print_function
import argparse
import os
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from model.LinearAverage import LinearAverage
from utils.loss import entropy, adentropy
from utils.lr_schedule import inv_lr_scheduler
from utils.return_dataset import return_dataset
from torch.utils.data import DataLoader
from model.basenet import FeatureExtractor
from model.basenet import Predictor_deep, ResClassifier_MME
from utils.utils import weights_init
from model.VIT import VisionTransformer
from sklearn.metrics import confusion_matrix as confusion_matrix_sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from t_SNE import visualizePerformance
# source_loader, target_loader, target_loader_unl, target_loader_val, \
#     target_loader_test, class_list = return_dataset()

# Training settings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='SSDA Classification')
parser.add_argument('--steps', type=int, default=50000, metavar='N',help='maximum number of iterations ''to train (default: 50000)')
parser.add_argument('--method', type=str, default='ST', choices=['ST', 'ENT', 'MME'],help='MME is proposed method, ENT is entropy minimization,'' S+T is training only on labeled examples')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',help='learning rate (default: 0.001)')
parser.add_argument('--T', type=float, default=0.0015, metavar='T', help='temperature (default: 0.05)')
parser.add_argument('--lamda', type=float, default=0.1, metavar='LAM',help='value of lamda')
parser.add_argument('--save_check', action='store_true', default=False,help='save checkpoint or not')
parser.add_argument('--checkpath', type=str, default='./save_model_ssda', help='dir to save checkpoint')
parser.add_argument('--seed', type=int, default=1, metavar='S',help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',help='how many batches to wait before logging ''training status')
parser.add_argument('--save_interval', type=int, default=500, metavar='N', help='how many batches to wait before saving a model')
parser.add_argument('--net', type=str, default='resnet34')
parser.add_argument('--source', type=int, default=40, help='source domain')
parser.add_argument('--target', type=int, default=50, help='target domain')
parser.add_argument('--num', type=int, default=3,  help='number of labeled examples in the target')
parser.add_argument('--patience', type=int, default=5, metavar='S', help='early stopping to wait for improvment ''before terminating. (default: 5 (5000 iterations))')
parser.add_argument('--early', action='store_false', default=False, help='early stopping on validation or not')
parser.add_argument('--batch_size',  default=8, help='batch size')
parser.add_argument('--UDA',  default=False, help='Whether use labeled target data')
parser.add_argument('--dataset',  default='multi', help='Whether use labeled target data')

args = parser.parse_args()
print(args)

source_dataset, target_dataset, target_dataset_val, target_dataset_unl, target_dataset_test = return_dataset()

source_loader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True,
                           drop_last=True)
target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True,
                           drop_last=True)
target_loader_val = DataLoader(target_dataset_val,
                               batch_size=args.batch_size,
                               shuffle=True, drop_last=True)
target_loader_unl = DataLoader(target_dataset_unl,
                               batch_size=args.batch_size * 2,
                               shuffle=True, drop_last=True)
target_loader_test = DataLoader(target_dataset_test,
                                batch_size=args.batch_size * 2,
                                shuffle=True, drop_last=True)

s_v_dataloader = DataLoader(source_dataset, batch_size=32, shuffle=True,
                           drop_last=True)

t_v_dataloader = DataLoader(target_dataset_unl,
                               batch_size=32,
                               shuffle=True, drop_last=True)

use_gpu = torch.cuda.is_available()

record_dir = 'record/%s/%s' % (args.dataset, args.method)
if not os.path.exists(record_dir):
    os.makedirs(record_dir)
record_file = os.path.join(record_dir,
                           '%s_net_%s_%s_to_%s_num_%s' %
                           (args.method, args.net, args.source,
                            args.target, args.num))

torch.cuda.manual_seed(args.seed)
# seed = 1
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)

G = FeatureExtractor()

F1 = Predictor_deep(num_class=8, inc=64, temp=args.T)

weights_init(F1)
lr = args.lr

G.cuda()
F1.cuda()

im_data_s = torch.FloatTensor(1)
im_data_t = torch.FloatTensor(1)
im_data_tu = torch.FloatTensor(1)
gt_labels_s = torch.LongTensor(1)
gt_labels_t = torch.LongTensor(1)
sample_labels_t = torch.LongTensor(1)
sample_labels_s = torch.LongTensor(1)


im_data_s = im_data_s.cuda()
im_data_t = im_data_t.cuda()
im_data_tu = im_data_tu.cuda()
gt_labels_s = gt_labels_s.cuda()
gt_labels_t = gt_labels_t.cuda()
sample_labels_t = sample_labels_t.cuda()
sample_labels_s = sample_labels_s.cuda()


im_data_s = Variable(im_data_s)
im_data_t = Variable(im_data_t)
im_data_tu = Variable(im_data_tu)
gt_labels_s = Variable(gt_labels_s)
gt_labels_t = Variable(gt_labels_t)
sample_labels_t = Variable(sample_labels_t)
sample_labels_s = Variable(sample_labels_s)

if os.path.exists(args.checkpath) == False:
    os.mkdir(args.checkpath)

def train():
    G.train()
    F1.train()

    optimizer_g = optim.SGD(list(G.parameters()), momentum=0.9, lr=0.01,
                            weight_decay=0.0005, nesterov=True)
    optimizer_f = optim.SGD(list(F1.parameters()), lr=1.0, momentum=0.9,
                            weight_decay=0.0005, nesterov=True)

    def zero_grad_all():
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()

    param_lr_g = []
    for param_group in optimizer_g.param_groups:
        param_lr_g.append(param_group["lr"])

    param_lr_f = []
    for param_group in optimizer_f.param_groups:
        param_lr_f.append(param_group["lr"])

    criterion = nn.CrossEntropyLoss().cuda()
    all_step = args.steps

    data_iter_s = iter(source_loader)
    data_iter_t = iter(target_loader)
    data_iter_t_unl = iter(target_loader_unl)

    len_train_source = len(source_loader)
    len_train_target = len(target_loader)
    len_train_target_semi = len(target_loader_unl)

    best_acc = 0
    counter = 0

    for step in range(all_step):

        optimizer_g = inv_lr_scheduler(param_lr_g, optimizer_g, step,
                                       init_lr=args.lr)
        optimizer_f = inv_lr_scheduler(param_lr_f, optimizer_f, step,
                                       init_lr=args.lr)
        lr = optimizer_f.param_groups[0]['lr']

        if step % len_train_target == 0:
            data_iter_t = iter(target_loader)
        if step % len_train_target_semi == 0:
            data_iter_t_unl = iter(target_loader_unl)
        if step % len_train_source == 0:
            data_iter_s = iter(source_loader)

        data_s = next(data_iter_s)
        data_t = next(data_iter_t)
        data_t_unl = next(data_iter_t_unl)

        im_data_s.resize_(data_s[0].size()).copy_(data_s[0])  # source_data
        gt_labels_s.resize_(data_s[1].size()).copy_(data_s[1])  # source_data_label
        im_data_t.resize_(data_t[0].size()).copy_(data_t[0])  # target_data
        gt_labels_t.resize_(data_t[1].size()).copy_(data_t[1])  # target_data_label
        im_data_tu.resize_(data_t_unl[0].size()).copy_(data_t_unl[0])  # unlabeled_data

        zero_grad_all()

        if args.UDA:
            data = im_data_s
            target = gt_labels_s
        else:
            data = torch.cat((im_data_s, im_data_t), 0)
            target = torch.cat((gt_labels_s, gt_labels_t), 0)
        # data = data.unsqueeze(1)
        # print(data)
        # print('shape of data is {}'.format(data.shape))
        # print('shape of data is {}'.format(target.shape))
        output = G(data.unsqueeze(1))
        out1 = F1(output)
        loss = criterion(out1, target)
        loss.backward(retain_graph=True)

        optimizer_g.step()
        optimizer_f.step()

        zero_grad_all()

        if not args.method == 'ST':
            # im_data_tu = VIT(im_data_tu)
            output = G(im_data_tu.unsqueeze(1))

            if args.method == 'ENT':
                loss_t = entropy(F1, output, args.lamda)
                loss_t.backward()
                optimizer_f.step()
                optimizer_g.step()
            elif args.method == 'MME':
                loss_t = adentropy(F1, output, args.lamda)
                loss_t.backward()
                optimizer_f.step()
                optimizer_g.step()
            else:
                raise ValueError('Method cannot be recognized.')
            log_train = 'S {} T {} Train Ep: {} lr{} \t ' \
                        'Loss Classification: {:.6f} Loss T {:.6f} ' \
                        'Method {}\n'.format(args.source, args.target,
                                             step, lr, loss.data,
                                             -loss_t.data, args.method)
        else:
            log_train = 'S {} T {} Train Ep: {} lr{} \t ' \
                        'Loss Classification: {:.6f} Method {}\n'. \
                format(args.source, args.target,
                       step, lr, loss.data,
                       args.method)
        G.zero_grad()
        F1.zero_grad()
        zero_grad_all()

        if step % args.log_interval == 0:
            print(log_train)

        if step % args.save_interval == 0 and step > 0:

            loss_val, acc_val = test(target_loader_val)
            loss_test, acc_test = test(target_loader_test)

            # Evaluate the performance by visualization

            # visualizePerformance(G, F1, s_v_dataloader,
            #                      t_v_dataloader,
            #                      num_of_samples= 150,
            #                      title='UDA of S {} T {} step {}'.format(args.source, args.target, step))

            G.train()
            F1.train()
            if acc_val >= best_acc:
                best_acc = acc_val
                best_acc_test = acc_test
                counter = 0
            else:
                counter += 1
            if args.early:
                if counter > args.patience:
                    break
            print('best acc test %f best acc val %f' % (best_acc_test,
                                                        acc_val))
            print('record %s' % record_file)
            with open(record_file, 'a') as f:
                f.write('step %d best %f final %f \n' % (step,
                                                             best_acc_test,
                                                             acc_val))

            G.train()
            F1.train()
            if args.save_check:
                print('saving model')
                torch.save(G.state_dict(),
                           os.path.join(args.checkpath,
                                        "G_iter_model_{}_{}_"
                                        "to_{}_step_{}.pth.tar".
                                        format(args.method, args.source,
                                               args.target, step)))
                torch.save(F1.state_dict(),
                           os.path.join(args.checkpath,
                                        "F1_iter_model_{}_{}_"
                                        "to_{}_step_{}.pth.tar".
                                        format(args.method, args.source,
                                               args.target, step)))

def test(loader):
    G.eval()
    F1.eval()
    test_loss = 0
    correct = 0
    size = 0
    num_class = 8
    output_all = np.zeros((0, num_class))
    criterion = nn.CrossEntropyLoss().cuda()
    confusion_matrix = torch.zeros(num_class, num_class)
    num_classes = 8
    class_TP = np.zeros(num_classes)
    class_FP = np.zeros(num_classes)
    class_FN = np.zeros(num_classes)

    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for batch_idx, data_t in enumerate(loader):
            im_data_t.resize_(data_t[0].size()).copy_(data_t[0])
            gt_labels_t.resize_(data_t[1].size()).copy_(data_t[1])
            feat = G(im_data_t.unsqueeze(1))
            output1 = F1(feat)
            output_all = np.r_[output_all, output1.data.cpu().numpy()]
            size += im_data_t.size(0)
            pred1 = output1.data.max(1)[1]

            # true_labels.extend(gt_labels_t.cpu().numpy())
            # predicted_labels.extend(pred1.cpu().numpy())
            #
            # for i in range(num_classes):
            #     gt_mask = gt_labels_t == i
            #     pred_mask = pred1 == i
            #     class_TP[i] += (gt_mask & pred_mask).sum()
            #     class_FP[i] += (~gt_mask & pred_mask).sum()
            #     class_FN[i] += (gt_mask & ~pred_mask).sum()

            for t, p in zip(gt_labels_t.view(-1), pred1.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            correct += pred1.eq(gt_labels_t.data).cpu().sum()
            test_loss += criterion(output1, gt_labels_t) / len(loader)

    # class_accuracy = class_TP / (class_TP + class_FN)
    # class_precision = class_TP / (class_TP + class_FP)
    # class_recall = class_TP / (class_TP + class_FN)
    # class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall)

    # Print or use class-wise metrics as needed
    # for i in range(num_classes):
    #     print(f"Class {i}: Accuracy = {class_accuracy[i]:.2f}, Precision = {class_precision[i]:.2f}, Recall = {class_recall[i]:.2f}, F1-Score = {class_f1[i]:.2f}")

    print('\nTest set: Average loss: {:.4f}, '
          'Accuracy: {}/{} F1 ({:.6f}%)\n'.
          format(test_loss, correct, size,
                 correct / size))

    # confusion = confusion_matrix_sklearn(true_labels, predicted_labels)
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=range(8), yticklabels=range(8))
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.title('Confusion Matrix')
    # plt.show()
    return test_loss.data, 100. * float(correct) / size


if __name__ == '__main__':
    train()