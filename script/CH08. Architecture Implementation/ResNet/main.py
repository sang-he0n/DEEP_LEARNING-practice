#00. __
import argparse
import utils
import datasets

if __name__ == "__main__":
    #01. __
    parser = argparse.ArgumentParser(description='CIFAR10 image classification')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--epoch_num', default=100, type=int, help='training epoch')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--l2', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--model_nm', default='resnet18', type=str, help='model name(type)')
    parser.add_argument('--mode', default='train', type=str, help='train and eval')
    parser.add_argument('--use_cp_yn', default='Y', type=str, help='checkpoint')
    parser.add_argument('--seed_num', default=2025, type=int, help='seed number')
    args = parser.parse_args()
    #02. __
    learning = utils.SupervisedLearning(
        model_nm=args.model_nm, 
        use_checkpoint_yn=args.use_cp_yn,
    )
    if args.mode == 'train' :
        train_loader, test_loader = datasets.load_dataset(batch_size=args.batch_size, mode='train')
        print('>> Loaded datasets successfully.')
        learning.train(train_loader=train_loader, epoch_num=args.epoch_num, learning_rate=args.lr, l2_rate=args.l2)
    else :
        train_loader, test_loader = datasets.load_dataset(batch_size=args.batch_size, mode='eval')
        print('>> Loaded datasets successfully.')
        train_metrics = learning.eval(loader=train_loader)
        test_metrics = learning.eval(loader=test_loader)
        print(f'>> Training Metrics : {train_metrics}.')
        print(f'>> Testing Metrics : {test_metrics}')

