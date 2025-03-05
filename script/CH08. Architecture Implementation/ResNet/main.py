#00. __
#(1)
import argparse
import torch
import utils
import datasets

if __name__ == "__main__":
    #01. __
    parser = argparse.ArgumentParser(description='Image Classification')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--epoch_num', default=100, type=int, help='training epoch')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--l2', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--model_nm', default='resnet18', type=str, help='model name(type)')
    parser.add_argument('--mode', default='train', type=str, help='train and eval')
    parser.add_argument('--use_cp_tf', default=True, type=bool, help='checkpoint flag(True/False)')
    parser.add_argument('--cp_path', default=None, type=str, help='checkpoint path')
    parser.add_argument('--seed_num', default=2025, type=int, help='seed number')
    args = parser.parse_args()
    #02. 
    torch.use_deterministic_algorithms(mode=True)
    torch.manual_seed(seed=args.seed_num)
    torch.mps.manual_seed(seed=args.seed_num)
    print('>> Set seed number successfully.')
    #03. Load dataset
    if args.mode == 'train' : 
        train_loader, test_loader = datasets.load_dataset(batch_size=args.batch_size, mode='train')
    else : 
        train_loader, test_loader = datasets.load_dataset(batch_size=args.batch_size, mode='eval')
    print('>> Loaded datasets successfully.')
    #04. Load Models 
    learning = utils.SupervisedLearning( 
        model_nm=args.model_nm, 
        use_checkpoint_tf=args.use_cp_tf,
        checkpoint_path=args.cp_path
    )
    #05. 
    if args.mode == 'train' :
        learning.train(train_loader=train_loader, epoch_num=args.epoch_num, learning_rate=args.lr, l2_rate=args.l2)
    else :
        print('>> Loaded datasets successfully.')
        train_metrics = learning.eval(loader=train_loader)
        test_metrics = learning.eval(loader=test_loader)
        print(f'>> Training Metrics : {train_metrics}.')
        print(f'>> Testing Metrics : {test_metrics}.')