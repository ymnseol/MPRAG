import argparse
from scripts.main import main


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MPRAG')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('-d', '--data_name', type=str, default='squad', help='Dataset name')
    parser.add_argument('-e', '--num_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('-sch', '--scheduler', type=str, default='StepLR', help='Scheduler')
    parser.add_argument('-opt', '--optimizer', type=str, default='Adam', help='Optimizer')
    parser.add_argument('-g', '--gamma', type=float, default=0.8, help='Gamma')
    parser.add_argument('-s', '--step_size', type=int, default=5, help='Step size')
    parser.add_argument('-p', '--pretrained', type=str, default='classifier_weight', help='Pretrained model')
    parser.add_argument('-idx', '--index_path', type=str, default='squad.index', help='Index path')
    args = parser.parse_args()

    main(args)
