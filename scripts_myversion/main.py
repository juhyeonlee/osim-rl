import argparse

from train import train

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train', help='start train', dest='train', action='store_true', default=True)
    parser.add_argument('--resume', help='resume training', type=str, default=None)
    parser.add_argument('--save_interval', help='save interval', type=int, default=10000)
    parser.add_argument('--test', help='start test', dest='test', action='store_true', default=False)
    parser.add_argument('--visualize', help='visualization', dest='visualize', action='store_true', default=False)
    parser.add_argument('--num_eps', help='the number of episodes', type=int, default=10000)
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    args = parser.parse_args()
    if args.train:
        train(args.num_eps, args.resume, args.save_interval, args.seed, args.visualize)

    # elif args.test:
    #     test()


if __name__ == '__main__':
    main()