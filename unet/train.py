from train_lib.cli import build_arg_parser, load_config, validate_args
from train_lib.runner import run_training


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    args = load_config(args.config)
    validate_args(args)
    run_training(args)


if __name__ == '__main__':
    main()
