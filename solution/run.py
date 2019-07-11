import argparse

from model import test_model, train_model, tune_model


def main():
    parser = argparse.ArgumentParser(
        description="A command line tool to manage the project")
    parser.add_argument('stage',
                        metavar='stage',
                        type=str,
                        choices=['tune', 'train', 'test'],
                        help="Stage to run.")

    stage = parser.parse_args().stage

    if stage == "tune":
        print("Tuning model...")
        tune_model()

    if stage == "train":
        train_model(print_params=False)
        print("Model was saved")

    elif stage == "test":
        test_model()


if __name__ == "__main__":
    main()
