import argparse
import configparser
import gym


def read_config(file='gym.ini'):
    parser = configparser.ConfigParser()
    parser.read(file)

    return parser


def read_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='path to gym results')

    return parser.parse_args()


def upload(gym_path, key):
    return gym.upload(gym_path, api_key=key)


if __name__ == '__main__':
    config = read_config()
    args = read_argparse()
    key = config['default']['GYM_API_KEY']

    if len(key) == 0 or key is None or key == "YOUR_API_KEY":
        print("Please enter the API key in gym.ini ")

    else:
        upload(args.path, key)
