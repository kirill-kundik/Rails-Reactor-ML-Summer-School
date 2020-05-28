import argparse
import functools
import logging
import multiprocessing
import pathlib
import time
import collections

import requests

PROJECT_ROOT = pathlib.Path(__file__).parent
SPECIAL_CHARS = {
    ' ': '<space>',
    '\n': '<newline>'
}


class StatusCodeWarning(Warning):
    pass


def start(base_url: str, num_processes: int):
    """
    parse base file and start processes for serving all files in base file, merge results from processes and write it
    :param base_url: base dataset folder url
    :param num_processes: number of process that will start in pool
    :return: None
    """
    dataset_files = [f'{base_url}/{x}' for x in read_file(base_url + '/files.txt').decode(encoding='utf-8').split('\n')
                     if x.endswith('.txt')]

    if not (PROJECT_ROOT / 'dataset').exists():  # check if dataset folder is exists
        (PROJECT_ROOT / 'dataset').mkdir(parents=True, exist_ok=True)
        logging.info('Dataset folder created')

    with multiprocessing.Pool(num_processes) as pool:
        logging.info('Starting processes')
        res = functools.reduce((lambda x, y: x + y), pool.map(process_url, dataset_files))
        with open(PROJECT_ROOT / 'result.txt', 'wb') as f:
            logging.info('Writing to result.txt file')
            for k, v in SPECIAL_CHARS.items():
                if k in res:
                    f.write(f'{v} {res[k]}\n'.encode(encoding='utf-8'))
                    res.pop(k)

            for k, v in res.items():
                f.write(f'{repr(k)[1:-1]} {v}\n'.encode(encoding='utf-8'))


def process_url(file_url: str) -> collections.Counter:
    """
    function for processing pool, it will download file and write it in directory and parse with counter
    :param file_url: file url for downloading
    :return: counter for current file in url
    """
    file_name = pathlib.Path(file_url).name
    if not (PROJECT_ROOT / 'dataset' / file_name).exists():  # check if file was already exists
        logging.info(msg=f'Getting file {file_url}')
        try:
            file_content = read_file(file_url)
            with open(PROJECT_ROOT / 'dataset' / file_name, "wb") as f:
                f.write(file_content)
        except StatusCodeWarning:
            logging.error(f'Downloading file error, url: {file_url}')
            return collections.Counter()
    else:
        logging.info(msg=f'Using existing file from dataset for /dataset/{file_name}')
        with open(PROJECT_ROOT / 'dataset' / file_name, "rb") as f:
            file_content = f.read()
    return parsing_file(file_content.decode(encoding='utf-8'))


def parse_args() -> (str, int):
    """
    Program argument parsing
    :return: url for the dataset and number of processes that will be started
    """
    parser = argparse.ArgumentParser(
        description='Second homework. Implement the character counter for the provided dataset. ')
    parser.add_argument('--url',
                        help='Link to the files.txt where dataset can be found, default: '
                             'http://ps2.railsreactor.net/datasets/wikipedia_articles',
                        type=str,
                        default='http://ps2.railsreactor.net/datasets/wikipedia_articles')
    parser.add_argument('--num_processes',
                        help='Number of processes which will started by this program, default: '
                             'num of cpu avaliable cores',
                        type=int,
                        default=multiprocessing.cpu_count())
    args = vars(parser.parse_args())

    return args['url'], args['num_processes']


def read_file(file_url: str) -> bytes:
    """
    Helper function for reading file by url
    :param file_url: url of the file
    :return: file content
    """
    r = requests.get(url=file_url)
    if r.status_code == 200:
        return r.content
    else:
        raise StatusCodeWarning('Status code of request is not 200 OK.')


def parsing_file(content: str) -> collections.Counter:
    """
    parse the content of the text file to the counter
    :param content: raw content of the file
    :return: counter for the current file
    """
    return collections.Counter(content)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    start_time = time.time()
    url, num = parse_args()
    start(url, num)
    logging.info(msg=f'Time for program execution: {format(time.time() - start_time, ".4f")} in seconds')
