from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
import os
import logging

logger = logging.getLogger(__name__)


def __download_and_unzip(url, extract_to):
    http_response = urlopen(url)
    zipfile = ZipFile(BytesIO(http_response.read()))
    zipfile.extractall(path=extract_to)

def __contains_subdirectory(directory):
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            return True
    return False

# https://github.com/LIAAD/KeywordExtractor-Datasets/tree/master
GITHUB_BASE_URL = "https://github.com/LIAAD/KeywordExtractor-Datasets/raw/8cd1e18ab750143e2075457ef5e0481754e3e966/datasets/"
SemEval2010_URL = GITHUB_BASE_URL + "SemEval2010.zip"
SemEval2017_URL = GITHUB_BASE_URL + "SemEval2017.zip"

TARGET_FOLDER = "data_cs"

def download_data():
    # if TARGET_FOLDER does contain any directories then return
    if __contains_subdirectory(TARGET_FOLDER):
        logger.info('Download skipped, data already exists in the target folder.')
        return
    logger.info('Downloading and unzipping data...')
    __download_and_unzip(SemEval2010_URL, TARGET_FOLDER)
    __download_and_unzip(SemEval2017_URL, TARGET_FOLDER)
