import os

_PROJECT_ROOT = os.getcwd()  # assume we always run pytest from the project root
_TEST_ROOT = os.path.dirname(__file__)  # root of test folder
_PATH_DATA = os.path.join(_PROJECT_ROOT, "data")  # root of data
_PATH_PROCESSED = os.path.join(_PATH_DATA, "processed")  # root of processed data
_PATH_RAW = os.path.join(_PATH_DATA, "raw")  # root of raw data

# run pytest with -s flag to see the output
print(_TEST_ROOT)
print(_PROJECT_ROOT)
print(_PATH_DATA)
print(_PATH_PROCESSED)
print(_PATH_RAW)
