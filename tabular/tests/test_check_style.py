import logging
import warnings
from subprocess import PIPE, Popen
from datetime import datetime

start_time = datetime.now()


def test_check_style():
    logging.getLogger().setLevel(logging.INFO)
    logging.info("PEP8 Style check")
    flake8_proc = Popen(["flake8", "--count", "--max-line-length", "300"], stdout=PIPE)
    flake8_out = flake8_proc.communicate()[0]
    lines = flake8_out.splitlines()
    count = int(lines[-1].decode())
    if count > 0:
        warnings.warn(f"{count} PEP8 warnings remaining")
    assert count < 1000, "Too many PEP8 warnings found, improve code quality to pass test."


end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))