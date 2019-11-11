__author__ = "Harshilkumar Patel"
__status__ = "Development"

import constants
from utils import logger

def get_training_data(file_to_read = constants.FILE):
    raw_data = open(file_to_read).read().strip().split("\n")
    data = []

    while raw_data:
        # new_line = raw_data.pop(0).split(constants.SEPERATOR)
        new_line = raw_data.pop(0).split()

        for i, value in enumerate(new_line):
            try:
                new_line[i] = float(value)
            except Exception as e:
                pass

        data.append(new_line)
    
    logger.debug(data)
    return data

