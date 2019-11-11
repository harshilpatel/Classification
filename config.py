import constants
from logger import logger

def get_training_data(file_to_read = constants.FILE):
    raw_data = open(file_to_read).read().strip().split("\n")
    data = []

    while raw_data:
        # new_line = raw_data.pop(0).split(constants.SEPERATOR)
        new_line = raw_data.pop(0).split()
        data.append(new_line)
    
    logger.debug(data)
    return data

