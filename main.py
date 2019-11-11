__author__ = "Harshilkumar Patel"
__status__ = "Development"

import config
import classifier
from logger import logger
import constants

data = config.get_training_data()

if constants.CLASSIFIER_CHOICE == "knn":
    Classifier = classifier.KNN(data)
else:
    Classifier = classifier.NaiveBayes(data)

logger.debug("formatted data is %s", Classifier.data)

result = Classifier.predict(config.get_training_data("input.txt")[1])
logger.debug("THE FINAL PREDICTION is %s", result)

# f = open('output.txt', 'w')
# f.write(result)
# f.close()