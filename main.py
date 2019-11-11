import config
import classifier
from logger import logger

data = config.get_training_data()
Classifier = classifier.NaiveBayes(data)

logger.debug(Classifier.data)


result = Classifier.predict(config.get_training_data("input.txt")[1])
logger.debug("prediction is %s", result)

f = open('output.txt', 'w')
f.write(result)
f.close()
# Classifier.