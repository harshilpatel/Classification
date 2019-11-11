__author__ = "Harshilkumar Patel"
__status__ = "Development"

from utils import logger
from pprint import pprint
import math

class NaiveBayes(object):
    def __init__(self, data):
        # 0 indexing
        self.data = data or []
        
        self.column_names = self.data[0]
        self.data = self.data[1:]
        self.data_size = len(self.data) * 1.0
        self.number_of_columns = len(self.data[0])

        self.result_labels = list(set([i[-1] for i in self.data]))
        
        self.column_value_count = [{} for i in range(self.number_of_columns)]
        self.is_continous = False

        if not self.is_continous:
            for i, row in enumerate(self.data):
                count = {}
                for j, column in enumerate(row):
                    self.column_value_count[j][column] = self.column_value_count[j].get(column, 0) + 1
                # self.column_value_count.append(count)

        logger.debug(self.column_value_count)        


    def get_distribution_for_value_in_column(self, column_index, value):
        # logger.debug("dist for value in column: %s", value)
        result = self.column_value_count[column_index].get(value)/self.data_size
        logger.debug("dist for %s is %s", value, result)
        return result
    
    def get_related_distribution_for_value_in_column(self, column1, value1, column2, value2):
        # A = value1
        # B = value2

        ab_count = 0

        for i, row in enumerate(self.data):
            if row[column1] == value1 and row[column2] == value2:
                ab_count += 1
        
        b_count = self.column_value_count[column2].get(value2, 1) * 1.0
        result = ab_count/b_count
        logger.debug("related dist for %s | %s is %s", value1, value2, result)
        return result
    
    def p_value_for_column_value_and_result_label(self, column1, value1, column2, value2):
        a = self.get_related_distribution_for_value_in_column(column2, value2, column1, value1)
        b = self.get_distribution_for_value_in_column(column1, value1)
        c = self.get_distribution_for_value_in_column(column2, value2) * 1.0
        return ( a * b)/c

    def predict(self, train_set):
        # main method to get dist
        n = self.number_of_columns - 1

        assert len(train_set) == self.number_of_columns - 1
        p_values = {i:1 for i in self.result_labels}
        
        for label in p_values.keys():
            for i, train_label in enumerate(train_set):
                p_values[label] *= self.p_value_for_column_value_and_result_label(i, train_label, n, label)

        
        logger.debug("distribution: %s", p_values)

        max_p = 0
        result = ""
        for key, value in p_values.items():
            if value > max_p:
                max_p = value
                result = key
        return result


class KNN(object):
    def __init__(self, data, k = 3):
        self.is_continous = True
        self.data = data
        self.k = k

        self.column_names = self.data[0]
        self.data = self.data[1:]
        
        self.data_size = len(self.data) * 1.0
        self.number_of_columns = len(self.data[0])

        # pprint(self.data)


    
    def get_hamming_distance(self, row_index, data_instance):
        result = 0
        for i, data_instance in enumerate(self.data[row_index][:-1]):
            # for j, column_data in enumerate(data_instance):
            # print(train_instance[i])
            # print(data_instance)
            result += 0 if train_instance[i] == data_instance else 1
    
    def get_euclidean_distance(self, row_index, train_instance):
        result = 0
        for i, data_instance in enumerate(self.data[row_index][:-1]):
            # for j, column_data in enumerate(data_instance):
            # print(train_instance[i])
            # print(data_instance)
            result += (train_instance[i] - data_instance)**2
        
        result = math.sqrt(result)
        logger.debug("distance for %s and %s is %s", self.data[row_index], train_instance, result)

        return result

    def get_distance(self, row_index, data_instance):
        if self.is_continous:
            return self.get_euclidean_distance(row_index, data_instance)
        else:
            return self.get_hamming_distance(row_index, data_instance)
    

    def predict(self, train_instance):
        distances = []

        for i, data_instance in enumerate(self.data):
            distances.append( self.get_distance(i, train_instance) )
        
        smallest_distances = sorted(distances)[:self.k]
        distances = [i if i in smallest_distances else 0 for i in distances]

        prediction = {}
        for i, row in enumerate(self.data):
            if distances[i]:
                label = row[-1]
                prediction[label] = prediction.get(label, 0) + 1
        

        max_p = 0
        result = ""
        for key, value in prediction.items():
            if value > max_p:
                max_p = value
                result = key
        return result
        


