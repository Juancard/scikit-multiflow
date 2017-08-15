__author__ = 'Guilherme Matsumoto'

from skmultiflow.data.generators.random_rbf_generator import RandomRBFGenerator
from skmultiflow.core.base_object import BaseObject
import numpy as np


class RandomRBFGeneratorDrift(RandomRBFGenerator, BaseObject):
    """ RandomRBFGeneratorDrift
    
    This class is an extension from the RandomRBFGenerator. It functions 
    as the parent class, except that drift can be introduced in objects 
    of this class. 
    
    The drift is created by adding a speed to certain centroids. As the 
    samples are generated each of the moving centroids' centers is 
    changed by an amount determined by its speed.
    
    Parameters
    ----------
    model_seed: int (Default: 21)
        The seed to be used by the model random generator.
        
    instance_seed: int (Default: 5)
        The seed to be used by the instance random generator.
        
    num_classes: int (Default: 2)
        The number of class labels to generate.
        
    num_att: int (Default: 10)
        The total number of attributes to generate.
        
    num_centroids: int (Default: 50)
        The total number of centroids to generate.
        
    change_speed: float (Default: 0.0)
        The concept drift speed.
        
    num_drift_centroids: int (Default: 50)
        The number of centroids that will drift.
        
    Examples
    --------
    >>> # Imports
    >>> from skmultiflow.data.generators.random_rbf_generator_drift import RandomRBFGeneratorDrift
    >>> # Setting up the stream
    >>> stream = RandomRBFGeneratorDrift(model_seed=99, instance_seed = 50, num_classes = 4, num_att = 10, 
    ... num_centroids = 50, change_speed=0.87, num_drift_centroids=50)
    >>> stream.prepare_for_use()
    >>> # Retrieving one sample
    >>> stream.next_instance()
    (array([[ 0.87640769,  1.11561069,  0.61592869,  1.0580048 ,  0.34237265,
         0.44265564,  0.8714499 ,  0.47178835,  1.07098717,  0.29090414]]), array([ 3.]))
    >>> # Retrieving 10 samples
    >>> stream.next_instance(10)
    (array([[ 0.78413886,  0.98797944,  0.26981191,  0.92217135,  0.61152321,
         1.02183543,  0.99855968,  0.71545227,  0.55584282,  0.32919095],
       [ 0.45714164,  0.2610933 ,  0.07065982,  0.62751192,  0.75317802,
         0.95785718,  0.32732265,  1.03553576,  0.58009199,  0.90331289],
       [ 0.04165148,  0.38215897, -0.0173352 ,  0.64773072,  0.50398859,
         1.00646399, -0.03972425,  0.62976581,  0.70082235,  0.90992945],
       [ 0.37416657,  0.45838559,  0.82463152,  0.17117448,  0.97320165,
         0.73638815,  0.80587782,  0.75280346,  0.40483112,  1.0012537 ],
       [ 0.79264171,  0.13507299,  0.79600514,  0.33743781,  0.67766074,
         0.70102531, -0.02483112,  0.1921961 ,  0.46693386, -0.02937016],
       [ 0.5129367 ,  0.42697567,  0.25741495,  0.68854096,  0.1119384 ,
         0.76748539,  0.91141342,  0.51498633,  0.17019881,  0.51172656],
       [-0.07820356,  1.19744888,  0.82647513,  1.08993095,  0.67718824,
         0.66486463,  0.52000702,  0.68708254,  0.21171053,  0.81696899],
       [ 0.57232341,  1.13725733,  0.97343092,  1.11889521,  0.68894022,
         1.27717546, -0.1063654 , -0.36732086,  0.54799583,  0.48858978],
       [ 0.27969972, -0.06563579,  0.02834469,  0.05250523,  0.52713213,
         0.73472713,  0.15381198, -0.07735765,  0.9792027 ,  0.92673772],
       [ 0.52641196,  0.3009952 ,  0.56104759,  0.40478501,  0.63097374,
         0.3797032 , -0.00446842,  0.52913688,  0.24908855,  0.22779074]]), array([ 3.,  3.,  3.,  2.,  3.,  2.,  0.,  2.,  0.,  2.]))
    >>> # Generators will have infinite remaining instances, so it returns -1
    >>> stream.estimated_remaining_instances()
    -1
    >>> stream.has_more_instances()
    True
          
    """

    def __init__(self, model_seed=21, instance_seed=5, num_classes=2, num_att=10, num_centroids=50,
                 change_speed=0.0, num_drift_centroids=50):
        super().__init__(model_seed, instance_seed, num_classes, num_att, num_centroids)
        #default values
        self.change_speed = change_speed
        self.num_drift_centroids = num_drift_centroids
        self.centroid_speed = None

    def next_instance(self, batch_size=1):
        """ next_instance
        
        Return batch_size samples generated by choosing a centroid at 
        random and randomly offsetting its attributes so that it is 
        placed inside the hypersphere of that centroid.
        
        In addition to that, drift is introduced to a chosen number of 
        centroids. Each chosen center is moved at each generated sample.
        
        Parameters
        ----------
        batch_size: int
            The number of samples to return.
        
        Returns
        -------
        tuple or tuple list
            Return a tuple with the features matrix and the labels matrix for 
            the batch_size samples that were requested. 
        
        """
        data = np.zeros([batch_size, self.num_numerical_attributes + 1])
        for k in range(batch_size):
            len = self.num_drift_centroids
            if (len > self.num_centroids):
                len = self.num_centroids

            for i in range(len):
                for j in range(self.num_numerical_attributes):
                    self.centroids[i].centre[j] += self.centroid_speed[i][j] * self.change_speed

                    if ((self.centroids[i].centre[j] > 1) | (self.centroids[i].centre[j] < 0)):
                        self.centroids[i].centre[j] = 1 if (self.centroids[i].centre[j] > 1) else 0
                        self.centroid_speed[i][j] = -self.centroid_speed[i][j]
            X, y = super().next_instance(1)
            data[k, :] = np.concatenate((X[0], y[0]))

        return (data[:, :self.num_numerical_attributes], data[:, self.num_numerical_attributes:].flatten())

    def generate_centroids(self):
        """ generate_centroids
        
        The centroids are generated just as it's done in the parent class, 
        the difference is the extra step taken to setup the drift, if there's 
        any.
        
        To __configure the drift, random offset speeds are chosen for 
        self.num_drift_centroids centroids. Finally, the speed are 
        normalized.
        
        """
        super().generate_centroids()
        model_random = np.random
        model_random.seed(self.model_seed)
        len = self.num_drift_centroids
        self.centroid_speed = []
        if (len > self.num_centroids):
            len = self.num_centroids

        for i in range(len):
            rand_speed = []
            norm_speed = 0.0

            for j in range(self.num_numerical_attributes):
                rand_speed.append(model_random.rand())
                norm_speed += rand_speed[j]*rand_speed[j]

            norm_speed = np.sqrt(norm_speed)

            for j in range(self.num_numerical_attributes):
                rand_speed[j] /= norm_speed

            self.centroid_speed.append(rand_speed)

    def prepare_for_use(self):
        self.restart()

    def restart(self):
        self.generate_centroids()
        self.instance_random.seed(self.instance_seed)

    def get_info(self):
        return 'RandomRBFGenerator: model_seed: ' + str(self.model_seed) + \
               ' - instance_seed: ' + str(self.instance_seed) + \
               ' - num_classes: ' + str(self.num_classes) + \
               ' - num_att: ' + str(self.num_numerical_attributes) + \
               ' - num_centroids: ' + str(self.num_centroids) + \
               ' - change_speed: ' + str(self.change_speed) + \
               ' - num_drift_centroids: ' + str(self.num_drift_centroids)

    def get_num_targeting_tasks(self):
        return 1

if __name__ == '__main__':
    stream = RandomRBFGeneratorDrift(change_speed=0.02, num_drift_centroids=50)
    stream.prepare_for_use()

    X, y = stream.next_instance(4)
    print(X)
    print(y)