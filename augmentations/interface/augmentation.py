from abc import ABC, abstractmethod

class Augmentation(ABC):
    
    @abstractmethod
    def augment(self, image):
        pass