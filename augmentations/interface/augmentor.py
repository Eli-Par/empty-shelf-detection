from abc import ABC, abstractmethod

class Augmentor(ABC):
    
    @abstractmethod
    def augment(self, images):
        pass