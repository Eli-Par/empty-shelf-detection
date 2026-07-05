from abc import ABC, abstractmethod

class Train(ABC):

    @abstractmethod
    def train(self):
        pass

    def setProject(self, project):
        self.project = project