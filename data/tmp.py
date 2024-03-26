from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class Base(ABC):
    """ DatasetFactory """
    a: str
    b: str
    
    def __post_init__(self) -> None:
        """ __post_init__ """
        self.config = {k: v for k, v in self.__dict__.items()}
        self.c = "3"
        print("super")
    
    @abstractmethod
    def xxx(self):
        ...
        


@dataclass
class DatasetFactory(Base):
    """ DatasetFactory """
    d: str
    e: str
    def __post_init__(self) -> None:
        super().__post_init__()
        self.f = "6"
        
    def do(self):
        self.f = "6"

@dataclass
class DDatasetFactory(DatasetFactory):
    """ DDatasetFactory """
    g: str
    h: str
    def __post_init__(self) -> None:
        super().__post_init__()
        self.i = "6"
    
    def xxx(self):
        print("xxxx")
        
    
data = DDatasetFactory(a="1", b="2", d="2", e="2", g="2", h="2")
data.do()
print(data.config)