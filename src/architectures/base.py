

class BaseArchitecture():
    def __init__(self, name: str, id: int, description: str, layers: list[dict]) -> None:
        self.name = name
        self.id = id
        self.description = description
        self.layers = layers