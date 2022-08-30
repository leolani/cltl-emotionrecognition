import abc


class EmotionResponder(abc.ABC):
    def respond(self, statement: str) -> str:
        raise NotImplementedError("")
