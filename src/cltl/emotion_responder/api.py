import abc


class EmotionResponder(abc.ABC):
    def respond(self, emotion: []) -> str:
        raise NotImplementedError("")
