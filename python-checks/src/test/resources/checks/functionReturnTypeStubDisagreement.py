from logging import Logger, getLogger


class CustomLogger(Logger):
    def custom_method(self):
        return 42


def factory() -> CustomLogger:
    return getLogger("name")
