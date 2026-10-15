class TopLevel:
    pass


def lambda_handler(event, context):
    class Nested:
        pass
    return Nested()
