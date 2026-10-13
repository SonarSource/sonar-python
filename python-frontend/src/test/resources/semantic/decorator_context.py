from typing import final

mark = final

@mark
class TopLevel:
    pass


def lambda_handler(event, context):
    local_mark = final
    @local_mark
    class Nested:
        pass
    return Nested()
