from typing import final


@final
class C:
    pass


# The project decorator installs both methods; its runtime effects remain unknown.
assert C[0] == 42
assert C() + C() == 42
