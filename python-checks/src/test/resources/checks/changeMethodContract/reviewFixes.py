class KeywordContract:
    def run(self, value=0, *, keep="first"): ...


class OptionalKeywordAfter(KeywordContract):
    def run(self, value=0, *, keep="first", extra=False): ...


class OptionalKeywordBefore(KeywordContract):
    def run(self, value=0, *, extra=False, keep="first"): ...


class RequiredKeywordAddition(KeywordContract):
    def run(self, value=0, *, keep="first", required): ...  # Noncompliant {{Change this method signature to accept the same arguments as the method it overrides.}}
#       ^^^


class MissingKeyword(KeywordContract):
    def run(self, value=0, *, other=False): ...  # Noncompliant {{Change this method signature to accept the same arguments as the method it overrides.}}
#       ^^^


class RequiredPositionalAfterOptionalKeywordAddition(KeywordContract):
    def run(self, value, *, keep="first", extra=False): ...  # Noncompliant {{Add a default value to parameter value.}}
#                 ^^^^^


class RequiredKeywordAfterOptionalKeywordAddition(KeywordContract):
    def run(self, value=0, *, keep, extra=False): ...  # Noncompliant {{Add a default value to parameter keep.}}
#                             ^^^^


class NarrowedModeAfterOptionalKeywordAddition(KeywordContract):
    def run(self, *, value=0, keep="first", extra=False): ...  # Noncompliant {{Make parameter value keyword-or-positional.}}
#                    ^^^^^^^


class PositionalContract:
    def run(self, value): ...


class RenamedAndExtended(PositionalContract):
    def run(self, renamed, extra): ...  # Noncompliant {{Remove parameter extra or provide default value.}}
#                          ^^^^^


class RenamedAndExtendedTwice(PositionalContract):
    def run(self, renamed, extra, another): ...  # Noncompliant {{Remove parameter extra or provide default value.}} {{Remove parameter another or provide default value.}}


class InsertedBeforeNamedParameter(PositionalContract):
    def run(self, extra, value): ...  # Noncompliant {{Remove parameter extra or provide default value.}}
#                 ^^^^^


class OptionalPositionalAddition(PositionalContract):
    def run(self, renamed, extra=None): ...


class OptionalKeywordAddition(PositionalContract):
    def run(self, value, *, extra=None): ...


class MissingPositionalWithOptionalKeywordAddition(PositionalContract):
    def run(self, *, extra=None): ...  # Noncompliant {{Add missing parameters value.}}
#       ^^^


class RequiredKeywordAfterRenamedOptionalPositional(PositionalContract):
    def run(self, renamed=None, *, required): ...  # Noncompliant {{Remove parameter required or provide default value.}}
#                                  ^^^^^^^^


class TwoPositionalContract:
    def run(self, first, second): ...


class TwoRenamedParameters(TwoPositionalContract):
    def run(self, renamed_first, renamed_second, extra): ...  # Noncompliant {{Remove parameter extra or provide default value.}}
#                                                ^^^^^


class PartlyRenamedParameters(TwoPositionalContract):
    def run(self, renamed_first, second, extra): ...  # Noncompliant {{Remove parameter extra or provide default value.}}
#                                        ^^^^^


class PositionalOnlyContract:
    def run(self, value, /): ...


class RenamedPositionalOnly(PositionalOnlyContract):
    def run(self, renamed, extra, /): ...  # Noncompliant {{Remove parameter extra or provide default value.}}
#                          ^^^^^
