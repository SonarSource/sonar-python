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


class ThreeParameterContract:
    def run(self, input_ids, remove_from_pkv=0, num_added_tokens=1): ...


class MissingMiddleParameter(ThreeParameterContract):
    def run(self, input_ids, num_added_tokens=1): ...  # Noncompliant {{Add missing parameters remove_from_pkv.}}


class MissingFirstParameter(ThreeParameterContract):
    def run(self, remove_from_pkv=0, num_added_tokens=1): ...  # Noncompliant {{Add missing parameters input_ids.}}


class MissingMiddleWithRename(ThreeParameterContract):
    def run(self, assistant_input_ids, num_added_tokens=1): ...  # Noncompliant {{Add missing parameters remove_from_pkv.}}


class MissingTwoParameters(ThreeParameterContract):
    def run(self, num_added_tokens=1): ...  # Noncompliant {{Add missing parameters input_ids remove_from_pkv.}}


class MissingMiddleWithOptionalKeyword(ThreeParameterContract):
    def run(self, assistant_input_ids, num_added_tokens=1, *, extra=None): ...  # Noncompliant {{Add missing parameters remove_from_pkv.}}


class MissingMiddleWithAllRenames(ThreeParameterContract):
    def run(self, first, second): ...  # Noncompliant {{Add missing parameters num_added_tokens.}}


class ThreePositionalOnlyContract:
    def run(self, first, second, third, /): ...


class MissingPositionalOnlyParameters(ThreePositionalOnlyContract):
    def run(self, third, /): ...  # Noncompliant {{Add missing parameters second third.}}


class PositionalOnlyThenNamedContract:
    def run(self, first, /, second=None, third=None): ...


class MissingAfterPositionalOnly(PositionalOnlyThenNamedContract):
    def run(self, second, /, third=None): ...  # Noncompliant {{Add missing parameters second.}}
