#
# SonarQube Python Plugin
# Copyright (C) SonarSource Sàrl
# mailto:info AT sonarsource DOT com
#
# You can redistribute and/or modify this program under the terms of
# the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the Sonar Source-Available License for more details.
#
# You should have received a copy of the Sonar Source-Available License
# along with this program; if not, see https://sonarsource.com/license/ssal/
#

from collections.abc import Iterable
from enum import Enum

import mypy.nodes as mpn

import mypy.types as mpt

from serializer.proto_out import symbols_pb2


class BareTypeArgumentStatus(Enum):
    UNKNOWN = 0
    CONTAINS_ANY = 1
    NO_ANY = 2


class GenericInfo(Enum):
    """Valid generic states; argument information is only meaningful for a generic class."""
    UNKNOWN = 0
    NON_GENERIC = 1
    GENERIC_UNKNOWN = 2
    GENERIC_CONTAINS_ANY = 3
    GENERIC_NO_ANY = 4

    @property
    def is_generic(self) -> bool:
        return self in (self.GENERIC_UNKNOWN, self.GENERIC_CONTAINS_ANY, self.GENERIC_NO_ANY)

    def to_proto(self) -> symbols_pb2.GenericInfo:
        proto = symbols_pb2.GenericInfo()
        if self == self.NON_GENERIC:
            proto.non_generic.SetInParent()
        elif self.is_generic:
            proto.generic = {
                self.GENERIC_UNKNOWN: symbols_pb2.GenericInfo.UNKNOWN,
                self.GENERIC_CONTAINS_ANY: symbols_pb2.GenericInfo.CONTAINS_ANY,
                self.GENERIC_NO_ANY: symbols_pb2.GenericInfo.NO_ANY,
            }[self]
        return proto

    @classmethod
    def from_type_info(cls, type_info: mpn.TypeInfo) -> "GenericInfo":
        if not type_info.is_generic():
            return cls.NON_GENERIC
        status = combine_statuses(
            type_argument_status(parameter.default) if parameter.has_default() else BareTypeArgumentStatus.CONTAINS_ANY
            for parameter in type_info.defn.type_vars
        )
        return {
            BareTypeArgumentStatus.UNKNOWN: cls.GENERIC_UNKNOWN,
            BareTypeArgumentStatus.CONTAINS_ANY: cls.GENERIC_CONTAINS_ANY,
            BareTypeArgumentStatus.NO_ANY: cls.GENERIC_NO_ANY,
        }[status]


def combine_statuses(statuses: Iterable[BareTypeArgumentStatus]) -> BareTypeArgumentStatus:
    """Combine components of one type: a known Any dominates unresolved components."""
    result = BareTypeArgumentStatus.NO_ANY
    for status in statuses:
        if status == BareTypeArgumentStatus.CONTAINS_ANY:
            return status
        if status == BareTypeArgumentStatus.UNKNOWN:
            result = status
    return result


def type_argument_status(type_: mpt.Type, visited_aliases: frozenset[mpn.TypeAlias] = frozenset()) -> BareTypeArgumentStatus:
    """Classify nested Any without treating unsupported type forms as negative facts."""
    if isinstance(type_, mpt.TypeAliasType):
        if type_.alias is None:
            return BareTypeArgumentStatus.UNKNOWN
        arguments = combine_statuses(type_argument_status(arg, visited_aliases) for arg in type_.args)
        if type_.alias in visited_aliases or arguments == BareTypeArgumentStatus.CONTAINS_ANY:
            return arguments
        # A recursive back edge adds no new evidence; the other branches still need inspection.
        return combine_statuses((arguments, type_argument_status(
            mpt.get_proper_type(type_), visited_aliases | {type_.alias})))
    type_ = mpt.get_proper_type(type_)
    if isinstance(type_, mpt.AnyType):
        source = type_.source_any or type_
        if source.type_of_any in (mpt.TypeOfAny.from_unimported_type, mpt.TypeOfAny.from_error):
            return BareTypeArgumentStatus.UNKNOWN
        return BareTypeArgumentStatus.CONTAINS_ANY
    if isinstance(type_, (mpt.NoneType, mpt.LiteralType)):
        return BareTypeArgumentStatus.NO_ANY
    if isinstance(type_, mpt.Instance):
        return combine_statuses(type_argument_status(arg, visited_aliases) for arg in type_.args)
    if isinstance(type_, mpt.TypeType):
        return type_argument_status(type_.item, visited_aliases)
    if isinstance(type_, mpt.CallableType):
        return combine_statuses(type_argument_status(arg, visited_aliases) for arg in [type_.ret_type, *type_.arg_types])
    if isinstance(type_, (mpt.UnionType, mpt.TupleType)):
        return combine_statuses(type_argument_status(item, visited_aliases) for item in type_.items)
    # Type-variable substitution and other unsupported constructs require richer parameter metadata.
    return BareTypeArgumentStatus.UNKNOWN
