/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python.semantic.v2.types;

import java.util.List;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.symbols.v2.SymbolV2;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.types.v2.ClassType;
import org.sonar.plugins.python.api.types.v2.ObjectType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.TypeSource;
import org.sonar.plugins.python.api.types.v2.UnionType;
import org.sonar.python.types.v2.matchers.TypePredicateContext;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.sonar.python.types.v2.TypesTestUtils.INT_TYPE;
import static org.sonar.python.types.v2.TypesTestUtils.LIST_TYPE;
import static org.sonar.python.types.v2.TypesTestUtils.PROJECT_LEVEL_TYPE_TABLE;
import static org.sonar.python.types.v2.TypesTestUtils.STR_TYPE;
import static org.sonar.python.types.v2.TypesTestUtils.lastName;

class LoopAssignmentTest {
  @Test
  void tuple_iteration_includes_all_positional_element_types() {
    for (var annotation : List.of("tuple[int, str]", "tuple[int, str] | tuple[str, int]", "Tuple[int, str]")) {
      var result = lastName("""
        from typing import Tuple
        def inspect(values: %s):
          for element in values:
            element
        """.formatted(annotation)).typeV2();
      assertThat(result).isInstanceOfSatisfying(UnionType.class, union -> {
        assertThat(union.candidates()).allSatisfy(candidate -> {
          assertThat(candidate).isInstanceOf(ObjectType.class);
          assertThat(candidate.typeSource()).isEqualTo(TypeSource.TYPE_HINT);
        }).extracting(PythonType::unwrappedType).containsExactlyInAnyOrder(INT_TYPE, STR_TYPE);
      });
    }
  }

  @Test
  void dictionary_iteration_keeps_only_the_key_type() {
    var result = lastName("""
      def inspect(values: dict[int, str]):
        for element in values:
          element
      """).typeV2();
    assertThat(result.unwrappedType()).isEqualTo(INT_TYPE);
    assertThat(result.typeSource()).isEqualTo(TypeSource.TYPE_HINT);
  }

  @Test
  void homogeneous_tuple_iteration_preserves_the_element_type() {
    for (var annotation : List.of("tuple[int, ...]", "Tuple[int, ...]", "tuple[int]")) {
      var result = lastName("""
        from typing import Tuple
        def inspect(values: %s):
          for element in values:
            element
        """.formatted(annotation)).typeV2();
      assertThat(result.unwrappedType()).isEqualTo(INT_TYPE);
      assertThat(result.typeSource()).isEqualTo(TypeSource.TYPE_HINT);
    }
  }

  @Test
  void source_union_of_iterables_preserves_values_and_type_hint_provenance() {
    var result = lastName("""
      def inspect(values: list[int] | set[str]):
        for element in values:
          element
      """).typeV2();
    assertThat(result).isInstanceOfSatisfying(UnionType.class, union -> {
      assertThat(union.candidates()).allSatisfy(candidate -> {
        assertThat(candidate).isInstanceOf(ObjectType.class);
        assertThat(candidate.typeSource()).isEqualTo(TypeSource.TYPE_HINT);
      }).extracting(PythonType::unwrappedType).containsExactlyInAnyOrder(INT_TYPE, STR_TYPE);
    });
  }

  @Test
  void optional_iterable_keeps_the_element_type_unknown() {
    var result = lastName("""
      def inspect(values: list[int] | None):
        for element in values:
          element
      """).typeV2();
    assertThat(result).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void unions_of_iterables_preserve_each_element_type() {
    var integer = ObjectType.fromType(new ClassType("int", "builtins.int"));
    var string = ObjectType.fromType(new ClassType("str", "builtins.str"));
    var iterable = LIST_TYPE;
    var first = ObjectType.Builder.fromType(iterable).withAttributes(List.of(integer)).build();
    var second = ObjectType.Builder.fromType(iterable).withAttributes(List.of(string)).build();
    assertThat(elementType(UnionType.or(first, second))).isEqualTo(UnionType.or(integer, string));
  }

  @Test
  void non_iterable_or_unknown_candidate_keeps_the_result_unknown() {
    var integer = ObjectType.fromType(new ClassType("int", "builtins.int"));
    var iterable = LIST_TYPE;
    var values = ObjectType.Builder.fromType(iterable).withAttributes(List.of(integer)).build();
    assertThat(elementType(UnionType.or(values, integer))).isEqualTo(PythonType.UNKNOWN);
    assertThat(elementType(UnionType.or(values, PythonType.UNKNOWN))).isEqualTo(PythonType.UNKNOWN);
  }

  @Test
  void literal_collections_preserve_class_values() {
    for (var collection : List.of("[int]", "(int,)", "[int, str]", "[int] if flag else [str]", "(int,) if flag else (str,)")) {
      var result = lastName("""
        def inspect(flag):
          for element in %s:
            element
        """.formatted(collection)).typeV2();
      if (collection.contains("str")) {
        assertThat(result).isInstanceOfSatisfying(UnionType.class,
          union -> assertThat(union.candidates()).containsExactlyInAnyOrder(INT_TYPE, STR_TYPE));
      } else {
        assertThat(result).isEqualTo(INT_TYPE);
      }
    }
  }

  @Test
  void custom_generic_iterators_do_not_assume_the_first_argument_is_the_element_type() {
    for (var base : List.of("Generic[T]", "list[T]")) {
      var result = lastName("""
        from typing import Callable, Generic, Iterator, TypeVar
        T = TypeVar("T")
        class Callbacks(%s):
          def __iter__(self) -> Iterator[Callable]:
            yield lambda: None
        def inspect(values: Callbacks[int] | Callbacks[str]):
          for element in values:
            element
        """.formatted(base)).typeV2();
      assertThat(result).isEqualTo(PythonType.UNKNOWN);
    }
  }

  @Test
  void known_iterable_annotations_preserve_element_values() {
    for (var annotation : List.of("list[int]", "set[int]", "frozenset[int]", "deque[int]",
      "Iterable[int]", "Iterator[int]", "Generator[int, None, None]", "Sequence[int]", "Mapping[int, str]",
      "ChainMap[int, str]", "Reversible[int]", "AbcReversible[int]", "filter[int]", "map[int]", "reversed[int]")) {
      var result = lastName("""
        from collections import ChainMap, deque
        from collections.abc import Reversible as AbcReversible
        from typing import Iterable, Iterator, Generator, Sequence, Mapping, Reversible
        def inspect(values: %s):
          for element in values:
            element
        """.formatted(annotation)).typeV2();
      assertThat(result).as(annotation).isInstanceOf(ObjectType.class);
      assertThat(result.unwrappedType()).as(annotation).isEqualTo(INT_TYPE);
      assertThat(result.typeSource()).as(annotation).isEqualTo(TypeSource.TYPE_HINT);
    }
  }

  @Test
  void text_stream_iteration_preserves_line_values() {
    var result = lastName("""
      from typing import IO
      def inspect(stream: IO[str]):
        for line in stream:
          line
      """).typeV2();
    assertThat(result).isInstanceOf(ObjectType.class);
    assertThat(result.unwrappedType()).isEqualTo(STR_TYPE);
    assertThat(result.typeSource()).isEqualTo(TypeSource.TYPE_HINT);
  }

  @Test
  void unlisted_standard_iterables_intentionally_keep_element_types_unknown() {
    // Accepted precision tradeoff: only explicitly supported types use their generic arguments for iteration.
    // These annotations previously benefited from assuming every iterable's first argument was its element type.
    // Keep UNKNOWN for unlisted types rather than generalizing that assumption to arbitrary iterators.
    for (var annotation : List.of("chain[int]", "islice[int]", "repeat[int]", "WeakValueDictionary[str, object]", "list[int] | chain[int]")) {
      var result = lastName("""
        from __future__ import annotations
        from itertools import chain, islice, repeat
        from weakref import WeakValueDictionary
        def inspect(values: %s):
          for element in values:
            element
        """.formatted(annotation)).typeV2();
      assertThat(result).as(annotation).isEqualTo(PythonType.UNKNOWN);
    }
  }

  private static PythonType elementType(PythonType rhsType) {
    var expression = mock(Expression.class);
    when(expression.typeV2()).thenReturn(rhsType);
    return new LoopAssignment(mock(SymbolV2.class), mock(Name.class), expression, TypePredicateContext.of(PROJECT_LEVEL_TYPE_TABLE)).rhsType();
  }
}
