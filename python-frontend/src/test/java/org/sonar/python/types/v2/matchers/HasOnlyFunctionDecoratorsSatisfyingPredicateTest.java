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
package org.sonar.python.types.v2.matchers;

import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.types.v2.FunctionType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.TypeOrigin;
import org.sonar.plugins.python.api.types.v2.TypeWrapper;
import org.sonar.plugins.python.api.types.v2.UnionType;
import org.sonar.plugins.python.api.types.v2.matchers.MatchersTestUtils;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.v2.FunctionTypeBuilder;
import org.sonar.python.semantic.v2.typetable.ProjectLevelTypeTable;
import org.sonar.python.semantic.v2.typetable.TypeTable;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class HasOnlyFunctionDecoratorsSatisfyingPredicateTest {
  private final ProjectLevelTypeTable table = new ProjectLevelTypeTable(ProjectLevelSymbolTable.empty());
  private final TypePredicateContext context = TypePredicateContext.of(table);
  private final TypePredicate predicate = new HasOnlyFunctionDecoratorsSatisfyingPredicate(new IsTypePredicate("typing.overload"));
  private final PythonType overload = table.getType("typing.overload");

  @Test
  void matchesCanonicalAndReexportedOverload() {
    assertThat(predicate.check(function(overload).build(), context)).isEqualTo(TriBool.TRUE);
    assertThat(predicate.check(function(table.getType("typing_extensions.overload")).build(), context)).isEqualTo(TriBool.TRUE);
  }

  @Test
  void doesNotRestrictFunctionOriginOrSourceLocation() {
    assertThat(predicate.check(function(overload).withTypeOrigin(TypeOrigin.LOCAL).build(), context)).isEqualTo(TriBool.TRUE);
    var source = function(overload).withDefinitionLocation(new LocationInFile("source.py", 1, 2, 1, 10)).build();
    assertThat(source.typeOrigin()).isEqualTo(TypeOrigin.STUB);
    assertThat(predicate.check(source, context)).isEqualTo(TriBool.TRUE);
  }

  @Test
  void rejectsEmptyAndAdditionalKnownDecorators() {
    assertThat(predicate.check(function().build(), context)).isEqualTo(TriBool.FALSE);
    var decorated = function(overload, table.getType("typing.final")).build();
    assertThat(predicate.check(decorated, context)).isEqualTo(TriBool.FALSE);
    // The spelling alone must not establish canonical callable identity.
    var impostor = function().withName("overload").withFullyQualifiedName("typing.overload").build();
    assertThat(predicate.check(function(impostor).build(), context)).isEqualTo(TriBool.FALSE);
  }

  @Test
  void retainsUnknownDecoratorEntries() {
    assertThat(predicate.check(function(overload, PythonType.UNKNOWN).build(), context)).isEqualTo(TriBool.UNKNOWN);
    assertThat(predicate.check(function(PythonType.UNKNOWN, overload).build(), context)).isEqualTo(TriBool.UNKNOWN);
  }

  @Test
  void rejectsOtherTypeKindsAndRetainsUnknownInputs() {
    assertThat(predicate.check(table.getType("int"), context)).isEqualTo(TriBool.FALSE);
    assertThat(predicate.check(PythonType.UNKNOWN, context)).isEqualTo(TriBool.UNKNOWN);
  }

  @Test
  void unresolvedCanonicalTypeCannotEstablishAMatch() {
    var otherTable = mock(TypeTable.class);
    when(otherTable.getType("typing.overload")).thenReturn(PythonType.UNKNOWN);
    assertThat(predicate.check(function(overload).build(), TypePredicateContext.of(otherTable))).isEqualTo(TriBool.UNKNOWN);
  }

  @Test
  void factoryRequiresEveryUnionAlternativeToMatch() {
    var matcher = InternalTypeMatchers.hasOnlyFunctionDecoratorsSatisfying(TypeMatchers.isType("typing.overload"));
    var subscription = mock(SubscriptionContext.class);
    when(subscription.typeTable()).thenReturn(table);
    var expression = mock(Expression.class);
    FunctionType first = function(overload).build();
    FunctionType second = function(table.getType("typing_extensions.overload")).build();
    when(expression.typeV2()).thenReturn(UnionType.or(first, second));
    assertThat(matcher.isTrueFor(expression, subscription)).isTrue();
    when(expression.typeV2()).thenReturn(UnionType.or(first, function().build()));
    assertThat(matcher.isTrueFor(expression, subscription)).isFalse();
    when(expression.typeV2()).thenReturn(UnionType.or(first, function(overload, PythonType.UNKNOWN).build()));
    assertThat(matcher.evaluateFor(expression, subscription)).isEqualTo(TriBool.UNKNOWN);
  }

  @Test
  void acceptsComposedDecoratorMatchers() {
    var matcher = InternalTypeMatchers.hasOnlyFunctionDecoratorsSatisfying(
      TypeMatchers.any(TypeMatchers.isType("typing.overload"), TypeMatchers.isType("typing.final")));
    var allowed = MatchersTestUtils.getPredicate(matcher);
    assertThat(allowed.check(function(overload, table.getType("typing.final")).build(), context)).isEqualTo(TriBool.TRUE);
    assertThat(allowed.check(function(table.getType("typing.final")).build(), context)).isEqualTo(TriBool.TRUE);
    assertThat(allowed.check(function(overload, PythonType.UNKNOWN).build(), context)).isEqualTo(TriBool.UNKNOWN);
    assertThat(allowed.check(function(overload, table.getType("int")).build(), context)).isEqualTo(TriBool.FALSE);
  }

  @Test
  void knownMismatchTakesPrecedenceOverUnknownDecorators() {
    var other = table.getType("typing.final");
    assertThat(predicate.check(function(PythonType.UNKNOWN, other).build(), context)).isEqualTo(TriBool.FALSE);
    assertThat(predicate.check(function(other, PythonType.UNKNOWN).build(), context)).isEqualTo(TriBool.FALSE);
  }

  @Test
  void evaluatesEveryAlternativeOfADecoratorUnion() {
    assertThat(predicate.check(function(UnionType.or(overload, PythonType.UNKNOWN)).build(), context)).isEqualTo(TriBool.UNKNOWN);
    assertThat(predicate.check(function(UnionType.or(overload, table.getType("typing.final"))).build(), context)).isNotEqualTo(TriBool.TRUE);
  }

  private static FunctionTypeBuilder function(PythonType... decorators) {
    return new FunctionTypeBuilder().withName("library").withFullyQualifiedName("lib.library")
      .withAttributes(List.of()).withParameters(List.of()).withHasDecorators(true)
      .withDecorators(Arrays.stream(decorators).map(TypeWrapper::of).toList());
  }
}
