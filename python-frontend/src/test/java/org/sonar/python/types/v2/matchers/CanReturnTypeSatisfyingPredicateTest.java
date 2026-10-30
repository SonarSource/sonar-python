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

import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.types.v2.ObjectType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.UnionType;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.v2.FunctionTypeBuilder;
import org.sonar.python.semantic.v2.typetable.ProjectLevelTypeTable;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class CanReturnTypeSatisfyingPredicateTest {
  private final ProjectLevelTypeTable table = new ProjectLevelTypeTable(ProjectLevelSymbolTable.empty());
  private final TypePredicateContext context = TypePredicateContext.of(table);
  private final TypePredicate predicate = new CanReturnTypeSatisfyingPredicate(new HasMemberPredicate("__setitem__"));
  private final PythonType dictionary = ObjectType.Builder.fromType(table.getType("builtins.dict")).build();
  private final PythonType integer = ObjectType.Builder.fromType(table.getType("builtins.int")).build();
  private final PythonType floating = ObjectType.Builder.fromType(table.getType("builtins.float")).build();

  @Test
  void checksKnownReturnTypes() {
    assertThat(predicate.check(function(dictionary).build(), context)).isEqualTo(TriBool.TRUE);
    assertThat(predicate.check(function(integer).build(), context)).isEqualTo(TriBool.FALSE);
  }

  @Test
  void preservesUnresolvedReturnsAndCallableShapes() {
    assertThat(predicate.check(function(PythonType.UNKNOWN).build(), context)).isEqualTo(TriBool.UNKNOWN);
    assertThat(predicate.check(PythonType.UNKNOWN, context)).isEqualTo(TriBool.UNKNOWN);
    assertThat(predicate.check(integer, context)).isEqualTo(TriBool.UNKNOWN);
  }

  @Test
  void doesNotUseDecoratedOrAsynchronousFunctionAnnotations() {
    assertThat(predicate.check(function(integer).withHasDecorators(true).build(), context)).isEqualTo(TriBool.UNKNOWN);
    assertThat(predicate.check(function(integer).withAsynchronous(true).build(), context)).isEqualTo(TriBool.UNKNOWN);
  }

  @Test
  void evaluatesAllReturnUnionAlternatives() {
    assertThat(predicate.check(function(UnionType.or(integer, floating)).build(), context)).isEqualTo(TriBool.FALSE);
    assertThat(predicate.check(function(ObjectType.fromType(UnionType.or(integer, floating))).build(), context)).isEqualTo(TriBool.FALSE);
    assertThat(predicate.check(function(UnionType.or(integer, dictionary)).build(), context)).isEqualTo(TriBool.TRUE);
    assertThat(predicate.check(function(UnionType.or(integer, PythonType.UNKNOWN)).build(), context)).isEqualTo(TriBool.UNKNOWN);
    var integerReturn = new CanReturnTypeSatisfyingPredicate(new IsObjectSatisfyingPredicate(new IsTypePredicate("builtins.int")));
    assertThat(integerReturn.check(function(UnionType.or(integer, floating)).build(), context)).isEqualTo(TriBool.TRUE);
  }

  @Test
  void evaluatesAllCallableUnionAlternatives() {
    assertThat(predicate.check(UnionType.or(function(integer).build(), function(floating).build()), context)).isEqualTo(TriBool.FALSE);
    assertThat(predicate.check(UnionType.or(function(integer).build(), function(dictionary).build()), context)).isEqualTo(TriBool.TRUE);
    assertThat(predicate.check(UnionType.or(function(integer).build(), PythonType.UNKNOWN), context)).isEqualTo(TriBool.UNKNOWN);
  }

  @Test
  void composesWithMemberMatcherForOverloadedGetters() {
    var getter = UnionType.or(function(integer).build(), function(dictionary).build());
    PythonType descriptor = mock(PythonType.class);
    when(descriptor.resolveMember("__get__")).thenReturn(Optional.of(getter));
    var expression = mock(Expression.class);
    when(expression.typeV2()).thenReturn(descriptor);
    var subscription = mock(SubscriptionContext.class);
    when(subscription.typeTable()).thenReturn(table);
    var matcher = TypeMatchers.hasMemberSatisfying("__get__",
      InternalTypeMatchers.canReturnTypeSatisfying(TypeMatchers.hasMember("__setitem__")));
    assertThat(matcher.evaluateFor(expression, subscription)).isEqualTo(TriBool.TRUE);
    when(descriptor.resolveMember("__get__")).thenReturn(Optional.of(function(integer).build()));
    assertThat(matcher.evaluateFor(expression, subscription)).isEqualTo(TriBool.FALSE);
    when(descriptor.resolveMember("__get__")).thenReturn(Optional.of(function(dictionary).build()));
    assertThat(matcher.evaluateFor(expression, subscription)).isEqualTo(TriBool.TRUE);
  }

  @Test
  void factoryPreservesExistentialCallableUnionSemantics() {
    var expression = mock(Expression.class);
    var subscription = mock(SubscriptionContext.class);
    when(subscription.typeTable()).thenReturn(table);
    var matcher = InternalTypeMatchers.canReturnTypeSatisfying(TypeMatchers.hasMember("__setitem__"));
    when(expression.typeV2()).thenReturn(UnionType.or(function(integer).build(), function(dictionary).build()));
    assertThat(matcher.evaluateFor(expression, subscription)).isEqualTo(TriBool.TRUE);
    when(expression.typeV2()).thenReturn(UnionType.or(function(integer).build(), function(integer).withHasDecorators(true).build()));
    assertThat(matcher.evaluateFor(expression, subscription)).isEqualTo(TriBool.UNKNOWN);
    when(expression.typeV2()).thenReturn(UnionType.or(function(integer).build(), function(floating).build()));
    assertThat(matcher.evaluateFor(expression, subscription)).isEqualTo(TriBool.FALSE);
  }

  private static FunctionTypeBuilder function(PythonType returnType) {
    return new FunctionTypeBuilder().withName("get").withFullyQualifiedName("module.get").withReturnType(returnType);
  }
}
