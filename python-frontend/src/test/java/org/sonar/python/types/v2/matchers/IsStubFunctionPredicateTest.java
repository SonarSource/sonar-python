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

import java.util.List;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.TypeOrigin;
import org.sonar.plugins.python.api.types.v2.TypeWrapper;
import org.sonar.plugins.python.api.types.v2.UnionType;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.v2.FunctionTypeBuilder;
import org.sonar.python.semantic.v2.typetable.ProjectLevelTypeTable;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class IsStubFunctionPredicateTest {
  private final ProjectLevelTypeTable table = new ProjectLevelTypeTable(ProjectLevelSymbolTable.empty());
  private final TypePredicateContext context = TypePredicateContext.of(table);
  private final TypePredicate predicate = new IsStubFunctionPredicate();

  @Test
  void matchesUndecoratedAndDecoratedLibraryFunctions() {
    assertThat(predicate.check(function().build(), context)).isEqualTo(TriBool.TRUE);
    assertThat(predicate.check(function().withDecorators(List.of(TypeWrapper.UNKNOWN_TYPE_WRAPPER)).build(), context)).isEqualTo(TriBool.TRUE);
  }

  @Test
  void excludesSourceDefinitionsEvenWithStubOrigin() {
    assertThat(predicate.check(function().withTypeOrigin(TypeOrigin.LOCAL).build(), context)).isEqualTo(TriBool.FALSE);
    var source = function().withDefinitionLocation(new LocationInFile("source.py", 1, 2, 1, 10)).build();
    assertThat(source.typeOrigin()).isEqualTo(TypeOrigin.STUB);
    assertThat(predicate.check(source, context)).isEqualTo(TriBool.FALSE);
  }

  @Test
  void rejectsOtherKindsAndRetainsUnknownInputs() {
    assertThat(predicate.check(table.getType("int"), context)).isEqualTo(TriBool.FALSE);
    assertThat(predicate.check(PythonType.UNKNOWN, context)).isEqualTo(TriBool.UNKNOWN);
  }

  @Test
  void composesWithDecoratorMatchersAndPreservesUnionSemantics() {
    var matcher = TypeMatchers.all(
      InternalTypeMatchers.isStubFunction(),
      InternalTypeMatchers.hasOnlyFunctionDecoratorsSatisfying(TypeMatchers.isType("typing.overload")));
    var subscription = mock(SubscriptionContext.class);
    when(subscription.typeTable()).thenReturn(table);
    var expression = mock(Expression.class);
    var decorated = function().withDecorators(List.of(TypeWrapper.of(table.getType("typing.overload")))).build();
    when(expression.typeV2()).thenReturn(decorated);
    assertThat(matcher.isTrueFor(expression, subscription)).isTrue();
    when(expression.typeV2()).thenReturn(function().build());
    assertThat(matcher.isTrueFor(expression, subscription)).isFalse();
    var local = function().withTypeOrigin(TypeOrigin.LOCAL).withDecorators(decorated.decorators()).build();
    when(expression.typeV2()).thenReturn(local);
    assertThat(matcher.isTrueFor(expression, subscription)).isFalse();
    when(expression.typeV2()).thenReturn(UnionType.or(decorated, local));
    assertThat(matcher.isTrueFor(expression, subscription)).isFalse();
    when(expression.typeV2()).thenReturn(UnionType.or(decorated, PythonType.UNKNOWN));
    assertThat(matcher.evaluateFor(expression, subscription)).isEqualTo(TriBool.UNKNOWN);
  }

  private static FunctionTypeBuilder function() {
    return new FunctionTypeBuilder().withName("library").withFullyQualifiedName("lib.library")
      .withAttributes(List.of()).withParameters(List.of()).withDecorators(List.of());
  }
}
