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
package org.sonar.python.checks.utils;

import java.time.Duration;
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.AnnotatedAssignment;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.ConditionalExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.NumericLiteral;
import org.sonar.python.parser.PythonParser;
import org.sonar.python.semantic.v2.SymbolTableBuilderV2;
import org.sonar.python.tree.PythonTreeMaker;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class ReachingValuesTest {

  private final SubscriptionContext ctx = mock(SubscriptionContext.class);

  @Test
  void preservesTerminalExpressionsWithoutDescendingIntoCalls() {
    Expression call = lastExpression(parse("factory(argument)"));
    assertThat(ReachingValues.resolvedValues(call, ctx)).containsExactly(call);
  }

  @Test
  void unwrapsNestedParentheses() {
    Expression expression = lastExpression(parse("((42))"));
    assertThat(ReachingValues.resolvedValues(expression, ctx)).singleElement()
      .isInstanceOfSatisfying(NumericLiteral.class, literal -> assertThat(literal.valueAsString()).isEqualTo("42"));
  }

  @Test
  void followsAnnotatedAndPlainAssignmentsWithoutDataFlow() {
    FileInput file = parse("source: int = 42; alias = source; ((alias))");
    Expression value = ((AnnotatedAssignment) file.statements().statements().get(0)).assignedValue();
    assertThat(ReachingValues.resolvedValues(lastExpression(file), ctx)).containsExactly(value);
  }

  @Test
  void expandsConditionalBranchesThroughAliases() {
    FileInput file = parse("source = left() if flag else right(); alias = source; alias");
    var conditional = (ConditionalExpression) ((AssignmentStatement) file.statements().statements().get(0)).assignedValue();
    assertThat(ReachingValues.resolvedValues(lastExpression(file), ctx))
      .containsExactlyInAnyOrder(conditional.trueExpression(), conditional.falseExpression());
  }

  @Test
  void reachingDefinitionsTakePrecedenceOverTheSingleBindingFallback() {
    FileInput file = parse("value = stale(); value");
    Name name = (Name) lastExpression(file);
    Expression reachingValue = lastExpression(parse("current()"));
    when(ctx.valuesAtLocation(name)).thenReturn(Set.of(reachingValue));

    assertThat(ReachingValues.resolvedValues(name, ctx)).containsExactly(reachingValue);
    assertThat(ReachingValues.singleValue(name, ctx)).isSameAs(reachingValue);
  }

  @Test
  void preservesAllReachingAlternativesIncludingUnresolvedNames() {
    Name name = (Name) lastExpression(parse("value"));
    Expression known = lastExpression(parse("factory()"));
    Expression unresolved = lastExpression(parse("unknown"));
    when(ctx.valuesAtLocation(name)).thenReturn(Set.of(known, unresolved));

    assertThat(ReachingValues.resolvedValues(name, ctx)).containsExactlyInAnyOrder(known, unresolved);
    assertThat(ReachingValues.singleValue(name, ctx)).isNull();
  }

  @ParameterizedTest
  @ValueSource(strings = {
    "missing",
    "value: int; value",
    "value = first(); value = second(); value",
    "value, other = pair(); value",
    "import value; value"
  })
  void preservesNamesWhenNoUniqueAssignmentCanBeEstablished(String code) {
    Name name = (Name) lastExpression(parse(code));
    assertThat(ReachingValues.resolvedValues(name, ctx)).containsExactly(name);
    assertThat(ReachingValues.singleValue(name, ctx)).isNull();
  }

  @Test
  void terminatesAliasCyclesWithoutDroppingTheUnresolvedValue() {
    FileInput file = parse("first = second; second = first; first");
    Expression cycle = ((AssignmentStatement) file.statements().statements().get(0)).assignedValue();
    assertTimeoutPreemptively(Duration.ofSeconds(2), () ->
      assertThat(ReachingValues.resolvedValues(lastExpression(file), ctx)).containsExactly(cycle));
  }

  @Test
  void aKnownAlternativeDoesNotHideACycle() {
    Name name = (Name) lastExpression(parse("value"));
    Expression known = lastExpression(parse("factory()"));
    when(ctx.valuesAtLocation(name)).thenReturn(Set.of(name, known));
    assertTimeoutPreemptively(Duration.ofSeconds(2), () ->
      assertThat(ReachingValues.resolvedValues(name, ctx)).containsExactlyInAnyOrder(name, known));
  }

  @ParameterizedTest
  @ValueSource(strings = {
    "source = factory(); first = source; second = source; first if flag else second",
    "source = factory(); alias = source; alias if flag else alias",
    "source = factory(); alias = source; first = alias; second = alias; first if flag else second"
  })
  void deduplicatesValuesReachedThroughSharedAliases(String code) {
    FileInput file = parse(code);
    Expression value = ((AssignmentStatement) file.statements().statements().get(0)).assignedValue();
    assertThat(ReachingValues.resolvedValues(lastExpression(file), ctx)).containsExactly(value);
  }

  private static FileInput parse(String code) {
    FileInput file = new PythonTreeMaker().fileInput(PythonParser.create().parse(code));
    new SymbolTableBuilderV2(file).build();
    return file;
  }

  private static Expression lastExpression(FileInput file) {
    return ((ExpressionStatement) file.statements().statements().getLast()).expressions().get(0);
  }
}
