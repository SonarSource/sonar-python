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

import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.types.v2.FunctionType;
import org.sonar.plugins.python.api.types.v2.UnknownType;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.semantic.v2.typetable.TypeTable;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class HasFQNSatisfyingPredicateTest {

  @Test
  void testCheck() {
    FunctionType numpyZeros = mock(FunctionType.class);
    FunctionType numpyRandom = mock(FunctionType.class);
    FunctionType other = mock(FunctionType.class);
    FunctionType nullFqn = mock(FunctionType.class);
    UnknownType.UnresolvedImportType unresolvedNumpy = mock(UnknownType.UnresolvedImportType.class);
    UnknownType.UnresolvedImportType unresolvedOther = mock(UnknownType.UnresolvedImportType.class);
    UnknownType.UnknownTypeImpl unknownType = mock(UnknownType.UnknownTypeImpl.class);

    when(numpyZeros.fullyQualifiedName()).thenReturn("numpy.zeros");
    when(numpyRandom.fullyQualifiedName()).thenReturn("numpy.random.randn");
    when(other.fullyQualifiedName()).thenReturn("pandas.Series");
    when(nullFqn.fullyQualifiedName()).thenReturn(null);
    when(unresolvedNumpy.importPath()).thenReturn("numpy.polynomial.legendre.legval");
    when(unresolvedOther.importPath()).thenReturn("numpyish.zeros");

    Expression numpyZerosExpr = mock(Expression.class);
    when(numpyZerosExpr.typeV2()).thenReturn(numpyZeros);

    HasFQNSatisfyingPredicate predicate = new HasFQNSatisfyingPredicate(
      typeFqn -> typeFqn.startsWith("numpy.") ? TriBool.TRUE : TriBool.FALSE);
    TypePredicateContext predicateContext = TypePredicateContext.of(mock(TypeTable.class));
    SubscriptionContext subscriptionContext = mock(SubscriptionContext.class);

    assertThat(predicate.check(numpyZeros, predicateContext)).isEqualTo(TriBool.TRUE);
    assertThat(predicate.check(numpyRandom, predicateContext)).isEqualTo(TriBool.TRUE);
    assertThat(predicate.check(unresolvedNumpy, predicateContext)).isEqualTo(TriBool.TRUE);
    assertThat(predicate.check(other, predicateContext)).isEqualTo(TriBool.FALSE);
    assertThat(predicate.check(unresolvedOther, predicateContext)).isEqualTo(TriBool.FALSE);
    assertThat(predicate.check(nullFqn, predicateContext)).isEqualTo(TriBool.UNKNOWN);
    assertThat(predicate.check(unknownType, predicateContext)).isEqualTo(TriBool.UNKNOWN);

    assertThat(TypeMatchers.withFQNPrefix("numpy.").evaluateFor(numpyZerosExpr, subscriptionContext))
      .isEqualTo(TriBool.TRUE);
    assertThat(TypeMatchers.withFQNSatisfying(typeFqn -> typeFqn.contains("zeros") ? TriBool.TRUE : TriBool.FALSE)
      .evaluateFor(numpyZerosExpr, subscriptionContext))
      .isEqualTo(TriBool.TRUE);
  }
}
