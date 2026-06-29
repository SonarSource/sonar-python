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
import org.sonar.plugins.python.api.types.v2.ClassType;
import org.sonar.plugins.python.api.types.v2.FunctionType;
import org.sonar.plugins.python.api.types.v2.ModuleType;
import org.sonar.plugins.python.api.types.v2.UnknownType;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.types.v2.SpecialFormType;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class HasFQNPredicateTest {

  @Test
  void testCheck() {
    FunctionType function1 = mock(FunctionType.class);
    FunctionType function2 = mock(FunctionType.class);
    FunctionType nullFunctionType = mock(FunctionType.class);

    ClassType class1 = mock(ClassType.class);
    ClassType class2 = mock(ClassType.class);
    ClassType classWithSameFQNAsFunction = mock(ClassType.class);

    UnknownType.UnresolvedImportType unresolvedImport = mock(UnknownType.UnresolvedImportType.class);
    UnknownType.UnresolvedImportType unresolvedImport2 = mock(UnknownType.UnresolvedImportType.class);

    ModuleType module1 = mock(ModuleType.class);
    ModuleType module2 = mock(ModuleType.class);

    SpecialFormType specialFormType1 = mock(SpecialFormType.class);
    SpecialFormType specialFormType2 = mock(SpecialFormType.class);

    UnknownType.UnknownTypeImpl unknownType = mock(UnknownType.UnknownTypeImpl.class);

    Expression func1Expression = mock(Expression.class);
    when(func1Expression.typeV2()).thenReturn(function1);

    when(function1.fullyQualifiedName()).thenReturn("foo.bar.func1");
    when(function2.fullyQualifiedName()).thenReturn("foo.bar.func2");
    when(nullFunctionType.fullyQualifiedName()).thenReturn(null);

    when(class1.fullyQualifiedName()).thenReturn("foo.bar.class1");
    when(class2.fullyQualifiedName()).thenReturn("foo.bar.class2");
    when(classWithSameFQNAsFunction.fullyQualifiedName()).thenReturn("foo.bar.func1");

    when(module1.fullyQualifiedName()).thenReturn("mod.module1");
    when(module2.fullyQualifiedName()).thenReturn("mod.module2");

    when(specialFormType1.fullyQualifiedName()).thenReturn("typing.List");
    when(specialFormType2.fullyQualifiedName()).thenReturn("typing.Set");

    when(unresolvedImport.importPath()).thenReturn("imported.module1");
    when(unresolvedImport2.importPath()).thenReturn("imported.module2");

    HasFQNPredicate hasFQNPredicateFunction1 = new HasFQNPredicate("foo.bar.func1");
    HasFQNPredicate hasFQNPredicateClass1 = new HasFQNPredicate("foo.bar.class1");
    HasFQNPredicate hasFQNPredicateImport1 = new HasFQNPredicate("imported.module1");
    HasFQNPredicate hasFQNPredicateModule1 = new HasFQNPredicate("mod.module1");
    HasFQNPredicate hasFQNPredicateSpecialFormList = new HasFQNPredicate("typing.List");

    TypePredicateContext predicateContext = TypePredicateContext.of(mock(org.sonar.python.semantic.v2.typetable.TypeTable.class));
    SubscriptionContext subscriptionContext = mock(SubscriptionContext.class);

    assertThat(hasFQNPredicateFunction1.check(function1, predicateContext)).isEqualTo(TriBool.TRUE);
    assertThat(TypeMatchers.withFQN("foo.bar.func1").evaluateFor(func1Expression, subscriptionContext)).isEqualTo(TriBool.TRUE);
    assertThat(hasFQNPredicateFunction1.check(function1, predicateContext)).isEqualTo(TriBool.TRUE);
    // Type is different but FQN is the same, we consider it as TRUE
    assertThat(hasFQNPredicateFunction1.check(classWithSameFQNAsFunction, predicateContext)).isEqualTo(TriBool.TRUE);
    assertThat(hasFQNPredicateFunction1.check(class1, predicateContext)).isEqualTo(TriBool.FALSE);

    assertThat(hasFQNPredicateClass1.check(class1, predicateContext)).isEqualTo(TriBool.TRUE);
    assertThat(hasFQNPredicateClass1.check(function1, predicateContext)).isEqualTo(TriBool.FALSE);
    assertThat(hasFQNPredicateClass1.check(class2, predicateContext)).isEqualTo(TriBool.FALSE);

    assertThat(hasFQNPredicateModule1.check(module1, predicateContext)).isEqualTo(TriBool.TRUE);
    assertThat(hasFQNPredicateModule1.check(module2, predicateContext)).isEqualTo(TriBool.FALSE);

    assertThat(hasFQNPredicateSpecialFormList.check(specialFormType1, predicateContext)).isEqualTo(TriBool.TRUE);
    assertThat(hasFQNPredicateSpecialFormList.check(specialFormType2, predicateContext)).isEqualTo(TriBool.FALSE);

    assertThat(hasFQNPredicateImport1.check(unresolvedImport, predicateContext)).isEqualTo(TriBool.TRUE);
    assertThat(hasFQNPredicateImport1.check(unresolvedImport2, predicateContext)).isEqualTo(TriBool.FALSE);

    assertThat(hasFQNPredicateFunction1.check(nullFunctionType, predicateContext)).isEqualTo(TriBool.UNKNOWN);
    assertThat(hasFQNPredicateFunction1.check(unknownType, predicateContext)).isEqualTo(TriBool.UNKNOWN);
  }
}

