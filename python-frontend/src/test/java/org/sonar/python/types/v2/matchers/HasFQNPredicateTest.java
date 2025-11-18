/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
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
import org.mockito.Mockito;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.types.v2.ClassType;
import org.sonar.plugins.python.api.types.v2.FunctionType;
import org.sonar.plugins.python.api.types.v2.ModuleType;
import org.sonar.plugins.python.api.types.v2.UnknownType;
import org.sonar.python.types.v2.SpecialFormType;

import static org.assertj.core.api.Assertions.assertThat;

class HasFQNPredicateTest {

  @Test
  void testCheck() {
    FunctionType function1 = Mockito.mock(FunctionType.class);
    FunctionType function2 = Mockito.mock(FunctionType.class);
    FunctionType nullFunctionType = Mockito.mock(FunctionType.class);

    ClassType class1 = Mockito.mock(ClassType.class);
    ClassType class2 = Mockito.mock(ClassType.class);
    ClassType classWithSameFQNAsFunction = Mockito.mock(ClassType.class);

    UnknownType.UnresolvedImportType unresolvedImport = Mockito.mock(UnknownType.UnresolvedImportType.class);
    UnknownType.UnresolvedImportType unresolvedImport2 = Mockito.mock(UnknownType.UnresolvedImportType.class);

    ModuleType module1 = Mockito.mock(ModuleType.class);
    ModuleType module2 = Mockito.mock(ModuleType.class);

    SpecialFormType specialFormType1 = Mockito.mock(SpecialFormType.class);
    SpecialFormType specialFormType2 = Mockito.mock(SpecialFormType.class);

    UnknownType.UnknownTypeImpl unknownType = Mockito.mock(UnknownType.UnknownTypeImpl.class);

    Expression func1Expression = Mockito.mock(Expression.class);
    Mockito.when(func1Expression.typeV2()).thenReturn(function1);

    Mockito.when(function1.fullyQualifiedName()).thenReturn("foo.bar.func1");
    Mockito.when(function2.fullyQualifiedName()).thenReturn("foo.bar.func2");
    Mockito.when(nullFunctionType.fullyQualifiedName()).thenReturn(null);

    Mockito.when(class1.fullyQualifiedName()).thenReturn("foo.bar.class1");
    Mockito.when(class2.fullyQualifiedName()).thenReturn("foo.bar.class2");
    Mockito.when(classWithSameFQNAsFunction.fullyQualifiedName()).thenReturn("foo.bar.func1");

    Mockito.when(module1.fullyQualifiedName()).thenReturn("mod.module1");
    Mockito.when(module2.fullyQualifiedName()).thenReturn("mod.module2");

    Mockito.when(specialFormType1.fullyQualifiedName()).thenReturn("typing.List");
    Mockito.when(specialFormType2.fullyQualifiedName()).thenReturn("typing.Set");

    Mockito.when(unresolvedImport.importPath()).thenReturn("imported.module1");
    Mockito.when(unresolvedImport2.importPath()).thenReturn("imported.module2");

    HasFQNPredicate hasFQNPredicateFunction1 = new HasFQNPredicate("foo.bar.func1");
    HasFQNPredicate hasFQNPredicateClass1 = new HasFQNPredicate("foo.bar.class1");
    HasFQNPredicate hasFQNPredicateImport1 = new HasFQNPredicate("imported.module1");
    HasFQNPredicate hasFQNPredicateModule1 = new HasFQNPredicate("mod.module1");
    HasFQNPredicate hasFQNPredicateSpecialFormList = new HasFQNPredicate("typing.List");

    assertThat(hasFQNPredicateFunction1.check(function1, null)).isEqualTo(TriBool.TRUE);
    assertThat(TypeMatchers.withFQN("foo.bar.func1").isFor(func1Expression, null)).isEqualTo(TriBool.TRUE);
    assertThat(hasFQNPredicateFunction1.check(function1, null)).isEqualTo(TriBool.TRUE);
    // Type is different but FQN is the same, we consider it as TRUE
    assertThat(hasFQNPredicateFunction1.check(classWithSameFQNAsFunction, null)).isEqualTo(TriBool.TRUE);
    assertThat(hasFQNPredicateFunction1.check(class1, null)).isEqualTo(TriBool.FALSE);

    assertThat(hasFQNPredicateClass1.check(class1, null)).isEqualTo(TriBool.TRUE);
    assertThat(hasFQNPredicateClass1.check(function1, null)).isEqualTo(TriBool.FALSE);
    assertThat(hasFQNPredicateClass1.check(class2, null)).isEqualTo(TriBool.FALSE);

    assertThat(hasFQNPredicateModule1.check(module1, null)).isEqualTo(TriBool.TRUE);
    assertThat(hasFQNPredicateModule1.check(module2, null)).isEqualTo(TriBool.FALSE);

    assertThat(hasFQNPredicateSpecialFormList.check(specialFormType1, null)).isEqualTo(TriBool.TRUE);
    assertThat(hasFQNPredicateSpecialFormList.check(specialFormType2, null)).isEqualTo(TriBool.FALSE);

    assertThat(hasFQNPredicateImport1.check(unresolvedImport, null)).isEqualTo(TriBool.TRUE);
    assertThat(hasFQNPredicateImport1.check(unresolvedImport2, null)).isEqualTo(TriBool.FALSE);

    assertThat(hasFQNPredicateFunction1.check(nullFunctionType, null)).isEqualTo(TriBool.UNKNOWN);
    assertThat(hasFQNPredicateFunction1.check(unknownType, null)).isEqualTo(TriBool.UNKNOWN);
  }
}

