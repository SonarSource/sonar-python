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
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.types.v2.ClassType;
import org.sonar.plugins.python.api.types.v2.ObjectType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.plugins.python.api.types.v2.SelfType;
import org.sonar.plugins.python.api.types.v2.UnknownType;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class IsSelfTypePredicateTest {

  private final IsSelfTypePredicate predicate = new IsSelfTypePredicate();
  private final TypePredicateContext ctx = TypePredicateContext.of(
    mock(org.sonar.python.semantic.v2.typetable.TypeTable.class));

  @Test
  void selfTypeReturnsTrue() {
    ClassType innerType = mock(ClassType.class);
    SelfType selfType = (SelfType) SelfType.of(innerType);

    assertThat(predicate.check(selfType, ctx)).isEqualTo(TriBool.TRUE);
  }

  @Test
  void objectTypeWrappingSelfTypeReturnsFalse() {
    ObjectType objectType = mock(ObjectType.class);
    SelfType selfType = mock(SelfType.class);
    when(objectType.type()).thenReturn(selfType);

    assertThat(predicate.check(objectType, ctx)).isEqualTo(TriBool.FALSE);
  }

  @Test
  void classTypeReturnsFalse() {
    ClassType classType = mock(ClassType.class);

    assertThat(predicate.check(classType, ctx)).isEqualTo(TriBool.FALSE);
  }

  @Test
  void unknownTypeReturnsUnknown() {
    assertThat(predicate.check(PythonType.UNKNOWN, ctx)).isEqualTo(TriBool.UNKNOWN);
  }

  @Test
  void unresolvedImportTypeReturnsUnknown() {
    UnknownType.UnresolvedImportType unresolvedImportType = mock(UnknownType.UnresolvedImportType.class);

    assertThat(predicate.check(unresolvedImportType, ctx)).isEqualTo(TriBool.UNKNOWN);
  }

  @Test
  void compositionWithIsObjectSatisfying() {
    ClassType innerType = mock(ClassType.class);
    SelfType selfType = (SelfType) SelfType.of(innerType);
    ObjectType objectType = mock(ObjectType.class);
    when(objectType.unwrappedType()).thenReturn(selfType);

    TypePredicate isObjectOfSelf = new IsObjectSatisfyingPredicate(predicate);

    assertThat(isObjectOfSelf.check(objectType, ctx)).isEqualTo(TriBool.TRUE);
    assertThat(isObjectOfSelf.check(selfType, ctx)).isEqualTo(TriBool.FALSE);
  }
}

