/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
package org.sonar.python.checks.django;

import java.util.Collection;
import java.util.Objects;
import java.util.Optional;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.Name;

public class DjangoUtils {

  private DjangoUtils() {
  }

  public static Optional<ClassDef> getMetaClass(ClassDef formClass) {
    return formClass.body()
      .statements()
      .stream()
      .filter(ClassDef.class::isInstance)
      .map(ClassDef.class::cast)
      .filter(nestedClass -> Objects.equals("Meta", nestedClass.name().name()))
      .findFirst();
  }

  public static Optional<AssignmentStatement> getFieldAssignment(ClassDef metaClass, String fieldName) {
    return metaClass.body()
      .statements()
      .stream()
      .filter(AssignmentStatement.class::isInstance)
      .map(AssignmentStatement.class::cast)
      .filter(assignment -> assignment.lhsExpressions()
        .stream()
        .map(ExpressionList::expressions)
        .flatMap(Collection::stream)
        .filter(Name.class::isInstance)
        .map(Name.class::cast)
        .anyMatch(name -> fieldName.equals(name.name())))
      .findFirst();
  }
}
