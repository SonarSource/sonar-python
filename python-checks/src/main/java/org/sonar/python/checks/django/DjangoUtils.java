/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
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
