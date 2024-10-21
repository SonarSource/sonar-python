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
package org.sonar.python.semantic.v2.types;


import java.util.Set;
import javax.annotation.CheckForNull;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.python.semantic.v2.ProjectLevelTypeTable;
import org.sonar.python.semantic.v2.SymbolV2;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.TypeSource;
import org.sonar.python.types.v2.UnknownType;

public class IsInstanceVisitor extends BaseTreeVisitor {
  private final PythonType isInstanceFunctionType;
  private TypeInferenceProgramState state;

  public IsInstanceVisitor(ProjectLevelTypeTable projectLevelTypeTable) {
    isInstanceFunctionType = projectLevelTypeTable.getType("isinstance");
  }

  public void setState(TypeInferenceProgramState state) {
    this.state = state;
  }

  @Override
  public void visitCallExpression(CallExpression callExpression) {
    if (callExpression.callee().typeV2() == isInstanceFunctionType && callExpression.arguments().size() == 2) {
      var firstArgumentSymbol = getFirstArgumentSymbol(callExpression);
      if (firstArgumentSymbol != null) {
        state.setTypes(firstArgumentSymbol, Set.of(PythonType.UNKNOWN));
      }
    }
    super.visitCallExpression(callExpression);
  }

  @CheckForNull
  private SymbolV2 getFirstArgumentSymbol(CallExpression callExpression) {
    var argument = callExpression.arguments().get(0);
    if (argument instanceof RegularArgument regularArgument
        && regularArgument.expression() instanceof Name variableName
        && state.getTypes(variableName.symbolV2()).stream()
          .anyMatch(type -> !(type instanceof UnknownType) && type.typeSource() == TypeSource.TYPE_HINT)) {
      return variableName.symbolV2();
    }
    return null;
  }
}
