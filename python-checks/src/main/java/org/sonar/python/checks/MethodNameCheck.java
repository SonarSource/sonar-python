/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
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
package org.sonar.python.checks;

import java.util.List;
import org.sonar.check.Rule;
import org.sonar.python.api.tree.PyArgListTree;
import org.sonar.python.api.tree.PyClassDefTree;
import org.sonar.python.api.tree.PyFunctionDefTree;
import org.sonar.python.api.tree.Tree;

@Rule(key = MethodNameCheck.CHECK_KEY)
public class MethodNameCheck extends AbstractFunctionNameCheck {
  public static final String CHECK_KEY = "S100";
  @Override
  public String typeName() {
    return "method";
  }

  @Override
  public boolean shouldCheckFunctionDeclaration(PyFunctionDefTree pyFunctionDefTree) {
    return pyFunctionDefTree.isMethodDefinition() && !classHasInheritance(getParentClassDef(pyFunctionDefTree));
  }

  PyClassDefTree getParentClassDef(Tree current) {
    if(current == null) {
      return null;
    } else if (current.is(Tree.Kind.CLASSDEF)) {
      return (PyClassDefTree) current;
    } else {
      return getParentClassDef(current.parent());
    }
  }

  private static boolean classHasInheritance(PyClassDefTree classDef) {
    PyArgListTree args = classDef.args();
    if(args == null) {
      return false;
    }
    List<Tree> children = args.children();
    if(children.isEmpty()) {
      return false;
    }
    return children.size() != 1 || !"object".equals(children.get(0).firstToken().getValue());
  }
}
