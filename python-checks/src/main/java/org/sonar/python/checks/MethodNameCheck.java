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
package org.sonar.python.checks;

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.tree.FunctionDef;

import static org.sonar.python.checks.utils.CheckUtils.classHasInheritance;
import static org.sonar.python.checks.utils.CheckUtils.getParentClassDef;

@Rule(key = MethodNameCheck.CHECK_KEY)
public class MethodNameCheck extends AbstractFunctionNameCheck {
  public static final String CHECK_KEY = "S100";

  @Override
  public String typeName() {
    return "method";
  }

  @Override
  public boolean shouldCheckFunctionDeclaration(FunctionDef pyFunctionDefTree) {
    return pyFunctionDefTree.isMethodDefinition() && !classHasInheritance(getParentClassDef(pyFunctionDefTree));
  }
}
