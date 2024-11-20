/*
 * SonarQube Python Plugin
 * Copyright (C) 2012-2024 SonarSource SA
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
package org.sonar.samples.python;

import org.sonar.check.Priority;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonVisitorCheck;
import org.sonar.plugins.python.api.tree.FunctionDef;

@Rule(
  key = CustomPythonVisitorCheck.RULE_KEY_VISITOR,
  priority = Priority.MINOR,
  name = "Python visitor check",
  description = "desc")
public class CustomPythonVisitorCheck extends PythonVisitorCheck {

  public static final String RULE_KEY_VISITOR = "visitor";

  @Override
  public void visitFunctionDef(FunctionDef pyFunctionDefTree) {
    addIssue(pyFunctionDefTree.name(), "Function def.");
    super.visitFunctionDef(pyFunctionDefTree);
  }

}
