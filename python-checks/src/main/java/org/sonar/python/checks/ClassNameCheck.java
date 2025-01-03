/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
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
import org.sonar.check.RuleProperty;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;

@Rule(key = ClassNameCheck.CHECK_KEY)
public class ClassNameCheck extends AbstractNameCheck {

  public static final String CHECK_KEY = "S101";
  private static final String DEFAULT = "^_?([A-Z_][a-zA-Z0-9]*|[a-z_][a-z0-9_]*)$";
  private static final String MESSAGE = "Rename class \"%s\" to match the regular expression %s.";

  @RuleProperty(
    key = "format",
    description = "Regular expression used to check the class names against",
    defaultValue = "" + DEFAULT)
  public String format = DEFAULT;

  @Override
  protected String format() {
    return format;
  }

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CLASSDEF, ctx -> {
      ClassDef pyClassDefTree = (ClassDef) ctx.syntaxNode();
      Name classNameTree = pyClassDefTree.name();
      String className = classNameTree.name();
      if(!pattern().matcher(className).matches()) {
        String message = String.format(MESSAGE, className, format);
        ctx.addIssue(classNameTree, message);
      }
    });
  }
}
