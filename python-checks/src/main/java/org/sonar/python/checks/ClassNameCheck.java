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
package org.sonar.python.checks;

import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;

@Rule(key = ClassNameCheck.CHECK_KEY)
public class ClassNameCheck extends AbstractNameCheck {

  public static final String CHECK_KEY = "S101";
  static final String DEFAULT = "^_?([A-Z_][a-zA-Z0-9]*|[a-z_][a-z0-9_]*)$";
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
        var issue = ctx.addIssue(classNameTree, message);
        createQuickFix(classNameTree).ifPresent(issue::addQuickFix);
      }
    });
  }

  private Optional<org.sonar.plugins.python.api.quickfix.PythonQuickFix> createQuickFix(Name classNameTree) {
    if (!DEFAULT.equals(format)) {
      return Optional.empty();
    }
    return NamingConventionQuickFixUtils.renameToPascalCase(classNameTree);
  }
}
