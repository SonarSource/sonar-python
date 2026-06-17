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
import org.sonar.check.RuleProperty;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;

public abstract class AbstractFunctionNameCheck extends AbstractNameCheck {

  static final String DEFAULT = "^[a-z_][a-z0-9_]*$";
  private static final String MESSAGE = "Rename %s \"%s\" to match the regular expression %s.";

  @RuleProperty(
    key = "format",
    description = "Regular expression used to check the names against.",
    defaultValue = "" + DEFAULT)
  public String format = DEFAULT;

  @Override
  protected String format() {
    return format;
  }

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> {
      FunctionDef pyFunctionDefTree = (FunctionDef) ctx.syntaxNode();
      if (!shouldCheckFunctionDeclaration(pyFunctionDefTree)) {
        return;
      }
      Name functionNameTree = pyFunctionDefTree.name();
      String name = functionNameTree.name();
      if (!pattern().matcher(name).matches()) {
        String message = String.format(MESSAGE, typeName(), name, this.format);
        var issue = ctx.addIssue(functionNameTree, message);
        createQuickFix(functionNameTree).ifPresent(issue::addQuickFix);
      }
    });
  }

  private Optional<PythonQuickFix> createQuickFix(Name functionNameTree) {
    if (!DEFAULT.equals(format)) {
      return Optional.empty();
    }
    return NamingConventionQuickFixUtils.renameToSnakeCase(functionNameTree);
  }

  public abstract String typeName();

  public abstract boolean shouldCheckFunctionDeclaration(FunctionDef pyFunctionDefTree);

}
