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

import com.intellij.lang.ASTNode;
import com.jetbrains.python.PyStubElementTypes;
import com.jetbrains.python.psi.PyFunction;
import org.sonar.check.RuleProperty;

public abstract class AbstractFunctionNameCheck extends AbstractNameCheck {
  private static final String DEFAULT = "^[a-z_][a-z0-9_]{2,}$";

  @RuleProperty(
    key = "format",
    defaultValue = "" + DEFAULT)
  public String format = DEFAULT;

  @Override
  protected String format() {
    return format;
  }

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(PyStubElementTypes.FUNCTION_DECLARATION, ctx -> {
      PyFunction node = (PyFunction) ctx.syntaxNode();
      if (!shouldCheckFunctionDeclaration(node)) {
        return;
      }
      ASTNode nameNode = node.getNameNode();
      if (nameNode == null) {
        return;
      }
      String name = nameNode.getText();
      if (!pattern().matcher(name).matches()) {
        String message = String.format("Rename %s \"%s\" to match the regular expression %s.", typeName(), name, format);
        ctx.addIssue(nameNode.getPsi(), message);
      }
    });
  }

  public abstract String typeName();

  public abstract boolean shouldCheckFunctionDeclaration(PyFunction function);

}
