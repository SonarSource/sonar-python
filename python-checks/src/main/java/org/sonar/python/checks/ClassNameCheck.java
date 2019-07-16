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
import com.jetbrains.python.PyElementTypes;
import com.jetbrains.python.psi.PyClass;
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;

@Rule(key = "S101")
public class ClassNameCheck extends AbstractNameCheck {
  private static final String DEFAULT = "^[A-Z_][a-zA-Z0-9]+$";

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
    context.registerSyntaxNodeConsumer(PyElementTypes.CLASS_DECLARATION, ctx -> {
      ASTNode classNameNode = ((PyClass) ctx.syntaxNode()).getNameNode();
      if (classNameNode == null) {
        return;
      }
      String className = classNameNode.getText();
      if (!pattern().matcher(className).matches()) {
        String message = String.format("Rename class \"%s\" to match the regular expression %s.", className, format);
        ctx.addIssue(classNameNode.getPsi(), message);
      }
    });
  }

}
