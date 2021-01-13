/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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
package org.sonar.python;

import javax.annotation.Nullable;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;

/**
 * Extractor of docstring tokens.
 * <p>
 * Reminder: a docstring is a string literal that occurs as the first statement
 * in a module, function, class, or method definition.
 */
public class DocstringExtractor {

  private DocstringExtractor() {
  }

  public static StringLiteral extractDocstring(@Nullable StatementList statements) {
    if (statements != null) {
      Statement firstStatement = statements.statements().get(0);
      if (firstStatement.is(Tree.Kind.EXPRESSION_STMT) && ((ExpressionStatement) firstStatement).expressions().size() == 1
        && firstStatement.children().get(0).is(Tree.Kind.STRING_LITERAL)) {
        return (StringLiteral) firstStatement.children().get(0);
      }
    }
    return null;
  }
}
