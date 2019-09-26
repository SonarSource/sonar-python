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

import java.text.MessageFormat;
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.python.PythonSubscriptionCheck;
import org.sonar.python.api.tree.FileInput;
import org.sonar.python.api.tree.Tree;
import org.sonar.python.tree.BaseTreeVisitor;

@Rule(key = TooManyLinesInFileCheck.CHECK_KEY)
public class TooManyLinesInFileCheck extends PythonSubscriptionCheck {

  public static final String CHECK_KEY = "S104";
  private static final int DEFAULT = 1000;
  private static final String MESSAGE = "File \"{0}\" has {1} lines, which is greater than {2} authorized. Split it into smaller files.";

  @RuleProperty(
    key = "maximum",
    defaultValue = "" + DEFAULT)
  public int maximum = DEFAULT;

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> {
      FileVisitor visitor = new FileVisitor();
      ctx.syntaxNode().accept(visitor);
      if (visitor.numberOfLines > maximum) {
        ctx.addFileIssue(MessageFormat.format(MESSAGE, ctx.pythonFile().fileName(), visitor.numberOfLines, maximum));
      }
    });
  }

  private static class FileVisitor extends BaseTreeVisitor {

    private int numberOfLines = 0;

    @Override
    public void visitFileInput(FileInput fileInput) {
      if (fileInput.statements() != null) {
        numberOfLines = fileInput.statements().tokens().get(fileInput.statements().tokens().size() - 1).line();
      }
    }
  }
}

