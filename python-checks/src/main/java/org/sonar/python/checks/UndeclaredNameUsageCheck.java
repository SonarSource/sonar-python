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

import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.ImportFrom;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;

@Rule(key = "S3827")
public class UndeclaredNameUsageCheck extends PythonSubscriptionCheck {
  private boolean hasWildcardImport = false;
  private boolean callGlobalsOrLocals = false;

  @Override
  public void initialize(Context context) {

    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> {
      FileInput fileInput = (FileInput) ctx.syntaxNode();
      ExceptionVisitor exceptionVisitor = new ExceptionVisitor();
      fileInput.accept(exceptionVisitor);
      hasWildcardImport = exceptionVisitor.hasWildcardImport;
      callGlobalsOrLocals = exceptionVisitor.callGlobalsOrLocals;
    });

    context.registerSyntaxNodeConsumer(Tree.Kind.NAME, ctx -> {
      Name name = (Name) ctx.syntaxNode();
      if (!callGlobalsOrLocals && !hasWildcardImport && name.isVariable() && name.symbol() == null) {
        ctx.addIssue(name, "Change its name or define it before using it");
      }
    });
  }

  private static class ExceptionVisitor extends BaseTreeVisitor {
    private boolean hasWildcardImport = false;
    private boolean callGlobalsOrLocals = false;

    @Override
    public void visitImportFrom(ImportFrom importFrom) {
      hasWildcardImport |= importFrom.isWildcardImport();
      super.visitImportFrom(importFrom);
    }

    @Override
    public void visitCallExpression(CallExpression callExpression) {
      if (callExpression.callee().is(Tree.Kind.NAME)) {
        String name = ((Name) callExpression.callee()).name();
        callGlobalsOrLocals |= name.equals("globals") || name.equals("locals");
      }
      super.visitCallExpression(callExpression);
    }
  }
}
