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

import com.intellij.psi.PsiComment;
import com.intellij.psi.PsiElement;
import com.intellij.psi.SyntaxTraverser;
import com.intellij.psi.tree.IElementType;
import com.intellij.util.containers.JBIterable;
import com.jetbrains.python.PyElementTypes;
import com.jetbrains.python.psi.PyStatement;
import com.jetbrains.python.psi.PyStatementList;
import java.util.Arrays;
import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.python.PythonCheck;

@Rule(key = "S108")
public class EmptyNestedBlockCheck extends PythonCheck {

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(PyElementTypes.STATEMENT_LIST, ctx -> {
      PyStatementList statementList = (PyStatementList) ctx.syntaxNode();

      PsiElement parent = statementList.getParent();
      IElementType parentType = parent.getNode().getElementType();
      if (parentType == PyElementTypes.FUNCTION_DECLARATION
        || parentType == PyElementTypes.CLASS_DECLARATION
        || parentType == PyElementTypes.EXCEPT_PART) {
        return;
      }

      Optional<PyStatement> nonPassStatement = Arrays.stream(statementList.getStatements())
        .filter(s -> s.getNode().getElementType() != PyElementTypes.PASS_STATEMENT)
        .findFirst();
      if (!nonPassStatement.isPresent() && !containsComment(statementList)) {
        ctx.addIssue(statementList, "Either remove or fill this block of code.");
      }
    });
  }

  private static boolean containsComment(PyStatementList statementList) {
    JBIterable<PsiComment> comments = SyntaxTraverser.psiTraverser(statementList.getParent()).traverse().filter(PsiComment.class);
    return !comments.isEmpty();
  }

}
