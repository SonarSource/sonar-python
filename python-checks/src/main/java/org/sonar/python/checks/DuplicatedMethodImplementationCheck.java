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
package org.sonar.python.checks;

import java.util.ArrayList;
import java.util.List;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.Tree;

@Rule(key = "S4144")
public class DuplicatedMethodImplementationCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Update this function so that its implementation is not identical to %s on line %s.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CLASSDEF, ctx -> {
      ClassDef classDef = (ClassDef) ctx.syntaxNode();
      MethodVisitor methodVisitor = new MethodVisitor();
      classDef.body().accept(methodVisitor);

      for (int i = 1; i < methodVisitor.methods.size(); i++) {
        checkMethods(methodVisitor.methods.get(i), methodVisitor.methods, i, ctx);
      }
    });
  }

  private static void checkMethods(FunctionDef suspiciousMethod, List<FunctionDef> methods, int index, SubscriptionContext ctx) {
    StatementList suspiciousBody = suspiciousMethod.body();
    if (isException(suspiciousMethod)) {
      return;
    }
    for (int j = 0; j < index; j++) {
      FunctionDef originalMethod = methods.get(j);
      Tree originalBody = originalMethod.body();
      if (CheckUtils.areEquivalent(originalBody, suspiciousBody)) {
        int line = originalMethod.name().firstToken().line();
        String message = String.format(MESSAGE, originalMethod.name().name(), line);
        ctx.addIssue(suspiciousMethod.name(), message).secondary(originalMethod.name(), "Original");
        break;
      }
    }
  }

  private static boolean isException(FunctionDef suspiciousMethod) {
    boolean hasDocString = suspiciousMethod.docstring() != null;
    StatementList suspiciousBody = suspiciousMethod.body();
    List<Statement> statements = suspiciousBody.statements();
    int nbActualStatements = hasDocString ? statements.size() - 1 : statements.size();
    if (nbActualStatements == 0 || isOnASingleLine(suspiciousBody, hasDocString)) {
      return true;
    }
    return nbActualStatements == 1 && statements.get(statements.size() - 1).is(Tree.Kind.RAISE_STMT);
  }


  private static boolean isOnASingleLine(StatementList statementList, boolean hasDocString) {
    int first = hasDocString ? 1 : 0;
    return statementList.statements().get(first).firstToken().line() == statementList.statements().get(statementList.statements().size() - 1).lastToken().line();
  }

  private static class MethodVisitor extends BaseTreeVisitor {

    List<FunctionDef> methods = new ArrayList<>();

    @Override
    public void visitClassDef(ClassDef classDef) {
      //avoid raising on nested classes methods
    }

    @Override
    public void visitFunctionDef(FunctionDef functionDef) {
      if (functionDef.isMethodDefinition()) {
        methods.add(functionDef);
      }
      super.visitFunctionDef(functionDef);
    }

  }
}
