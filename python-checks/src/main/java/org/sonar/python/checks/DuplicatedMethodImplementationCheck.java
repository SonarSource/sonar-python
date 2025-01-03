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

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.AnyParameter;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.ParameterList;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.quickfix.PythonTextEdit;
import org.sonar.python.checks.utils.CheckUtils;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.NameImpl;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S4144")
public class DuplicatedMethodImplementationCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Update this function so that its implementation is not identical to %s on line %s.";
  private static final String QUICK_FIX_MESSAGE = "Call %s inside this function.";

  private static final Set<String> ALLOWED_FIRST_ARG_NAMES = Set.of("self", "cls" ,"mcs", "metacls");
  private static final Set<String> CLASS_AND_STATIC_DECORATORS = Set.of("classmethod", "staticmethod");

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
        PreciseIssue issue = ctx.addIssue(suspiciousMethod.name(), message).secondary(originalMethod.name(), "Original");
        addQuickFix(issue, originalMethod, suspiciousMethod);
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

  private static void addQuickFix(PreciseIssue issue, FunctionDef originalMethod, FunctionDef suspiciousMethod) {
    ParameterList parameters = originalMethod.parameters();
    if (parameters != null) {
      List<AnyParameter> all = parameters.all();
      if (all.size() == 1 && !ALLOWED_FIRST_ARG_NAMES.contains(all.get(0).firstToken().value())) {
        return;
      }
      if (all.size() > 1) {
        return;
      }
    }
    boolean containsReturnStatement = originalMethod.body().statements().stream()
      .anyMatch(s -> s.is(Tree.Kind.RETURN_STMT));
    String replacementText = "";
    if (containsReturnStatement) {
      replacementText = "return ";
    }
    if (isClassOrStaticMethod(originalMethod)) {
      ClassDef methodClass = (ClassDef) TreeUtils.firstAncestorOfKind(originalMethod, Tree.Kind.CLASSDEF);
      if (methodClass != null) {
        replacementText = replacementText + methodClass.name().name() + ".";
      }
    } else {
      replacementText = replacementText + "self.";
    }
    replacementText = replacementText + originalMethod.name().name() + "()";
    PythonTextEdit edit = TextEditUtils.replace(suspiciousMethod.body(), replacementText);

    PythonQuickFix fix = PythonQuickFix
      .newQuickFix(String.format(QUICK_FIX_MESSAGE, originalMethod.name().name()))
      .addTextEdit(edit)
      .build();

    issue.addQuickFix(fix);
  }

  private static boolean isClassOrStaticMethod(FunctionDef originalMethod) {
    return originalMethod.decorators().stream()
      .anyMatch(d -> d.expression() instanceof NameImpl nameImpl
        && CLASS_AND_STATIC_DECORATORS.contains(nameImpl.name()));
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
