/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.ReturnStatement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.YieldStatement;
import org.sonar.plugins.python.api.types.BuiltinTypes;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.FunctionDefImpl;
import org.sonar.python.types.InferredTypes;

@Rule(key = "S6538")
public class MandatoryFunctionReturnTypeHintCheck extends PythonSubscriptionCheck {

  public static final String MESSAGE = "Add a return type hint to this function declaration.";
  public static final String CONSTRUCTOR_MESSAGE = "Annotate the return type of this constructor with `None`.";
  private static final List<String> SUPPORTED_TYPES = List.of(
    BuiltinTypes.STR,
    BuiltinTypes.NONE_TYPE,
    BuiltinTypes.BOOL,
    BuiltinTypes.COMPLEX,
    BuiltinTypes.FLOAT,
    BuiltinTypes.INT);

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> {
      FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();
      if (functionDef.returnTypeAnnotation() == null) {
        Name functionName = functionDef.name();
        FunctionDefImpl functionDefImpl = (FunctionDefImpl) functionDef;
        Optional.ofNullable(functionDefImpl.functionSymbol())
          .filter(functionSymbol -> "__init__".equals(functionName.name()) && functionSymbol.isInstanceMethod())
          .ifPresentOrElse(symbol -> raiseIssueForConstructor(ctx, functionName, functionDef), () -> raiseIssueForReturnType(ctx, functionName, functionDef));
      }
    });
  }

  private static void raiseIssueForConstructor(SubscriptionContext ctx, Name functionName, FunctionDef functionDef) {
    PreciseIssue preciseIssue = ctx.addIssue(functionName, CONSTRUCTOR_MESSAGE);
    PythonQuickFix quickFix = PythonQuickFix.newQuickFix(CONSTRUCTOR_MESSAGE)
      .addTextEdit(TextEditUtils.insertAfter(functionDef.rightPar(), " -> None"))
      .build();
    preciseIssue.addQuickFix(quickFix);
  }

  private static void raiseIssueForReturnType(SubscriptionContext ctx, Name functionName, FunctionDef functionDef) {
    PreciseIssue issue = ctx.addIssue(functionName, MESSAGE);
    ReturnStatementVisitor returnStatementVisitor = new ReturnStatementVisitor();
    functionDef.body().accept(returnStatementVisitor);
    if (!returnStatementVisitor.returnStatements.isEmpty()) {
      addQuickFixForReturnType(issue, functionDef, returnStatementVisitor.returnStatements);
    } else if (returnStatementVisitor.yieldStatements.isEmpty()) {
      addQuickFixForNoneType(issue, functionDef);
    }
  }

  private static void addQuickFixForNoneType(PreciseIssue issue, FunctionDef functionDef) {
    PythonQuickFix quickFix = PythonQuickFix.newQuickFix(MandatoryFunctionReturnTypeHintCheck.MESSAGE)
      .addTextEdit(TextEditUtils.insertAfter(functionDef.rightPar(), " -> None"))
      .build();
    issue.addQuickFix(quickFix);
  }

  private static void addQuickFixForReturnType(PreciseIssue issue, FunctionDef functionDef, List<ReturnStatement> statements) {
    Set<String> returnTypes = statements.stream()
      .flatMap(stmts -> stmts.expressions().stream())
      .map(Expression::type)
      .map(InferredTypes::typeName)
      .filter(Objects::nonNull)
      .collect(Collectors.toSet());
    if (returnTypes.size() == 1) {
      String typeName = returnTypes.stream().iterator().next();
      if (SUPPORTED_TYPES.contains(typeName)) {
        PythonQuickFix quickFix = PythonQuickFix.newQuickFix(MandatoryFunctionReturnTypeHintCheck.MESSAGE)
          .addTextEdit(TextEditUtils.insertAfter(functionDef.rightPar(), String.format(" -> %s", fixTypeName(typeName))))
          .build();
        issue.addQuickFix(quickFix);
      }
    }
  }

  private static String fixTypeName(String typeName) {
    return typeName.equals(BuiltinTypes.NONE_TYPE) ? "None" : typeName;
  }

  private static class ReturnStatementVisitor extends BaseTreeVisitor {

    private final List<ReturnStatement> returnStatements = new ArrayList<>();
    private final List<YieldStatement> yieldStatements = new ArrayList<>();

    @Override
    public void visitReturnStatement(ReturnStatement pyReturnStatementTree) {
      super.visitReturnStatement(pyReturnStatementTree);
      returnStatements.add(pyReturnStatementTree);
    }

    @Override
    public void visitYieldStatement(YieldStatement pyYieldStatementTree) {
      super.visitYieldStatement(pyYieldStatementTree);
      yieldStatements.add(pyYieldStatementTree);
    }
  }
}
