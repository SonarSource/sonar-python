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

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
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
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.FunctionDefImpl;

@Rule(key = "S6538")
public class MandatoryFunctionReturnTypeHintCheck extends PythonSubscriptionCheck {

  public static final String MESSAGE = "Add a return type hint to this function declaration.";
  public static final String CONSTRUCTOR_MESSAGE = "Annotate the return type of this constructor with `None`.";
  private static final List<SupportedReturnType> SUPPORTED_TYPES = List.of(
    new SupportedReturnType(TypeMatchers.isObjectOfType("builtins.str"), BuiltinTypes.STR),
    new SupportedReturnType(TypeMatchers.isObjectOfType("NoneType"), "None"),
    new SupportedReturnType(TypeMatchers.isObjectOfType("builtins.bool"), BuiltinTypes.BOOL),
    new SupportedReturnType(TypeMatchers.isObjectOfType("builtins.complex"), BuiltinTypes.COMPLEX),
    new SupportedReturnType(TypeMatchers.isObjectOfType("builtins.float"), BuiltinTypes.FLOAT),
    new SupportedReturnType(TypeMatchers.isObjectOfType("builtins.int"), BuiltinTypes.INT));

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
      addQuickFixForReturnType(ctx, issue, functionDef, returnStatementVisitor.returnStatements);
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

  private static void addQuickFixForReturnType(SubscriptionContext ctx, PreciseIssue issue, FunctionDef functionDef, List<ReturnStatement> statements) {
    Set<String> returnTypes = collectSupportedReturnTypeAnnotations(statements, ctx);
    if (returnTypes.size() == 1) {
      String annotation = returnTypes.iterator().next();
      PythonQuickFix quickFix = PythonQuickFix.newQuickFix(MandatoryFunctionReturnTypeHintCheck.MESSAGE)
        .addTextEdit(TextEditUtils.insertAfter(functionDef.rightPar(), String.format(" -> %s", annotation)))
        .build();
      issue.addQuickFix(quickFix);
    }
  }

  private static Set<String> collectSupportedReturnTypeAnnotations(List<ReturnStatement> statements, SubscriptionContext ctx) {
    Set<String> returnTypes = new HashSet<>();
    for (ReturnStatement stmt : statements) {
      for (Expression expression : stmt.expressions()) {
        Optional<String> annotation = supportedReturnTypeAnnotation(expression, ctx);
        if (annotation.isEmpty()) {
          return Set.of();
        }
        returnTypes.add(annotation.get());
      }
    }
    return returnTypes;
  }

  private static Optional<String> supportedReturnTypeAnnotation(Expression expression, SubscriptionContext ctx) {
    return SUPPORTED_TYPES.stream()
      .filter(supportedType -> supportedType.matcher().isTrueFor(expression, ctx))
      .map(SupportedReturnType::annotation)
      .findFirst();
  }

  private record SupportedReturnType(TypeMatcher matcher, String annotation) {
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
