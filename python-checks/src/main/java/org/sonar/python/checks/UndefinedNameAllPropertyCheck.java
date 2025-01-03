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

import java.io.File;
import java.net.URI;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.Function;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.ImportFrom;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.StringElement;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tuple;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.semantic.BuiltinSymbols;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S5807")
public class UndefinedNameAllPropertyCheck extends PythonSubscriptionCheck {

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.ASSIGNMENT_STMT, ctx -> {
      AssignmentStatement assignmentStatement = (AssignmentStatement) ctx.syntaxNode();
      if (TreeUtils.firstAncestorOfKind(assignmentStatement, Tree.Kind.CLASSDEF, Tree.Kind.FUNCDEF) != null) {
        // We only consider __all__ assignment at module level
        return;
      }
      ExpressionList expressionList = assignmentStatement.lhsExpressions().get(0);
      if (expressionList.expressions().size() > 1) {
        return;
      }
      Expression lhs = expressionList.expressions().get(0);
      if (lhs.is(Tree.Kind.NAME) && ((Name) lhs).name().equals("__all__")) {
        checkAllProperty(ctx, assignmentStatement);
      }
    });
  }

  private static void checkAllProperty(SubscriptionContext ctx, AssignmentStatement assignmentStatement) {
    Expression assignedValue = assignmentStatement.assignedValue();
    if (!assignedValue.is(Tree.Kind.LIST_LITERAL, Tree.Kind.TUPLE)) {
      return;
    }
    List<Tree> stringExpressions = getStringExpressions(assignedValue);
    FileInput fileInput = (FileInput) TreeUtils.firstAncestorOfKind(assignedValue, Tree.Kind.FILE_INPUT);
    if (fileInput == null || shouldExcludeFile(fileInput)) {
      return;
    }
    Map<String, Symbol> symbolsByName = fileInput.globalVariables().stream().collect(Collectors.toMap(Symbol::name, Function.identity()));
    for (Tree stringExpression : stringExpressions) {
      StringLiteral stringLiteral = retrieveStringLiteral(stringExpression);
      if (isUnknownSymbol(ctx.pythonFile(), symbolsByName, stringLiteral)) {
        PreciseIssue issue = ctx.addIssue(stringExpression, String.format("Change or remove this string; \"%s\" is not defined.", stringLiteral.trimmedQuotesValue()));
        if (stringExpression != stringLiteral) {
          issue.secondary(stringLiteral, "Assigned here.");
        }
      }
    }
  }

  private static boolean shouldExcludeFile(FileInput fileInput) {
    ModuleLevelVisitor moduleLevelVisitor = new ModuleLevelVisitor();
    fileInput.accept(moduleLevelVisitor);
    if (moduleLevelVisitor.hasGetAttrOrDirMethod) {
      return true;
    }
    UnknownNameSourcesVisitor unknownNameSourcesVisitor = new UnknownNameSourcesVisitor();
    fileInput.accept(unknownNameSourcesVisitor);
    return unknownNameSourcesVisitor.shouldNotReportIssue;
  }

  private static boolean isUnknownSymbol(PythonFile pythonFile, Map<String, Symbol> symbolsByName, StringLiteral stringLiteral) {
    String name = stringLiteral.trimmedQuotesValue();
    return stringLiteral.stringElements().stream().noneMatch(StringElement::isInterpolated)
      && !symbolsByName.containsKey(name)
      && !BuiltinSymbols.all().contains(name)
      && !isInitFileExportingModule(pythonFile, name);
  }

  /**
   * As __init__.py files can declare modules which they don't import in their __all__ property, we need to exclude those potential FPs
   */
  private static boolean isInitFileExportingModule(PythonFile pythonFile, String name) {
    return pythonFile.fileName().startsWith("__init__") && existsFileWithName(pythonFile.uri(), name);
  }

  private static List<Tree> getStringExpressions(Expression expression) {
    if (expression.is(Tree.Kind.LIST_LITERAL)) {
      return ((ListLiteral) expression).elements().expressions().stream()
        .filter(UndefinedNameAllPropertyCheck::isString)
        .collect(Collectors.toList());
    }
    return ((Tuple) expression).elements().stream()
      .filter(UndefinedNameAllPropertyCheck::isString)
      .collect(Collectors.toList());
  }

  private static boolean existsFileWithName(@Nullable URI uri, String name) {
    return Optional.ofNullable(uri)
      .map(u -> {
        String path = u.getPath();
        path = path.substring(0, path.lastIndexOf("/"));
        return new File(path, name + ".py").exists() || new File(path, name).exists();
      }).orElse(false);
  }

  private static boolean isString(Tree tree) {
    if (tree.is(Tree.Kind.STRING_LITERAL)) {
      return true;
    } else if (tree.is(Tree.Kind.NAME)) {
      Expression expression = Expressions.singleAssignedValue((Name) tree);
      return expression != null && expression.is(Tree.Kind.STRING_LITERAL);
    }
    return false;
  }

  private static StringLiteral retrieveStringLiteral(Tree tree) {
    if (tree.is(Tree.Kind.STRING_LITERAL)) {
      return (StringLiteral) tree;
    }
    return (StringLiteral) Expressions.singleAssignedValue((Name) tree);
  }

  private static class UnknownNameSourcesVisitor extends BaseTreeVisitor {

    private boolean shouldNotReportIssue = false;

    @Override
    public void visitImportFrom(ImportFrom importFrom) {
      shouldNotReportIssue |= importFrom.isWildcardImport();
      super.visitImportFrom(importFrom);
    }

    @Override
    public void visitCallExpression(CallExpression callExpression) {
      Symbol calleeSymbol = callExpression.calleeSymbol();
      shouldNotReportIssue |= isSymbolWithFQN(calleeSymbol, "globals");
      super.visitCallExpression(callExpression);
    }

    @Override
    public void visitSubscriptionExpression(SubscriptionExpression subscriptionExpression) {
      if (subscriptionExpression.object() instanceof HasSymbol hasSymbol) {
        Symbol symbol = hasSymbol.symbol();
        shouldNotReportIssue |= isSymbolWithFQN(symbol, "sys.modules");
      }
      super.visitSubscriptionExpression(subscriptionExpression);
    }

    private static boolean isSymbolWithFQN(@Nullable Symbol symbol, String fullyQualifiedName) {
      return symbol != null && fullyQualifiedName.equals(symbol.fullyQualifiedName());
    }
  }

  private static class ModuleLevelVisitor extends BaseTreeVisitor {

    private boolean hasGetAttrOrDirMethod = false;

    @Override
    public void visitFunctionDef(FunctionDef functionDef) {
      hasGetAttrOrDirMethod |= functionDef.name().name().equals("__getattr__") || functionDef.name().name().equals("__dir__");
      // Only visiting module-level functions
    }

    @Override
    public void visitClassDef(ClassDef classDef) {
      // Avoid visiting classes
    }
  }
}
