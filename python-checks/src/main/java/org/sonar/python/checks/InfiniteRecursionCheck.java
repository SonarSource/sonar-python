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

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Deque;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.cfg.CfgBlock;
import org.sonar.plugins.python.api.cfg.ControlFlowGraph;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.ComprehensionExpression;
import org.sonar.plugins.python.api.tree.ComprehensionIf;
import org.sonar.plugins.python.api.tree.ConditionalExpression;
import org.sonar.plugins.python.api.tree.DictCompExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.LambdaExpression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.api.PythonKeyword;
import org.sonar.python.checks.utils.CheckUtils;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S2190")
public class InfiniteRecursionCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Add a way to break out of this %s's recursion.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> {
      FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();
      List<Tree> allRecursiveCalls = new ArrayList<>();
      boolean endBlockIsReachable = collectRecursiveCallsAndCheckIfEndBlockIsReachable(functionDef, ctx.pythonFile(), allRecursiveCalls);
      if (!allRecursiveCalls.isEmpty() && !endBlockIsReachable) {
        String message = String.format(MESSAGE, functionDef.isMethodDefinition() ? "method" : "function");
        PreciseIssue issue = ctx.addIssue(functionDef.name(), message);
        allRecursiveCalls.forEach(call -> issue.secondary(call, "recursive call"));
      }
    });
  }

  private static boolean collectRecursiveCallsAndCheckIfEndBlockIsReachable(FunctionDef functionDef, PythonFile pythonFile, List<Tree> allRecursiveCalls) {
    Symbol functionSymbol = functionDef.name().symbol();
    if (functionSymbol == null) {
      return true;
    }
    ControlFlowGraph cfg = ControlFlowGraph.build(functionDef, pythonFile);
    if (cfg == null) {
      return true;
    }
    RecursiveCallCollector recursiveCallCollector = new RecursiveCallCollector(functionDef, functionSymbol);
    Set<CfgBlock> pushedBlocks = new HashSet<>();
    Deque<CfgBlock> blockToVisit = new ArrayDeque<>();
    blockToVisit.addLast(cfg.start());
    pushedBlocks.add(cfg.start());
    while (!blockToVisit.isEmpty()) {
      CfgBlock block = blockToVisit.removeFirst();
      if (block == cfg.end()) {
        return true;
      }
      List<Tree> blockRecursiveCalls = recursiveCallCollector.findRecursiveCalls(block.elements());
      if (!blockRecursiveCalls.isEmpty()) {
        allRecursiveCalls.addAll(blockRecursiveCalls.stream()
          .filter(tree -> !isInsideTryBlock(tree))
          .toList());
      } else {
        block.successors().stream().filter(pushedBlocks::add).forEach(blockToVisit::addLast);
      }
    }
    return recursiveCallCollector.functionSymbolHasBeenReassigned;
  }

  private static boolean isInsideTryBlock(Tree tree) {
    Tree ancestor = TreeUtils.firstAncestorOfKind(tree, Tree.Kind.FINALLY_CLAUSE, Tree.Kind.TRY_STMT);
    return ancestor != null && ancestor.is(Tree.Kind.TRY_STMT);
  }

  private static class RecursiveCallCollector extends BaseTreeVisitor {

    private final boolean isMethod;
    private final Symbol functionSymbol;
    @Nullable
    private final Symbol selfSymbol;
    // Classes can not only be compared by their string names with the current semantic
    @Nullable
    private final String className;
    private boolean functionSymbolHasBeenReassigned = false;
    private boolean isAsync = false;
    private final List<Tree> recursiveCalls = new ArrayList<>();

    private RecursiveCallCollector(FunctionDef currentFunction, Symbol functionSymbol) {
      isMethod = currentFunction.isMethodDefinition();
      this.functionSymbol = functionSymbol;
      if (currentFunction.asyncKeyword() != null) {
        isAsync = true;
      }
      if (isMethod) {
        boolean isStatic = currentFunction.decorators().stream()
          .map(d -> TreeUtils.decoratorNameFromExpression(d.expression()))
          .anyMatch(decorator -> "staticmethod".equals(decorator) || "classmethod".equals(decorator));
        if (isStatic) {
          selfSymbol = null;
          className = findParentClassName(currentFunction);
        } else {
          selfSymbol = CheckUtils.findFirstParameterSymbol(currentFunction);
          className = null;
        }
      } else {
        selfSymbol = null;
        className = null;
      }
    }

    private List<Tree> findRecursiveCalls(List<Tree> elements) {
      recursiveCalls.clear();
      elements.forEach(element -> element.accept(this));
      return recursiveCalls;
    }

    @Override
    public void visitCallExpression(CallExpression callExpression) {
      Expression callee = callExpression.callee();
      if (!isAsyncWithoutAwait(callExpression) && (matchesLookupFunction(callee) || matchesLookupMethod(callee))) {
        recursiveCalls.add(callee);
      }
      super.visitCallExpression(callExpression);
    }

    private boolean isAsyncWithoutAwait(CallExpression callExpression) {
      Tree parent = TreeUtils.firstAncestorOfKind(callExpression, Tree.Kind.AWAIT, Tree.Kind.CALL_EXPR);
      return isAsync && (parent == null || !parent.is(Tree.Kind.AWAIT));
    }

    @Override
    public void visitFunctionDef(FunctionDef pyFunctionDefTree) {
      // ignore
    }

    @Override
    public void visitLambda(LambdaExpression pyLambdaExpressionTree) {
      // ignore
    }

    @Override
    public void visitConditionalExpression(ConditionalExpression pyConditionalExpressionTree) {
      scan(pyConditionalExpressionTree.condition());
      // ignore trueExpression and falseExpression, not broken down in the cfg
    }

    @Override
    public void visitPyListOrSetCompExpression(ComprehensionExpression tree) {
      scan(tree.comprehensionFor());
      // ignore resultExpression, not broken down in the cfg
    }

    @Override
    public void visitComprehensionIf(ComprehensionIf tree) {
      // ignore, not broken down in the cfg
    }

    @Override
    public void visitDictCompExpression(DictCompExpression tree) {
      // ignore, not broken down in the cfg
    }

    @Override
    public void visitAssignmentStatement(AssignmentStatement assignment) {
      if (isMethod && assignment.lhsExpressions().stream()
        .map(ExpressionList::expressions)
        .flatMap(Collection::stream)
        .anyMatch(expression -> matchesLookupSelf(expression) || matchesLookupMethod(expression))) {
        this.functionSymbolHasBeenReassigned = true;
      }
      super.visitAssignmentStatement(assignment);
    }

    @Override
    public void visitBinaryExpression(BinaryExpression pyBinaryExpressionTree) {
      scan(pyBinaryExpressionTree.leftOperand());
      // ignore conditional rightOperand, not broken down in the cfg
      String operator = pyBinaryExpressionTree.operator().value();
      if (!(PythonKeyword.OR.getValue().equals(operator) || PythonKeyword.AND.getValue().equals(operator))) {
        scan(pyBinaryExpressionTree.rightOperand());
      }
    }

    private boolean matchesLookupFunction(Expression expression) {
      if (!expression.is(Tree.Kind.NAME)) {
        return false;
      }
      Name name = (Name) expression;
      return !isMethod && functionSymbol.equals(name.symbol()) &&
        functionSymbol.usages().stream().filter(Usage::isBindingUsage).count() < 2;
    }

    private boolean matchesLookupMethod(Expression expression) {
      if (!expression.is(Tree.Kind.QUALIFIED_EXPR)) {
        return false;
      }
      QualifiedExpression qualifiedExpression = (QualifiedExpression) expression;
      // qualifiedExpression.name() symbols can not only be compared by their string names with the current semantic
      if (!isMethod || !functionSymbol.name().equals(qualifiedExpression.name().name())) {
        return false;
      }
      Expression qualifier = qualifiedExpression.qualifier();
      return matchesLookupSelf(qualifier) || matchesLookupClassName(qualifier);
    }

    private boolean matchesLookupSelf(Expression expression) {
      if (selfSymbol == null || !expression.is(Tree.Kind.NAME)) {
        return false;
      }
      return selfSymbol.equals(((Name) expression).symbol());
    }

    private boolean matchesLookupClassName(Expression expression) {
      if (className == null || !expression.is(Tree.Kind.NAME)) {
        return false;
      }
      return className.equals(((Name) expression).name());
    }

    @CheckForNull
    private static String findParentClassName(FunctionDef functionDef) {
      ClassDef parentClass = CheckUtils.getParentClassDef(functionDef);
      if (parentClass != null) {
        // classes symbols can not only be compared by their string names with the current semantic
        // and to prevent false-position, we ignore then a local variable with the same name exists
        String className = parentClass.name().name();
        boolean conflictsWithLocalVariable = functionDef.localVariables().stream().map(Symbol::name).anyMatch(className::equals);
        if (!conflictsWithLocalVariable) {
          return className;
        }
      }
      return null;
    }

  }

}
