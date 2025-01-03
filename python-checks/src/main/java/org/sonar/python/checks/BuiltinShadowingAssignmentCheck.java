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

import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.AnnotatedAssignment;
import org.sonar.plugins.python.api.tree.AnyParameter;
import org.sonar.plugins.python.api.tree.AssignmentExpression;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionLike;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.ParameterList;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TupleParameter;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.TypeShed;

@Rule(key = "S5806")
public class BuiltinShadowingAssignmentCheck extends PythonSubscriptionCheck {

  public static final String MESSAGE = "Rename this variable; it shadows a builtin.";
  public static final String REPEATED_VAR_MESSAGE = "Variable also assigned here.";
  public static final String RENAME_PREFIX = "_";
  public static final String QUICK_FIX_MESSAGE_FORMAT = "Rename to " + RENAME_PREFIX+ " %s";
  private final Map<Symbol, PreciseIssue> variableIssuesRaised = new HashMap<>();
  private static final Set<String> notBuiltins = Set.of("ellipsis", "function");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> variableIssuesRaised.clear());
    context.registerSyntaxNodeConsumer(Tree.Kind.ASSIGNMENT_STMT, this::checkAssignment);
    context.registerSyntaxNodeConsumer(Tree.Kind.ANNOTATED_ASSIGNMENT, this::checkAnnotatedAssignment);
    context.registerSyntaxNodeConsumer(Tree.Kind.ASSIGNMENT_EXPRESSION, this::checkAssignmentExpression);
  }

  private void checkAssignmentExpression(SubscriptionContext ctx) {
    AssignmentExpression assignmentExpression = (AssignmentExpression) ctx.syntaxNode();
    Name lhsName = assignmentExpression.lhsName();
    if (shouldReportIssue(lhsName)) {
      raiseIssueForNonGlobalVariable(ctx, lhsName);
    }
  }

  private void checkAssignment(SubscriptionContext ctx) {
    AssignmentStatement assignment = (AssignmentStatement) ctx.syntaxNode();
    Tree ancestor = TreeUtils.firstAncestorOfKind(assignment, Tree.Kind.FUNCDEF, Tree.Kind.CLASSDEF);
    if (ancestor == null || ancestor.is(Tree.Kind.FUNCDEF)) {
      for (int i = 0; i < assignment.lhsExpressions().size(); i++) {
        for (Expression expression : assignment.lhsExpressions().get(i).expressions()) {
          if (shouldReportIssue(expression)) {
            raiseIssueForNonGlobalVariable(ctx, (Name) expression);
          }
        }
      }
    }
  }

  private void checkAnnotatedAssignment(SubscriptionContext ctx) {
    AnnotatedAssignment assignment = (AnnotatedAssignment) ctx.syntaxNode();
    Tree ancestor = TreeUtils.firstAncestorOfKind(assignment, Tree.Kind.FUNCDEF, Tree.Kind.CLASSDEF);
    if (ancestor == null || ancestor.is(Tree.Kind.FUNCDEF)) {
      Expression variable = assignment.variable();
      Token equalToken = assignment.equalToken();
      if (equalToken != null && shouldReportIssue(variable)) {
        raiseIssueForNonGlobalVariable(ctx, (Name) variable);
      }
    }
  }

  private static Set<String> collectUsedNames(Tree tree) {
    return Optional.ofNullable(TreeUtils.firstAncestorOfKind(tree, Tree.Kind.FUNCDEF))
      .map(FunctionLike.class::cast)
      .map(BuiltinShadowingAssignmentCheck::collectFunctionVariableNames)
      .stream()
      .flatMap(Function.identity())
      .collect(Collectors.toSet());
  }

  private static Stream<String> collectFunctionVariableNames(FunctionLike funcDef) {
    return Stream.concat(
      funcDef.localVariables().stream()
        .map(Symbol::name),
      Optional.of(funcDef)
        .map(FunctionLike::parameters)
        .map(ParameterList::all)
        .stream()
        .flatMap(Collection::stream)
        .map(BuiltinShadowingAssignmentCheck::collectParameterNames)
        .flatMap(Collection::stream)
    );
  }

  private static Set<String> collectParameterNames(AnyParameter parameter) {
    var result = new HashSet<String>();

    Optional.of(parameter)
      .filter(Parameter.class::isInstance)
      .map(Parameter.class::cast)
      .map(Parameter::name)
      .map(Name::name)
      .ifPresent(result::add);

    Optional.of(parameter)
      .filter(TupleParameter.class::isInstance)
      .map(TupleParameter.class::cast)
      .map(TupleParameter::parameters)
      .stream()
      .flatMap(Collection::stream)
      .map(BuiltinShadowingAssignmentCheck::collectParameterNames)
      .forEach(result::addAll);

    return result;
  }

  private void raiseIssueForNonGlobalVariable(SubscriptionContext ctx, Name variable) {
    Optional.ofNullable(variable.symbol())
      .filter(symbol -> symbol.usages().stream().map(Usage::kind).noneMatch(Usage.Kind.GLOBAL_DECLARATION::equals))
      .ifPresent(symbol -> {
        var existingIssue = variableIssuesRaised.get(symbol);
        if (existingIssue != null) {
          existingIssue.secondary(variable, REPEATED_VAR_MESSAGE);
        } else {
          var issue = ctx.addIssue(variable, MESSAGE);
          variableIssuesRaised.put(symbol, issue);

          var names = collectUsedNames(variable);
          if (!names.contains(RENAME_PREFIX + variable.name())) {
            var quickFix = createQuickFix(symbol);
            issue.addQuickFix(quickFix);
          }
        }
      });
  }

  private static PythonQuickFix createQuickFix(Symbol symbol) {
    var edits = symbol.usages()
      .stream()
      .map(Usage::tree)
      .map(Tree::firstToken)
      .map(token -> TextEditUtils.insertBefore(token, RENAME_PREFIX))
      .toList();

    return PythonQuickFix.newQuickFix(String.format(QUICK_FIX_MESSAGE_FORMAT, symbol.name()))
      .addTextEdit(edits)
      .build();
  }

  private boolean shouldReportIssue(Tree tree) {
    return tree.is(Tree.Kind.NAME) && isBuiltInName((Name) tree) && TreeUtils.firstAncestorOfKind(tree.parent(), Tree.Kind.FUNCDEF) != null;
  }

  private boolean isBuiltInName(Name name) {
    if (notBuiltins.contains(name.name())) {
      return false;
    }
    return TypeShed.builtinSymbols().containsKey(name.name());
  }
}
