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
package org.sonar.python.checks.django;

import java.util.Collection;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6553")
public class DjangoModelStringFieldCheck extends PythonSubscriptionCheck {

  private static final String REPLACE_MESSAGE = "Replace this \"null=True\" flag with \"blank=True\".";
  private static final String REMOVE_MESSAGE = "Remove this \"null=True\" flag.";
  private static final String REPLACE_QUICK_FIX_MESSAGE = "Replace with \"blank=True\"";
  private static final String REMOVE_QUICK_FIX_MESSAGE = "Remove the \"null=true\" flag";

  private static final String DJANGO_MODEL_FQN = "django.db.models.Model";
  public static final Set<String> FIELD_TYPES_FQN = Set.of(
    "django.db.models.CharField",
    "django.db.models.TextField"
  );

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CLASSDEF, ctx -> {
      var classDef = (ClassDef) ctx.syntaxNode();
      if (TreeUtils.getParentClassesFQN(classDef).contains(DJANGO_MODEL_FQN)) {
        var modelClassBodyStatements = classDef.body().statements();

        if (isNotManaged(modelClassBodyStatements)) {
          return;
        }

        modelClassBodyStatements.stream()
          .filter(AssignmentStatement.class::isInstance)
          .map(AssignmentStatement.class::cast)
          .map(AssignmentStatement::assignedValue)
          .filter(CallExpression.class::isInstance)
          .map(CallExpression.class::cast)
          .filter(DjangoModelStringFieldCheck::isTextField)
          .forEach(call -> validateTextFieldArguments(ctx, call));
      }
    });
  }

  private static boolean isNotManaged(List<Statement> modelClassBodyStatements) {
    return modelClassBodyStatements.stream()
      // lookup for nested Meta class
      .filter(ClassDef.class::isInstance)
      .map(ClassDef.class::cast)
      .filter(cd -> "Meta".equals(cd.name().name()))
      .map(ClassDef::body)
      // lookup for Meta class field assignments
      .map(StatementList::statements)
      .flatMap(Collection::stream)
      .filter(AssignmentStatement.class::isInstance)
      .map(AssignmentStatement.class::cast)
      // lookup for managed field assignment
      .filter(DjangoModelStringFieldCheck::isManagedFieldAssignment)
      // check if managed field assignment value is set to False
      .map(AssignmentStatement::assignedValue)
      .filter(Name.class::isInstance)
      .map(Name.class::cast)
      .map(Name::name)
      .anyMatch("False"::equals);
  }

  private static boolean isManagedFieldAssignment(AssignmentStatement fieldAssignment) {
    return fieldAssignment.lhsExpressions()
      .stream()
      .map(lhs -> TreeUtils.firstChild(lhs, Name.class::isInstance)
        .map(Name.class::cast)
        .orElse(null))
      .filter(Objects::nonNull)
      .map(Name::name)
      .anyMatch("managed"::equals);
  }

  private static boolean isTextField(CallExpression call) {
    return Optional.of(call)
      .map(CallExpression::calleeSymbol)
      .map(Symbol::fullyQualifiedName)
      .filter(FIELD_TYPES_FQN::contains)
      .isPresent();
  }

  private static void validateTextFieldArguments(SubscriptionContext ctx, CallExpression call) {
    var nullArg = getCallArgumentByName(call, "null")
      .filter(DjangoModelStringFieldCheck::isArgumentSetAsTrue);
    var blankArg = getCallArgumentByName(call, "blank")
      .filter(DjangoModelStringFieldCheck::isArgumentSetAsTrue);
    var uniqueArg = getCallArgumentByName(call, "unique")
      .filter(DjangoModelStringFieldCheck::isArgumentSetAsTrue);

    if (blankArg.isPresent() && uniqueArg.isPresent()) {
      return;
    }

    nullArg.ifPresent(arg -> {
      if (blankArg.isPresent()) {
        var issue = ctx.addIssue(arg, REMOVE_MESSAGE);
        createRemoveArgQuickFix(call, arg).ifPresent(issue::addQuickFix);
      } else {
        ctx.addIssue(arg, REPLACE_MESSAGE)
          .addQuickFix(PythonQuickFix.newQuickFix(REPLACE_QUICK_FIX_MESSAGE,
            TextEditUtils.replace(arg, "blank=True")));
      }
    });
  }

  private static Optional<RegularArgument> getCallArgumentByName(CallExpression call, String name) {
    return call.arguments().stream()
      .filter(RegularArgument.class::isInstance)
      .map(RegularArgument.class::cast)
      .filter(arg -> Objects.nonNull(arg.keywordArgument()))
      .filter(arg -> Optional.of(arg)
          .map(RegularArgument::keywordArgument)
          .map(Name::name)
          .filter(name::equals)
          .isPresent())
      .findFirst();
  }

  private static boolean isArgumentSetAsTrue(RegularArgument arg) {
    return Optional.of(arg)
      .map(RegularArgument::expression)
      .filter(TreeUtils::isBooleanLiteral)
      .filter(Name.class::isInstance)
      .map(Name.class::cast)
      .map(Name::name)
      .filter("True"::equals)
      .isPresent();
  }

  private static Optional<PythonQuickFix> createRemoveArgQuickFix(CallExpression call, Tree arg) {
    return Optional.ofNullable(call.argumentList())
      .map(Tree::children)
      .map(args -> {
        var count = args.size();
        var index = args.indexOf(arg);
        var removeFrom = arg;
        Tree removeTo;
        if (index == 0) {
          // if argument to be removed is first one - remove it and comma next to it
          removeTo = args.get(index + 2);
        } else {
          // if argument to be removed is not first one - remove comma going before the argument
          removeFrom = args.get(index - 1);
          if (index == count - 1) {
            // if argument to be removed is last one - remove until close parenthesis
            removeTo = call.rightPar();
          } else {
            // if argument to be removed is not last one - remove until next comma
            removeTo = args.get(index + 1);
          }
        }
        return TextEditUtils.removeUntil(removeFrom, removeTo);
      })
      .map(textEdit -> PythonQuickFix.newQuickFix(REMOVE_QUICK_FIX_MESSAGE, textEdit));
  }
}
