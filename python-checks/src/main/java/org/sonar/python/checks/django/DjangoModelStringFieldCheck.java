/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
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
package org.sonar.python.checks.django;

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
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6553")
public class DjangoModelStringFieldCheck extends PythonSubscriptionCheck {

  private static final String REPLACE_MESSAGE = "Replace this \"null=True\" flag with \"blank=True\".";
  private static final String REMOVE_MESSAGE = "Remove this \"null=True\" flag.";
  private static final String REPLACE_QUICK_FIX_MESSAGE = "Replace with \"blank=True\"";
  private static final String REMOVE_QUICK_FIX_MESSAGE = "Remove \"null\" argument";

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
        classDef.body().statements().stream()
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
          removeTo = args.get(index + 2);
        } else {
          removeFrom = args.get(index - 1);
          if (index == count - 1) {
            removeTo = call.rightPar();
          } else {
            removeTo = args.get(index + 1);
          }
        }
        return TextEditUtils.removeUntil(removeFrom, removeTo);
      })
      .map(textEdit -> PythonQuickFix.newQuickFix(REMOVE_QUICK_FIX_MESSAGE, textEdit));
  }
}
