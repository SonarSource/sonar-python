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

import java.util.List;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.v2.SymbolV2;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.IfStatement;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.ReturnStatement;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatcher;
import org.sonar.plugins.python.api.types.v2.matchers.TypeMatchers;

@Rule(key = "S9461")
public class MappingInsteadOfDictCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Use \"Mapping\" instead of \"dict\" here.";
  private static final TypeMatcher IS_INSTANCE = TypeMatchers.isType("builtins.isinstance");
  private static final TypeMatcher DICT = TypeMatchers.isType("builtins.dict");
  private static final Set<String> MAPPING_READ_METHODS = Set.of("get", "items", "keys", "values");

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.IF_STMT, MappingInsteadOfDictCheck::checkIfStatement);
  }

  private static void checkIfStatement(SubscriptionContext ctx) {
    IfStatement ifStatement = (IfStatement) ctx.syntaxNode();
    CallExpression callExpression = isDictInstanceCheck(ifStatement.condition(), ctx);
    if (callExpression == null || ifStatement.body().statements().size() != 1) {
      return;
    }
    Statement statement = ifStatement.body().statements().get(0);
    if (!(statement instanceof ReturnStatement returnStatement) || returnStatement.expressions().size() != 1) {
      return;
    }
    Name guardedValue = firstArgument(callExpression);
    SymbolV2 guardedSymbol = guardedValue == null ? null : guardedValue.symbolV2();
    if (guardedSymbol != null && mappingReadCount(returnStatement.expressions().get(0), guardedSymbol) > 0) {
      ctx.addIssue(callExpression.arguments().get(1), MESSAGE);
    }
  }

  private static CallExpression isDictInstanceCheck(Expression expression, SubscriptionContext ctx) {
    if (!(expression instanceof CallExpression callExpression) || !IS_INSTANCE.isTrueFor(callExpression.callee(), ctx)) {
      return null;
    }
    List<Argument> arguments = callExpression.arguments();
    if (arguments.size() != 2 || !arguments.stream().allMatch(RegularArgument.class::isInstance)) {
      return null;
    }
    Expression typeArgument = ((RegularArgument) arguments.get(1)).expression();
    return DICT.isTrueFor(typeArgument, ctx) ? callExpression : null;
  }

  private static Name firstArgument(CallExpression callExpression) {
    Expression expression = ((RegularArgument) callExpression.arguments().get(0)).expression();
    return expression instanceof Name name && name.symbolV2() != null ? name : null;
  }

  private static int mappingReadCount(Tree tree, SymbolV2 guardedValue) {
    int readCount = 0;
    for (Tree child : tree.children()) {
      if (child instanceof Name name && guardedValue.equals(name.symbolV2())) {
        if (!isMappingRead(name)) {
          return -1;
        }
        readCount++;
      }
      int childReadCount = mappingReadCount(child, guardedValue);
      if (childReadCount < 0) {
        return -1;
      }
      readCount += childReadCount;
    }
    return readCount;
  }

  private static boolean isMappingRead(Name name) {
    Tree parent = name.parent();
    if (parent instanceof SubscriptionExpression subscriptionExpression && subscriptionExpression.object() == name) {
      return true;
    }
    return parent instanceof QualifiedExpression qualifiedExpression
      && qualifiedExpression.qualifier() == name
      && qualifiedExpression.parent() instanceof CallExpression callExpression
      && callExpression.callee() == qualifiedExpression
      && MAPPING_READ_METHODS.contains(qualifiedExpression.name().name());
  }
}
