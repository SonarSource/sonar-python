/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

import java.util.Optional;
import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.ArgList;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.utils.CheckUtils;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6978")
public class TorchModuleShouldCallInitCheck extends PythonSubscriptionCheck {
  private static final String TORCH_NN_MODULE = "torch.nn.Module";
  private static final String MESSAGE = "Add a call to super().__init__()";
  private static final String SECONDARY_MESSAGE = "Inheritance happens here";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> {
      FunctionDef funcDef = (FunctionDef) ctx.syntaxNode();
      ClassDef classDef = CheckUtils.getParentClassDef(funcDef);
      if (isInheritingFromTorchModule(classDef) && isConstructor(funcDef) && isMissingSuperCall(funcDef)) {
        PreciseIssue issue = ctx.addIssue(funcDef.name(), MESSAGE);
        issue.secondary(classDef.name(), SECONDARY_MESSAGE);
      }
    });
  }

  private static boolean isConstructor(FunctionDef funcDef) {
    FunctionSymbol symbol = TreeUtils.getFunctionSymbolFromDef(funcDef);
    return symbol != null && "__init__".equals(symbol.name()) && funcDef.isMethodDefinition();
  }

  private static boolean isInheritingFromTorchModule(@Nullable ClassDef classDef) {
    if (classDef == null) return false;
    ArgList args = classDef.args();
    return args != null && args.arguments().stream()
      .flatMap(TreeUtils.toStreamInstanceOfMapper(RegularArgument.class))
      .map(arg -> getQualifiedName(arg.expression()))
      .anyMatch(expr -> expr.filter(TORCH_NN_MODULE::equals).isPresent());
  }

  private static Optional<String> getQualifiedName(Expression node) {
    return TreeUtils.getSymbolFromTree(node).flatMap(symbol -> Optional.ofNullable(symbol.fullyQualifiedName()));
  }

  private static boolean isMissingSuperCall(FunctionDef funcDef) {
    return !TreeUtils.hasDescendant(funcDef, t -> t.is(Tree.Kind.CALL_EXPR) && isSuperConstructorCall(((CallExpression) t)));
  }

  private static boolean isSuperConstructorCall(CallExpression callExpr) {
    return callExpr.callee() instanceof QualifiedExpression qualifiedCallee && isSuperCall(qualifiedCallee.qualifier()) && "__init__".equals(qualifiedCallee.name().name());
  }

  private static boolean isSuperCall(Expression qualifier) {
    if (qualifier instanceof CallExpression callExpression) {
      Symbol superSymbol = callExpression.calleeSymbol();
      return superSymbol != null && "super".equals(superSymbol.name());
    }
    return false;
  }
}
