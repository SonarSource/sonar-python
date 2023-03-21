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
package org.sonar.python.checks;

import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.FunctionDefImpl;

@Rule(key = "S6538")
public class MandatoryFunctionReturnTypeHintCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Add a return type hint to this function declaration.";
  private static final String CONSTRUCTOR_MESSAGE = "Annotate the return type of this constructor with `None`.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> {
      FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();
      if (functionDef.returnTypeAnnotation() == null) {
        Name functionName = functionDef.name();
        FunctionDefImpl functionDefImpl = (FunctionDefImpl) functionDef;
        Optional.ofNullable(functionDefImpl.functionSymbol())
          .filter(functionSymbol -> "__init__".equals(functionName.name()) && functionSymbol.isInstanceMethod())
          .ifPresentOrElse(symbol -> ctx.addIssue(functionName, CONSTRUCTOR_MESSAGE), () -> ctx.addIssue(functionName, MESSAGE));
      }
    });
  }
}
