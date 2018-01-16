/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2018 SonarSource SA
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

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.AstNodeType;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import org.sonar.check.Rule;
import org.sonar.python.PythonCheck;
import org.sonar.python.api.PythonGrammar;
import org.sonar.python.api.PythonTokenType;

@Rule(key = MethodShouldBeStaticCheck.CHECK_KEY)
public class MethodShouldBeStaticCheck extends PythonCheck {

  public static final String CHECK_KEY = "S2325";

  private static final String MESSAGE = "Make this method static.";

  @Override
  public Set<AstNodeType> subscribedKinds() {
    return Collections.singleton(PythonGrammar.FUNCDEF);
  }

  @Override
  public void visitNode(AstNode node) {
    if (CheckUtils.isMethodOfNonDerivedClass(node)
      && !alreadyStaticMethod(node)
      && !isBuiltInMethod(node)
      && hasValuableCode(node)
      && !mayRaiseNotImplementedError(node)) {
      String self = getFirstArgument(node);
      if (self != null && !isUsed(node, self)) {
        addIssue(node.getFirstChild(PythonGrammar.FUNCNAME), MESSAGE);
      }
    }
  }

  private static boolean mayRaiseNotImplementedError(AstNode function) {
    return function.getDescendants(PythonGrammar.RAISE_STMT).stream()
      .map(raise -> raise.getFirstDescendant(PythonGrammar.TEST))
      .filter(Objects::nonNull)
      .anyMatch(test -> "NotImplementedError".equals(test.getToken().getValue()));
  }
  
  private static boolean hasValuableCode(AstNode funcDef) {
    AstNode statementList = funcDef.getFirstChild(PythonGrammar.SUITE).getFirstChild(PythonGrammar.STMT_LIST);
    if (statementList != null && statementList.getChildren(PythonGrammar.SIMPLE_STMT).size() == 1) {
      return !statementList.getFirstChild(PythonGrammar.SIMPLE_STMT).getFirstChild().is(PythonGrammar.PASS_STMT);
    }

    List<AstNode> statements = funcDef.getFirstChild(PythonGrammar.SUITE).getChildren(PythonGrammar.STATEMENT);
    if (statements.size() == 1) {
      return !isDocstringOrPass(statements.get(0));
    }

    return statements.size() != 2 || !isDocstringAndPass(statements.get(0), statements.get(1));
  }

  private static boolean isDocstringOrPass(AstNode statement) {
    return statement.getFirstDescendant(PythonGrammar.PASS_STMT) != null || statement.getToken().getType().equals(PythonTokenType.STRING);
  }

  private static boolean isDocstringAndPass(AstNode statement1, AstNode statement2) {
    return statement1.getToken().getType().equals(PythonTokenType.STRING) && statement2.getFirstDescendant(PythonGrammar.PASS_STMT) != null;
  }

  private static boolean isUsed(AstNode funcDef, String self) {
    List<AstNode> names = funcDef.getFirstChild(PythonGrammar.SUITE).getDescendants(PythonGrammar.NAME);
    for (AstNode name : names) {
      if (name.getTokenValue().equals(self)) {
        return true;
      }
    }
    return false;
  }

  private static String getFirstArgument(AstNode funcDef) {
    AstNode argList = funcDef.getFirstChild(PythonGrammar.TYPEDARGSLIST);
    if (argList != null) {
      return argList.getFirstDescendant(PythonGrammar.NAME).getTokenValue();
    } else {
      return null;
    }
  }

  private static boolean alreadyStaticMethod(AstNode funcDef) {
    AstNode decorators = funcDef.getFirstChild(PythonGrammar.DECORATORS);
    if (decorators != null) {
      List<AstNode> decoratorList = decorators.getChildren(PythonGrammar.DECORATOR);
      for (AstNode decorator : decoratorList) {
        AstNode name = decorator.getFirstDescendant(PythonGrammar.NAME);
        if (name != null && ("staticmethod".equals(name.getTokenValue()) || "classmethod".equals(name.getTokenValue()))) {
          return true;
        }
      }
    }
    return false;
  }

  private static boolean isBuiltInMethod(AstNode funcDef) {
    String name = funcDef.getFirstChild(PythonGrammar.FUNCNAME).getToken().getValue();
    String doubleUnderscore = "__";
    return name.startsWith(doubleUnderscore) && name.endsWith(doubleUnderscore);
  }

}
