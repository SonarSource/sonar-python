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
import com.sonar.sslr.api.GenericTokenType;
import com.sonar.sslr.api.Token;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.python.PythonCheck;
import org.sonar.python.api.PythonGrammar;

@Rule(key = LocalVariableAndParameterNameConventionCheck.CHECK_KEY)
public class LocalVariableAndParameterNameConventionCheck extends PythonCheck {

  public static final String CHECK_KEY = "S117";

  public static final String MESSAGE = "Rename this %s \"%s\" to match the regular expression %s.";
  public static final String PARAMETER = "parameter";
  public static final String LOCAL_VAR = "local variable";

  private static final String CONSTANT_PATTERN = "^[_A-Z][A-Z0-9_]*$";

  private static final String DEFAULT = "^[_a-z][a-z0-9_]*$";
  @RuleProperty(key = "format", defaultValue = DEFAULT)
  public String format = DEFAULT;

  private Pattern pattern = null;
  private Pattern constantPattern = null;

  @Override
  public Set<AstNodeType> subscribedKinds() {
    return Collections.singleton(PythonGrammar.FUNCDEF);
  }

  @Override
  public void visitNode(AstNode astNode) {
    List<Token> parameters = visitParameters(astNode);
    visitLocalVariables(astNode, parameters);
  }

  private void visitLocalVariables(AstNode funcDef, List<Token> parameters) {
    AstNode suite = funcDef.getFirstChild(PythonGrammar.SUITE);
    if (suite != null) {
      List<AstNode> expressions = suite.getDescendants(PythonGrammar.EXPRESSION_STMT);
      List<Token> varNames = new LinkedList<>();
      List<Token> forCounterNames = getForCounterNames(suite);
      for (AstNode expression : expressions) {
        if (CheckUtils.isAssignmentExpression(expression) && CheckUtils.insideFunction(expression, funcDef)) {
          varNames = new NewSymbolsAnalyzer().getVariablesFromLongAssignmentExpression(varNames, expression);
        }
      }
      for (int i = 0; i < varNames.size(); i++) {
        if (CheckUtils.containsValue(parameters, varNames.get(i).getValue()) || CheckUtils.containsValue(forCounterNames, varNames.get(i).getValue())) {
          varNames.remove(i);
        }
      }
      checkNames(varNames, forCounterNames);
    }
  }

  private void checkNames(List<Token> varNames, List<Token> forCounterNames) {
    if (constantPattern == null) {
      constantPattern = Pattern.compile(CONSTANT_PATTERN);
    }

    for (Token name : varNames) {
      if (!constantPattern.matcher(name.getValue()).matches()) {
        checkName(name, LOCAL_VAR);
      }
    }

    for (Token name : forCounterNames) {
      if (name.getValue().length() > 1) {
        checkName(name, LOCAL_VAR);
      }
    }
  }

  private static List<Token> getForCounterNames(AstNode suite) {
    List<AstNode> forStatements = suite.getDescendants(PythonGrammar.FOR_STMT);
    List<Token> result = new LinkedList<>();
    for (AstNode forStatement : forStatements) {
      AstNode counters = forStatement.getFirstChild(PythonGrammar.EXPRLIST);
      for (AstNode name : counters.getDescendants(PythonGrammar.NAME)) {
        Token token = name.getToken();
        if (token.getType().equals(GenericTokenType.IDENTIFIER)) {
          result.add(token);
        }
      }
    }
    return result;
  }

  private List<Token> visitParameters(AstNode astNode) {
    List<Token> parameterTokens = new LinkedList<>();
    AstNode varArgList = astNode.getFirstChild(PythonGrammar.TYPEDARGSLIST);
    if (varArgList != null) {
      List<AstNode> funcParameters = varArgList.getDescendants(PythonGrammar.TFPDEF);
      funcParameters.addAll(varArgList.getChildren(PythonGrammar.NAME));
      for (AstNode parameter : funcParameters) {
        Token token = parameter.getToken();
        if (token.getType().equals(GenericTokenType.IDENTIFIER)) {
          parameterTokens.add(token);
          checkName(token, PARAMETER);
        }
      }
    }
    return parameterTokens;
  }

  private void checkName(Token token, String type) {
    String name = token.getValue();
    if (pattern == null) {
      pattern = Pattern.compile(format);
    }
    if (!pattern.matcher(name).matches()) {
      addIssue(token, String.format(MESSAGE, type, name, format));
    }
  }

}
