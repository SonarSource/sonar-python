/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2016 SonarSource SA
 * mailto:contact AT sonarsource DOT com
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
import com.sonar.sslr.api.Grammar;
import com.sonar.sslr.api.Token;
import org.sonar.api.server.rule.RulesDefinition;
import org.sonar.check.Priority;
import org.sonar.check.Rule;
import org.sonar.python.api.PythonGrammar;
import org.sonar.squidbridge.annotations.ActivatedByDefault;
import org.sonar.squidbridge.annotations.SqaleConstantRemediation;
import org.sonar.squidbridge.annotations.SqaleSubCharacteristic;
import org.sonar.squidbridge.checks.SquidCheck;
import org.sonar.sslr.ast.AstSelect;

import java.io.Serializable;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;

@Rule(
    key = DuplicatedMethodFieldNamesCheck.CHECK_KEY,
    priority = Priority.CRITICAL,
    name = "Methods and field names should not differ only by capitalization",
    tags = Tags.CONFUSING
)
@SqaleSubCharacteristic(RulesDefinition.SubCharacteristics.UNDERSTANDABILITY)
@SqaleConstantRemediation("10min")
@ActivatedByDefault
public class DuplicatedMethodFieldNamesCheck extends SquidCheck<Grammar> {

  public static final String CHECK_KEY = "S1845";
  private static final String MESSAGE = "Rename %s \"%s\" to prevent any misunderstanding/clash with %s \"%s\" defined on line %s";

  private static class TokenWithTypeInfo {
    private final Token token;
    private final String type;

    TokenWithTypeInfo(Token token, String type){
      this.token = token;
      this.type = type;
    }

    String getValue(){
      return token.getValue();
    }

    int getLine(){
      return token.getLine();
    }

    String getType(){
      return type;
    }
  }

  private static class LineComparator implements Comparator<TokenWithTypeInfo>, Serializable {
    @Override
    public int compare(TokenWithTypeInfo t1, TokenWithTypeInfo t2) {
      return Integer.compare(t1.getLine(), t2.getLine());
    }
  }

  @Override
  public void init() {
    subscribeTo(PythonGrammar.CLASSDEF);
  }

  @Override
  public void visitNode(AstNode astNode) {
    List<Token> fieldNames = new NewSymbolsAnalyzer().getClassFields(astNode);
    List<Token> methodNames = getFieldNameTokens(astNode);
    lookForDuplications(fieldNames, methodNames);
  }

  private List<Token> getFieldNameTokens(AstNode astNode) {
    List<Token> methodNames = new LinkedList<>();
    AstSelect funcDefSelect = astNode.select()
        .children(PythonGrammar.SUITE)
        .children(PythonGrammar.STATEMENT)
        .children(PythonGrammar.COMPOUND_STMT)
        .children(PythonGrammar.FUNCDEF);
    for (AstNode node : funcDefSelect) {
      methodNames.add(node.getFirstChild(PythonGrammar.FUNCNAME).getToken());
    }
    return methodNames;
  }

  private void lookForDuplications(List<Token> fieldNames, List<Token> methodNames) {
    List<TokenWithTypeInfo> allTokensWithInfo = mergeLists(fieldNames, methodNames);
    Collections.sort(allTokensWithInfo, new LineComparator());
    for (int i = 1; i < allTokensWithInfo.size(); i++){
      for (int j = i-1; j >= 0; j--){
        TokenWithTypeInfo token1 = allTokensWithInfo.get(j);
        TokenWithTypeInfo token2 = allTokensWithInfo.get(i);
        if (differOnlyByCapitalization(token1.getValue(), token2.getValue())){
          getContext().createLineViolation(this, getMessage(token1, token2), token2.getLine());
          break;
        }
      }
    }
  }

  private boolean differOnlyByCapitalization(String name1, String name2) {
    return name1.equalsIgnoreCase(name2) && !name1.equals(name2);
  }

  private List<TokenWithTypeInfo> mergeLists(List<Token> fieldNames, List<Token> methodNames) {
    List<TokenWithTypeInfo> allTokensWithInfo = new LinkedList<>();
    for (Token token : fieldNames){
      allTokensWithInfo.add(new TokenWithTypeInfo(token, "field"));
    }
    for (Token token : methodNames){
      allTokensWithInfo.add(new TokenWithTypeInfo(token, "method"));
    }
    return allTokensWithInfo;
  }

  private String getMessage(TokenWithTypeInfo token1, TokenWithTypeInfo token2) {
    return String.format(MESSAGE, token2.getType(), token2.getValue(), token1.getType(), token1.getValue(), token1.getLine());
  }

}
