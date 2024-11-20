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
package org.sonar.python.parser;

import com.sonar.sslr.api.AstNode;
import com.sonar.sslr.api.GenericTokenType;
import com.sonar.sslr.api.TokenType;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.api.PythonTokenType;
import org.sonar.python.tree.TreeUtils;
import org.sonar.sslr.grammar.GrammarRuleKey;

import static org.assertj.core.api.Assertions.assertThat;

public abstract class RuleTest {

  protected PythonParser p = PythonParser.create();
  protected PythonParser ip = PythonParser.createIPythonParser();

  protected void setRootRule(GrammarRuleKey ruleKey) {
    p.setRootRule(p.getGrammar().rule(ruleKey));
    ip.setRootRule(ip.getGrammar().rule(ruleKey));
  }

  protected  <T extends Tree> T parse(String code, Function<AstNode, T> func) {
    return parse(p, code, func);
  }

  protected  <T extends Tree> T parseIPython(String code, Function<AstNode, T> func) {
    return parse(ip, code, func);
  }

  protected  <T extends Tree> T parse(PythonParser parser, String code, Function<AstNode, T> func) {
    AstNode ast = parser.parse(code);
    T tree = func.apply(ast);
    // ensure every visit method of base tree visitor is called without errors
    BaseTreeVisitor visitor = new BaseTreeVisitor();
    tree.accept(visitor);
    List<TokenType> ptt = Arrays.asList(PythonTokenType.NEWLINE, PythonTokenType.DEDENT, PythonTokenType.INDENT, GenericTokenType.EOF);
    List<Token> tokenList = TreeUtils.tokens(tree);

    String tokens = tokenList.stream().filter(t -> !ptt.contains(t.type())).map(token -> {
      if((token.type() == PythonTokenType.STRING) || (token.type() == PythonTokenType.FSTRING_MIDDLE)) {
        return token.value().replaceAll("\n", "").replaceAll(" ", "");
      }
      return token.value();
    }).collect(Collectors.joining(""));
    String originalCode = code.replaceAll("#.*\\n", "")
      .replaceAll("\\\\?\\n", "")
      .replaceAll(" ", "")
      ;
    assertThat(tokens).isEqualTo(originalCode);
    return tree;
  }
}
