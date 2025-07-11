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
package org.sonar.plugins.python.nosonar;

import com.sonar.sslr.api.GenericTokenType;
import java.util.ArrayDeque;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.Nullable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.api.scanner.ScannerSide;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Trivia;
import org.sonarsource.api.sonarlint.SonarLintSide;

@ScannerSide
@SonarLintSide
public class NoSonarLineInfoCollector {

  private static final Logger LOG = LoggerFactory.getLogger(NoSonarLineInfoCollector.class);

  public static final String NOSONAR_PATTERN_REGEX = "^#\\s*NOSONAR(?:\\(([^)]*)\\))?$";

  private final Pattern noSonarPattern;
  private final Map<String, Map<Integer, NoSonarLineInfo>> componentKeyToNoSonarLineInfoMap;

  public NoSonarLineInfoCollector() {
    this.noSonarPattern = Pattern.compile(NOSONAR_PATTERN_REGEX);
    this.componentKeyToNoSonarLineInfoMap = new ConcurrentHashMap<>();
  }

  public void collect(String componentKey, @Nullable FileInput fileInput) {
    if (fileInput == null) {
      return;
    }
    var result = scan(fileInput);
    if (!result.isEmpty()) {
      LOG.debug("Component with key: {} has {} NOSONAR comments: {}", componentKey, result.size(), result);
      componentKeyToNoSonarLineInfoMap.put(componentKey, result);
    }
  }

  public Map<Integer, NoSonarLineInfo> get(String key) {
    return componentKeyToNoSonarLineInfoMap.getOrDefault(key, new ConcurrentHashMap<>());
  }

  public Set<Integer> getLinesWithEmptyNoSonar(String key) {
    return get(key)
      .values()
      .stream()
      .filter(NoSonarLineInfo::isSuppressedRuleKeysEmpty)
      .map(NoSonarLineInfo::line)
      .collect(Collectors.toSet());
  }

  private Map<Integer, NoSonarLineInfo> scan(Tree element) {
    var result = new ConcurrentHashMap<Integer, NoSonarLineInfo>();
    var stack = new ArrayDeque<Tree>();
    stack.push(element);
    while (!stack.isEmpty()) {
      var currentElement = stack.pop();
      if (currentElement instanceof Token token) {
        visitToken(token)
          .forEach(info -> result.put(info.line(), info));
      }

      currentElement.children()
        .stream()
        .filter(Objects::nonNull)
        .forEach(stack::push);
    }
    return result;
  }

  private List<NoSonarLineInfo> visitToken(Token token) {
    if (token.type() == GenericTokenType.EOF) {
      return List.of();
    }

    return token.trivia()
      .stream()
      .flatMap(trivia -> visitComment(trivia, token))
      .filter(Objects::nonNull)
      .toList();
  }

  private Stream<NoSonarLineInfo> visitComment(Trivia trivia, Token parentToken) {
    String commentLine = getContents(trivia.token().value());
    int line = trivia.token().line();
    if (containsNoSonarComment(commentLine)) {
      return parse(line, commentLine, parentToken);
    }
    return Stream.of();
  }

  private static String getContents(String comment) {
    // Comment always starts with "#"
    return comment.substring(comment.indexOf('#'));
  }

  private static boolean containsNoSonarComment(String commentLine) {
    return commentLine.trim().contains("NOSONAR");
  }

  private Stream<NoSonarLineInfo> parse(int line, String noSonarCommentLine, Token parentToken) {
    var rules = parseNoSonarRules(noSonarCommentLine);
    var lines = new HashSet<Integer>();
    lines.add(line);

    if (parentToken.parent() instanceof ExpressionStatement expressionStatement
        && !expressionStatement.expressions().isEmpty()
        && expressionStatement.expressions().get(0) instanceof StringLiteral stringLiteral) {
      var firstLine = stringLiteral.firstToken().line();
      for (int i = firstLine; i < line + 1; i++) {
        lines.add(i);
      }
    }

    return lines.stream().map(l -> new NoSonarLineInfo(l, rules));
  }

  private Set<String> parseNoSonarRules(String noSonarCommentLine) {
    var rules = new HashSet<String>();

    var matcher = noSonarPattern.matcher(noSonarCommentLine);

    // Check if the entire string matches the pattern
    if (matcher.matches()) {
      var contentInsideParentheses = matcher.group(1);

      if (contentInsideParentheses != null && !contentInsideParentheses.isEmpty()) {
        var ruleArray = contentInsideParentheses.split(",");
        for (var rule : ruleArray) {
          var trimmedRule = rule.trim();
          // Add only non-empty rules
          if (!trimmedRule.isEmpty()) {
            rules.add(trimmedRule);
          }
        }
      }
    }
    return rules;
  }

}
