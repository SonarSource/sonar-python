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

import java.util.ArrayDeque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.api.scanner.ScannerSide;
import org.sonar.plugins.python.api.nosonar.NoSonarInfoParser;
import org.sonar.plugins.python.api.nosonar.NoSonarLineInfo;
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

  private final NoSonarInfoParser parser;
  private final Map<String, Map<Integer, NoSonarLineInfo>> componentKeyToNoSonarLineInfoMap;

  public NoSonarLineInfoCollector() {
    parser = new NoSonarInfoParser();
    this.componentKeyToNoSonarLineInfoMap = new ConcurrentHashMap<>();
  }

  public void collect(String componentKey, @Nullable FileInput fileInput) {
    if (fileInput == null) {
      return;
    }
    var result = scan(fileInput);
    if (!result.isEmpty()) {
      LOG.debug("File with key: {} has {} NOSONAR comments: {}", componentKey, result.size(), result);
      componentKeyToNoSonarLineInfoMap.put(componentKey, result);
    }
  }

  public Map<Integer, NoSonarLineInfo> get(String key) {
    return componentKeyToNoSonarLineInfoMap.getOrDefault(key, new ConcurrentHashMap<>());
  }

  public Set<Integer> getLinesWithEmptyNoSonar(String key) {
    return get(key)
      .entrySet()
      .stream()
      .filter(entry -> entry.getValue().isSuppressedRuleKeysEmpty())
      .map(Map.Entry::getKey)
      .collect(Collectors.toSet());
  }

  private Map<Integer, NoSonarLineInfo> scan(Tree element) {
    var result = new ConcurrentHashMap<Integer, NoSonarLineInfo>();
    var stack = new ArrayDeque<Tree>();
    stack.push(element);
    while (!stack.isEmpty()) {
      var currentElement = stack.pop();
      if (currentElement instanceof Token token) {
        var tokenResults = visitToken(token);
        result.putAll(tokenResults);
      }

      currentElement.children()
        .stream()
        .filter(Objects::nonNull)
        .forEach(stack::push);
    }
    return result;
  }

  private Map<Integer, NoSonarLineInfo> visitToken(Token token) {
    var result = new HashMap<Integer, NoSonarLineInfo>();
    for (var trivia : token.trivia()) {
      parseComment(trivia)
        .ifPresent(info -> {
          var commentLine = trivia.token().line();
          calculateLines(commentLine, token).forEach(line -> result.put(line, info));
        });
    }

    return result;
  }

  private Optional<NoSonarLineInfo> parseComment(Trivia trivia) {
    var commentString = getContents(trivia.token().value());
    return parser.parse(commentString);
  }

  private static String getContents(String comment) {
    // Comment always starts with "#"
    return comment.substring(comment.indexOf('#'));
  }

  private static Set<Integer> calculateLines(int commentLine, Token parentToken) {
    var lines = new HashSet<Integer>();
    lines.add(commentLine);

    if (parentToken.parent() instanceof ExpressionStatement expressionStatement
        && !expressionStatement.expressions().isEmpty()
        && expressionStatement.expressions().get(0) instanceof StringLiteral stringLiteral) {
      var firstLine = stringLiteral.firstToken().line();
      for (int i = firstLine; i < commentLine + 1; i++) {
        lines.add(i);
      }
    }
    return lines;
  }

  public String getSuppressedRuleIds(){
    return componentKeyToNoSonarLineInfoMap.values().stream()
      .flatMap(map -> map.values().stream())
      .flatMap(noSonarLineInfo -> noSonarLineInfo.suppressedRuleKeys().stream())
      .distinct()
      .sorted()
      .collect(Collectors.joining(","));
  }
}
