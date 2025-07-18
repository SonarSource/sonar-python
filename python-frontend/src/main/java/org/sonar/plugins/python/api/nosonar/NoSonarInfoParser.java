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
package org.sonar.plugins.python.api.nosonar;

import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.function.Predicate;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;

public class NoSonarInfoParser {

  private static final String NOQA_PREFIX_REGEX = "#\\s*noqa([\\s:].*)?";
  private static final String NOQA_PATTERN_REGEX = "^#\\s*noqa(?::\\s*(.+))?.*";
  private static final String NOSONAR_PREFIX_REGEX = "^#\\s*NOSONAR(\\W.*)?";
  private static final String NOSONAR_PATTERN_REGEX = "^#\\s*NOSONAR(?:\\s*\\(([^)]*)\\))?($|\\s.*)";

  private final Pattern noSonarPattern;
  private final Pattern noQaPattern;

  public NoSonarInfoParser() {
    noSonarPattern = Pattern.compile(NOSONAR_PATTERN_REGEX);
    noQaPattern = Pattern.compile(NOQA_PATTERN_REGEX);
  }

  public boolean isInvalidIssueSuppressionComment(String commentsLine) {
    return splitInlineComments(commentsLine)
      .stream()
      .anyMatch(comment -> isInvalidNoSonarComment(comment) || isInvalidNoQaComment(comment));
  }

  private boolean isInvalidNoSonarComment(String comment) {
    if (!Pattern.matches(NOSONAR_PREFIX_REGEX, comment)) {
      return false;
    }
    if (!isValidNoSonar(comment)) {
      return true;
    }
    var rules = parseNoSonarRules(comment).toList();
    return rules.size() > 1 && rules.stream().anyMatch(r -> r.isBlank() || r.contains(" "));
  }

  private boolean isInvalidNoQaComment(String comment) {
    if (!comment.matches(NOQA_PREFIX_REGEX)) {
      return false;
    }
    if (!isValidNoQa(comment)) {
      return true;
    }
    var rules = parseNoQaRules(comment).toList();
    return !rules.isEmpty() && rules.stream().anyMatch(r -> r.isBlank() || r.contains(" "));
  }

  private static boolean isValidNoSonar(String noSonarCommentLine) {
    return noSonarCommentLine.matches(NOSONAR_PATTERN_REGEX);
  }

  private static boolean isValidNoQa(String noSonarCommentLine) {
    return noSonarCommentLine.matches(NOQA_PATTERN_REGEX);
  }

  public Optional<NoSonarLineInfo> parse(String commentLine) {
    var rules = new HashSet<String>();
    var comments = splitInlineComments(commentLine);
    for (var comment : comments) {
      var noSonarLineInfo = parseComment(comment);

      if (noSonarLineInfo != null) {
        if (noSonarLineInfo.isSuppressedRuleKeysEmpty()) {
          return Optional.of(noSonarLineInfo);
        }
        rules.addAll(noSonarLineInfo.suppressedRuleKeys());
      }
    }
    return Optional.of(rules).filter(Predicate.not(HashSet::isEmpty)).map(NoSonarLineInfo::new);
  }

  @CheckForNull
  private NoSonarLineInfo parseComment(String comment) {
    var rules = new HashSet<String>();
    if (isValidNoSonar(comment)) {
      parseNoSonarRules(comment)
        .filter(Predicate.not(String::isEmpty))
        .forEach(rules::add);
    } else if (isValidNoQa(comment)) {
      parseNoQaRules(comment)
        .filter(Predicate.not(String::isEmpty))
        .forEach(rules::add);
    } else {
      return null;
    }
    return new NoSonarLineInfo(rules);
  }

  private static List<String> splitInlineComments(String commentsLine) {
    return Stream.of(commentsLine.split("#"))
      .filter(Predicate.not(String::isBlank))
      .map(s -> "#" + s)
      .toList();
  }

  private Stream<String> parseNoSonarRules(String noSonarCommentLine) {
    var contentInsideParentheses = getParamsString(noSonarPattern, noSonarCommentLine);
    return parseParamsString(contentInsideParentheses);
  }


  private Stream<String> parseNoQaRules(String noSonarCommentLine) {
    var contentInsideParentheses = getParamsString(noQaPattern, noSonarCommentLine);
    return parseParamsString(contentInsideParentheses);
  }

  @CheckForNull
  private static String getParamsString(Pattern pattern, String noSonarCommentLine) {
    return Optional.of(noSonarCommentLine)
      .map(pattern::matcher)
      .filter(Matcher::matches)
      .map(m -> m.group(1))
      .orElse(null);
  }

  private static Stream<String> parseParamsString(@Nullable String paramsString) {
    if (paramsString == null) {
      return Stream.of();
    }

    return Stream.of(paramsString.split(",", -1))
      .map(String::trim);
  }
}
