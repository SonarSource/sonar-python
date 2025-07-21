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
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;

public class NoSonarInfoParser {

  private static final Integer MAX_COMMENT_LENGTH = 50;
  private static final String NOQA_PREFIX_REGEX = "#\\s*noqa([\\s:].*)?";
  private static final String NOQA_PATTERN_REGEX = "^#\\s*noqa(?::\\s*(.+))?(?:[\\s;:].*)?";
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
    var rules = parseNoQaRules(comment);
    return !rules.isEmpty() && rules.stream().anyMatch(r -> r.isBlank() || r.contains(" "));
  }

  private static boolean isValidNoSonar(String noSonarCommentLine) {
    return noSonarCommentLine.matches(NOSONAR_PATTERN_REGEX);
  }

  public static boolean isValidNoQa(String noSonarCommentLine) {
    return noSonarCommentLine.matches(NOQA_PATTERN_REGEX);
  }

  public Optional<NoSonarLineInfo> parse(String commentLine) {
    var rules = new HashSet<String>();
    StringBuilder concatenatedCommentBuilder = new StringBuilder();
    var comments = splitInlineComments(commentLine);
    for (var comment : comments) {
      var noSonarLineInfo = parseComment(comment);

      if (noSonarLineInfo != null) {
        if (noSonarLineInfo.isSuppressedRuleKeysEmpty()) {
          return Optional.of(noSonarLineInfo);
        }
        rules.addAll(noSonarLineInfo.suppressedRuleKeys());
        concatenatedCommentBuilder.append(noSonarLineInfo.comment());
      } else {
        concatenatedCommentBuilder.append(comment.strip());
      }
    }
    if (rules.isEmpty()) {
      return Optional.empty();
    }
    return Optional.of(new NoSonarLineInfo(rules, concatenatedCommentBuilder.toString()));
  }

  @CheckForNull
  private NoSonarLineInfo parseComment(String commentLine) {
    var rules = new HashSet<String>();
    String comment = "";
    if (isValidNoSonar(commentLine)) {
      parseNoSonarRules(commentLine)
        .filter(Predicate.not(String::isEmpty))
        .forEach(rules::add);
      comment = parseNoSonarComment(commentLine);
    } else if (isValidNoQa(commentLine)) {
      parseNoQaRules(commentLine)
        .stream()
        .filter(Predicate.not(String::isEmpty))
        .forEach(rules::add);
      comment = parseNoQaComment(commentLine);
    } else {
      return null;
    }
    return new NoSonarLineInfo(rules, comment);
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

  private String parseNoSonarComment(String noSonarCommentLine) {
    return getTruncatedCommentString(noSonarPattern, noSonarCommentLine).strip();
  }

  private List<String> parseNoQaRules(String noSonarCommentLine) {
    var paramsString = getParamsString(noQaPattern, noSonarCommentLine);
    var paramsList = parseParamsString(paramsString).collect(Collectors.toList());
    if (!paramsList.isEmpty()) {
      // to get the last suppressed rule ID we need to split it to cut the trailing comment text.
      // valid cases are:
      // split by space: ruleID1, ruleID2 some comment
      // or colon with space: ruleID1, ruleID2: some comment
      var lastParamIndex = paramsList.size() - 1;
      var lastParamRaw = paramsList.get(lastParamIndex);
      var lastParam = lastParamRaw.split("(:?\\s)", 0)[0].trim();
      paramsList.set(lastParamIndex, lastParam);
    }
    return paramsList;
  }

  private String parseNoQaComment(String noSonarCommentLine) {
    return getTruncatedCommentString(noQaPattern, noSonarCommentLine).strip();
  }

  private static String getParamsString(Pattern pattern, String noSonarCommentLine) {
    return getPatternGroup(1, pattern, noSonarCommentLine);
  }

  private static String getTruncatedCommentString(Pattern pattern, String noSonarCommentLine) {
    // Actually, group 1 could be the comment group if no rule is specified but we are only interested in comment with exactly one rule suppressed.
    String commentString = getPatternGroup(2, pattern, noSonarCommentLine);
    return commentString.length() > MAX_COMMENT_LENGTH ? commentString.substring(0, MAX_COMMENT_LENGTH) : commentString;
  }

  private static String getPatternGroup(int groupIndex, Pattern pattern, String noSonarCommentLine) {
    return Optional.of(noSonarCommentLine)
      .map(pattern::matcher)
      .filter(Matcher::matches)
      .filter(m -> m.groupCount() >= groupIndex)
      .map(m -> m.group(groupIndex))
      .orElse("");
  }

  private static Stream<String> parseParamsString(String paramsString) {
    if (paramsString.isBlank()) {
      return Stream.of();
    }

    var paramsArray = paramsString.split(",", -1);
    return Stream.of(paramsArray)
      .map(String::trim);
  }
}
