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
import java.util.Optional;
import java.util.function.Predicate;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;

public class NoSonarInfoParser {

  private static final String NOQA_PATTERN_REGEX = "^#\\s*noqa(?::\\s*([^\\s]+))?($|\\s.*)";
  private static final String NOSONAR_PATTERN_REGEX = "^#\\s*NOSONAR(?:\\s*\\(([^)]*)\\))?($|\\s.*)";

  private final Pattern noSonarPattern;
  private final Pattern noQaPattern;

  public NoSonarInfoParser() {
    noSonarPattern = Pattern.compile(NOSONAR_PATTERN_REGEX);
    noQaPattern = Pattern.compile(NOQA_PATTERN_REGEX);
  }

  private static boolean isValidNoSonar(String noSonarCommentLine) {
    return noSonarCommentLine.matches(NOSONAR_PATTERN_REGEX);
  }

  private static boolean isValidNoQa(String noSonarCommentLine) {
    return noSonarCommentLine.matches(NOQA_PATTERN_REGEX);
  }

  public Optional<NoSonarLineInfo> parse(String commentLine) {
    var rules = new HashSet<String>();
    if (isValidNoSonar(commentLine)) {
      parseNoSonarRules(commentLine)
        .filter(Predicate.not(String::isEmpty))
        .forEach(rules::add);
    } else if (isValidNoQa(commentLine)) {
      parseNoQaRules(commentLine)
        .filter(Predicate.not(String::isEmpty))
        .forEach(rules::add);
    } else {
      return Optional.empty();
    }
    return Optional.of(new NoSonarLineInfo(rules));
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
