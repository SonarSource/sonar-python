/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python.checks;

import java.util.Arrays;
import java.util.Locale;
import java.util.Optional;
import java.util.stream.Collectors;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.python.quickfix.TextEditUtils;

final class NamingConventionQuickFixUtils {

  private NamingConventionQuickFixUtils() {
  }

  static Optional<PythonQuickFix> renameToSnakeCase(Name name) {
    String newName = toSnakeCase(name.name());
    return rename(name, newName);
  }

  static Optional<PythonQuickFix> renameToPascalCase(Name name) {
    String newName = toPascalCase(name.name());
    return rename(name, newName);
  }

  private static Optional<PythonQuickFix> rename(Name name, String newName) {
    if (newName.equals(name.name()) || name.symbol() == null) {
      return Optional.empty();
    }
    return Optional.of(PythonQuickFix.newQuickFix(String.format("Rename '%s' to '%s'", name.name(), newName))
      .addTextEdit(TextEditUtils.renameAllUsages(name, newName))
      .build());
  }

  private static String toSnakeCase(String currentName) {
    String prefix = leadingUnderscorePrefix(currentName);
    String convertedBody = splitWords(stripLeadingUnderscores(currentName)).stream()
      .map(word -> word.toLowerCase(Locale.ROOT))
      .collect(Collectors.joining("_"));

    if (convertedBody.isEmpty()) {
      convertedBody = "renamed";
    }
    if (Character.isDigit(convertedBody.charAt(0))) {
      return prefix + "_" + convertedBody;
    }
    return prefix + convertedBody;
  }

  private static String toPascalCase(String currentName) {
    String prefix = leadingUnderscorePrefix(currentName);
    String convertedBody = splitWords(stripLeadingUnderscores(currentName)).stream()
      .map(NamingConventionQuickFixUtils::capitalize)
      .collect(Collectors.joining());

    if (convertedBody.isEmpty()) {
      convertedBody = "Renamed";
    }
    if (Character.isDigit(convertedBody.charAt(0))) {
      convertedBody = "Class" + convertedBody;
    }
    return prefix + convertedBody;
  }

  private static String capitalize(String word) {
    if (word.isEmpty()) {
      return word;
    }
    String lowerCaseWord = word.toLowerCase(Locale.ROOT);
    return Character.toUpperCase(lowerCaseWord.charAt(0)) + lowerCaseWord.substring(1);
  }

  private static String leadingUnderscorePrefix(String name) {
    return name.startsWith("_") ? "_" : "";
  }

  private static String stripLeadingUnderscores(String name) {
    return name.replaceFirst("^_+", "");
  }

  private static java.util.List<String> splitWords(String name) {
    String normalized = name
      .replaceAll("([a-z0-9])([A-Z])", "$1_$2")
      .replaceAll("([A-Z]+)([A-Z][a-z])", "$1_$2")
      .replaceAll("[^A-Za-z0-9]+", "_")
      .replaceAll("_+", "_")
      .replaceAll("^_+|_+$", "");

    if (normalized.isEmpty()) {
      return java.util.List.of();
    }
    return Arrays.stream(normalized.split("_"))
      .filter(part -> !part.isEmpty())
      .toList();
  }
}
