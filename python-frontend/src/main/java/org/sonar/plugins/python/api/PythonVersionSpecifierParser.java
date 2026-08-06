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
package org.sonar.plugins.python.api;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.sonar.plugins.python.api.PythonVersionUtils.Version;

final class PythonVersionSpecifierParser {

  private static final Pattern VERSION_SPECIFIER_PATTERN = Pattern.compile(
    "^(<=|>=|==|!=|~=|<|>|\\^)\\s*(\\d+)(?:\\.(\\d+))?(?:\\.(\\d+))?(\\.\\*)?$");

  private PythonVersionSpecifierParser() {
  }

  static boolean containsVersionSpecifier(String[] versions) {
    return Arrays.stream(versions)
      .map(String::trim)
      .filter(value -> !value.isEmpty())
      .map(value -> value.charAt(0))
      .anyMatch(character -> "<>=!~^".indexOf(character) >= 0);
  }

  static Optional<Set<Version>> parse(String[] values) {
    List<VersionSpecifier> specifiers = new ArrayList<>();
    for (String value : values) {
      Optional<VersionSpecifier> specifier = VersionSpecifier.parse(value.trim());
      if (specifier.isEmpty()) {
        return Optional.empty();
      }
      specifiers.add(specifier.get());
    }

    Set<Version> matchingVersions = EnumSet.noneOf(Version.class);
    for (Version version : Version.values()) {
      if (matchesAll(specifiers, VersionLine.from(version))) {
        matchingVersions.add(version);
      }
    }
    return Optional.of(matchingVersions);
  }

  private static boolean matchesAll(List<VersionSpecifier> specifiers, VersionLine version) {
    return specifiers.stream().allMatch(specifier -> specifier.matches(version));
  }

  private enum SpecifierOperator {
    LESS_THAN,
    LESS_THAN_OR_EQUAL,
    GREATER_THAN,
    GREATER_THAN_OR_EQUAL,
    EQUAL,
    NOT_EQUAL,
    COMPATIBLE,
    CARET;

    private static Optional<SpecifierOperator> fromString(String value) {
      return Optional.ofNullable(switch (value) {
        case "<" -> LESS_THAN;
        case "<=" -> LESS_THAN_OR_EQUAL;
        case ">" -> GREATER_THAN;
        case ">=" -> GREATER_THAN_OR_EQUAL;
        case "==" -> EQUAL;
        case "!=" -> NOT_EQUAL;
        case "~=" -> COMPATIBLE;
        case "^" -> CARET;
        default -> null;
      });
    }
  }

  private record VersionSpecifier(
    SpecifierOperator operator,
    VersionLine version,
    int precision,
    boolean wildcard,
    Optional<VersionLine> upperBound) {

    private static Optional<VersionSpecifier> parse(String value) {
      Matcher matcher = VERSION_SPECIFIER_PATTERN.matcher(value);
      if (!matcher.matches()) {
        return Optional.empty();
      }
      Optional<SpecifierOperator> parsedOperator = SpecifierOperator.fromString(matcher.group(1));
      if (parsedOperator.isEmpty()) {
        return Optional.empty();
      }
      SpecifierOperator operator = parsedOperator.get();
      try {
        int major = Integer.parseInt(matcher.group(2));
        int minor = matcher.group(3) == null ? 0 : Integer.parseInt(matcher.group(3));
        int precision;
        if (matcher.group(4) != null) {
          precision = 3;
        } else if (matcher.group(3) != null) {
          precision = 2;
        } else {
          precision = 1;
        }
        boolean wildcard = matcher.group(5) != null;
        boolean wildcardWithUnsupportedOperator = wildcard
          && operator != SpecifierOperator.EQUAL
          && operator != SpecifierOperator.NOT_EQUAL;
        boolean compatibleReleaseWithoutMinor = operator == SpecifierOperator.COMPATIBLE
          && precision < 2;
        if (wildcardWithUnsupportedOperator || compatibleReleaseWithoutMinor) {
          return Optional.empty();
        }
        VersionLine version = new VersionLine(major, minor);
        Optional<VersionLine> upperBound = upperBound(operator, major, minor, precision);
        return Optional.of(new VersionSpecifier(operator, version, precision, wildcard, upperBound));
      } catch (NumberFormatException e) {
        return Optional.empty();
      }
    }

    private static Optional<VersionLine> upperBound(SpecifierOperator operator, int major, int minor, int precision) {
      if (operator == SpecifierOperator.COMPATIBLE) {
        // Precision 1 is rejected during parsing, so only minor and patch precision remain.
        return Optional.of(precision == 2
          ? new VersionLine(major + 1, 0)
          : new VersionLine(major, minor + 1));
      }
      if (operator == SpecifierOperator.CARET) {
        return Optional.of(new VersionLine(major + 1, 0));
      }
      return Optional.empty();
    }

    private boolean matches(VersionLine candidate) {
      int comparison = candidate.compareTo(version);
      return switch (operator) {
        case LESS_THAN -> comparison < 0;
        case LESS_THAN_OR_EQUAL -> comparison <= 0;
        case GREATER_THAN -> comparison > 0;
        case GREATER_THAN_OR_EQUAL -> comparison >= 0;
        case EQUAL -> matchesEquality(candidate);
        case NOT_EQUAL -> !matchesEquality(candidate);
        case COMPATIBLE, CARET -> comparison >= 0 && candidate.compareTo(upperBound.orElseThrow()) < 0;
      };
    }

    private boolean matchesEquality(VersionLine candidate) {
      if (wildcard && precision == 1) {
        return candidate.major() == version.major();
      }
      return candidate.major() == version.major() && candidate.minor() == version.minor();
    }
  }

  private record VersionLine(int major, int minor) implements Comparable<VersionLine> {

    private static VersionLine from(Version version) {
      return new VersionLine(version.major(), version.minor());
    }

    @Override
    public int compareTo(VersionLine other) {
      int majorComparison = Integer.compare(major, other.major);
      return majorComparison == 0 ? Integer.compare(minor, other.minor) : majorComparison;
    }
  }
}
