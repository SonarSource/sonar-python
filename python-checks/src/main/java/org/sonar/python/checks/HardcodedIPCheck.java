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
package org.sonar.python.checks;

import java.util.Arrays;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.checks.utils.Expressions;

@Rule(key = "S1313")
public class HardcodedIPCheck extends PythonSubscriptionCheck {

  private static final String IPV4_ALONE = "(?<ipv4>(?:\\d{1,3}\\.){3}\\d{1,3})";
  private static final Pattern LOCAL_IPV4_MAPPED_TO_IPV6 = Pattern.compile("::[f,F]{4}(:0)?:127\\.(\\d{1,3}\\.){2}\\d{1,3}");

  private static final String IPV6_NO_PREFIX_COMPRESSION = "(\\p{XDigit}{1,4}::?){1,7}\\p{XDigit}{1,4}(::)?";
  private static final String IPV6_PREFIX_COMPRESSION = "::((\\p{XDigit}{1,4}:){0,6}\\p{XDigit}{1,4})?";
  private static final String IPV6_ALONE = ("(?<ipv6>(" + IPV6_NO_PREFIX_COMPRESSION + "|" + IPV6_PREFIX_COMPRESSION + ")??(:?" + IPV4_ALONE + ")?" + ")");
  private static final String IPV6_URL = "([^\\d.]*/)?\\[" + IPV6_ALONE + "]((:\\d{1,5})?(?!\\d|\\.))(/.*)?";

  private static final Pattern IPV4_URL_REGEX = Pattern.compile("([^\\d.]*/)?" + IPV4_ALONE + "((:\\d{1,5})?(?!\\d|\\.))(/.*)?");
  private static final List<Pattern> IPV6_REGEX_LIST = Arrays.asList(
    Pattern.compile(IPV6_ALONE),
    Pattern.compile(IPV6_URL));

  private static final Pattern IPV6_LOOPBACK = Pattern.compile("[0:]++0*+1");
  private static final Pattern IPV6_NON_ROUTABLE = Pattern.compile("[0:]++");

  private static final List<String> RESERVED_IP_PREFIXES = List.of(
    "192.0.2.",
    "198.51.100.",
    "203.0.113.",
    "2001:db8:",
    "127.",
    "2.5.",
    "255.255.255.255",
    "0.0.0.0"
  );

  String message = "Make sure using this hardcoded IP address \"%s\" is safe here.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.STRING_LITERAL, ctx -> {
      StringLiteral stringLiteral = (StringLiteral) ctx.syntaxNode();
      if (isMultilineString(stringLiteral)) {
        return;
      }
      String content = Expressions.unescape(stringLiteral);
      Matcher matcher = IPV4_URL_REGEX.matcher(content);
      if (matcher.matches()) {
        String ip = matcher.group("ipv4");
        if (isValidIPV4(ip) && !isReservedIP(ip)) {
          ctx.addIssue(stringLiteral, String.format(message, ip));
        }
      } else {
        IPV6_REGEX_LIST.stream()
          .map(pattern -> pattern.matcher(content))
          .filter(Matcher::matches)
          .findFirst()
          .map(match -> {
            String ipv6 = match.group("ipv6");
            String ipv4 = match.group("ipv4");
            return isValidIPV6(ipv6, ipv4) && !isIPV6Exception(ipv6) ? ipv6 : null;
          })
          .ifPresent(ipv6 -> ctx.addIssue(stringLiteral, String.format(message, ipv6)));
      }
    });
  }

  private static boolean isMultilineString(StringLiteral pyStringLiteralTree) {
    return pyStringLiteralTree.stringElements().size() > 1;
  }

  private static boolean isValidIPV4(String ip) {
    String[] numbersAsStrings = ip.split("\\.");
    return Arrays.stream(numbersAsStrings).noneMatch(value -> Integer.valueOf(value) > 255);
  }

  private static boolean isValidIPV6(String ipv6, @Nullable String ipv4) {
    String[] split = ipv6.split("::?");
    int partCount = split.length;
    int compressionSeparatorCount = getCompressionSeparatorCount(ipv6);
    boolean validUncompressed;
    boolean validCompressed;
    if (ipv4 != null) {
      boolean hasValidIPV4 = isValidIPV4(ipv4);
      validUncompressed = hasValidIPV4 && compressionSeparatorCount == 0 && partCount == 7;
      validCompressed = hasValidIPV4 && compressionSeparatorCount == 1 && partCount <= 6;
    } else {
      validUncompressed = compressionSeparatorCount == 0 && partCount == 8;
      validCompressed = compressionSeparatorCount == 1 && partCount <= 7;
    }

    return validUncompressed || validCompressed;
  }

  private static boolean isIPV6Exception(String ip) {
    return isReservedIP(ip) || 
      IPV6_LOOPBACK.matcher(ip).matches() || 
      IPV6_NON_ROUTABLE.matcher(ip).matches() || 
      LOCAL_IPV4_MAPPED_TO_IPV6.matcher(ip).matches(); 
  }


  private static int getCompressionSeparatorCount(String str) {
    int count = 0;
    for (int i = 0; (i = str.indexOf("::", i)) != -1; i += 2) {
      ++count;
    }
    return count;
  }

  private static boolean isReservedIP(String ip) {
    return RESERVED_IP_PREFIXES.stream().anyMatch(ip::startsWith);
  }

}
