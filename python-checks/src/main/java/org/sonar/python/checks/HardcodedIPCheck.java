/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
 * mailto:info AT sonarsource DOT com
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

import com.intellij.lang.ASTNode;
import com.intellij.psi.PsiElement;
import com.jetbrains.python.PyElementTypes;
import com.jetbrains.python.psi.PyStringLiteralExpression;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.python.PythonCheck;
import org.sonar.python.SubscriptionContext;

@Rule(key = "S1313")
public class HardcodedIPCheck extends PythonCheck {

  private static final String IPV4_ALONE = "(?<ipv4>(?:\\d{1,3}\\.){3}\\d{1,3})";

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

  String message = "Make sure using this hardcoded IP address \"%s\" is safe here.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(PyElementTypes.STRING_LITERAL_EXPRESSION, ctx -> {
      PyStringLiteralExpression expression = (PyStringLiteralExpression) ctx.syntaxNode();
      for (ASTNode stringNode : expression.getStringNodes()) {
        checkLiteral(ctx, stringNode.getPsi());
      }
    });
  }

  private void checkLiteral(SubscriptionContext ctx, PsiElement literal) {
    String value = literal.getNode().getText();
    if (value.length() <= 2 || isMultilineString(value)) {
      return;
    }
    String content = value.substring(1, value.length() - 1);
    Matcher matcher = IPV4_URL_REGEX.matcher(content);
    if (matcher.matches()) {
      String ip = matcher.group("ipv4");
      if (isValidIPV4(ip) && !isIPV4Exception(ip)) {
        ctx.addIssue(literal, String.format(message, ip));
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
        .ifPresent(ipv6 -> ctx.addIssue(literal, String.format(message, ipv6)));
    }
  }

  private static boolean isMultilineString(String string) {
    return string.endsWith("'''") || string.endsWith("\"\"\"");
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

  private static boolean isIPV4Exception(String ip) {
    return ip.startsWith("127.")
      || "255.255.255.255".equals(ip)
      || "0.0.0.0".equals(ip)
      || ip.startsWith("2.5.");
  }

  private static boolean isIPV6Exception(String ip) {
    return IPV6_LOOPBACK.matcher(ip).matches() || IPV6_NON_ROUTABLE.matcher(ip).matches();
  }

  private static int getCompressionSeparatorCount(String str) {
    int count = 0;
    for (int i = 0; (i = str.indexOf("::", i)) != -1; i += 2) {
      ++count;
    }
    return count;
  }

}
