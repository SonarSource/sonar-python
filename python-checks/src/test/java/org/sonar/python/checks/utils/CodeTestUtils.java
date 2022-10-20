package org.sonar.python.checks.utils;

import org.sonar.api.internal.apachecommons.lang.StringUtils;

public class CodeTestUtils {

  public static String code(String... lines) {
    return StringUtils.join(lines, '\n');
  }
}
