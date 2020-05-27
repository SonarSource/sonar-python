/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2020 SonarSource SA
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

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;

public class StringFormat {

  public static class ReplacementField {
    private String conversionType;
    private String mappingKey;

    public ReplacementField(String conversionType, @Nullable String mappingKey) {
      this.conversionType = conversionType;
      this.mappingKey = mappingKey;
    }

    public String conversionType() {
      return conversionType;
    }

    @CheckForNull
    public String mappingKey() {
      return mappingKey;
    }
  }

  private List<ReplacementField> replacementFields;

  private StringFormat(List<ReplacementField> replacementFields) {
    this.replacementFields = replacementFields;
  }

  public List<ReplacementField> replacementFields() {
    return this.replacementFields;
  }

  public int numExpectedArguments() {
    return this.replacementFields.size();
  }

  // Format -> '%'[MapKey][Flag][Width]['.'Precision][Length]Type
  // MapKey -> '(' Str ')'
  // Flag -> '#' | '0' | '-' | ' ' | '+' | '-'
  // Length -> 'h' | 'H' | 'L'
  // Width -> '*' | Number
  // Precision -> '*' | Number
  private static final String PRINTF_MAPKEY_PATTERN = "(?:\\((.*?)\\))?";
  private static final String PRINTF_CONVERSION_TYPE_PATTERN = "([A-Za-z]|%)";

  private static final Pattern PRTINF_PARAM_PATTERN = Pattern.compile(
    "%" + PRINTF_MAPKEY_PATTERN + PRINTF_CONVERSION_TYPE_PATTERN
  );

  public static StringFormat createFromPrintfStyle(String input) {
    List<ReplacementField> result = new ArrayList<>();
    Matcher matcher = PRTINF_PARAM_PATTERN.matcher(input);

    while (matcher.find()) {
      String mapKey = matcher.group(1);
      String conversionType = matcher.group(2);

      if (conversionType.equals("%")) {
        // If the conversion type is '%', we are dealing with a '%%'
        continue;
      }
      result.add(new ReplacementField(conversionType, mapKey));
    }

    return new StringFormat(result);
  }
}
