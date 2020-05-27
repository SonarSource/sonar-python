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

import java.util.Arrays;
import java.util.List;
import org.junit.Test;

import static org.assertj.core.api.Assertions.assertThat;

public class StringFormatTest {

  @Test
  public void test_printf() {
    assertPrintfFormat("Hello %s!", field("s"));
    assertPrintfFormat("Hello %(name)s!", field("name", "s"));
    assertPrintfFormat("%()s", field("", "s"));
    assertPrintfFormat("%%%()s", field("", "s"));
    assertPrintfFormat("%(k+ey\\\\)s", field("k+ey\\\\", "s"));
    assertPrintfFormat("Hello %s, %d!", field("s"), field("d"));
    assertPrintfFormat("%s %%s", field("s"));
  }

  private static void assertPrintfFormat(String input, StringFormat.ReplacementField... fields) {
    List<StringFormat.ReplacementField> actual = StringFormat.createFromPrintfStyle(input).replacementFields();
    List<StringFormat.ReplacementField> expected = Arrays.asList(fields);

    assertThat(actual).hasSize(expected.size());
    for (int i = 0; i < expected.size(); ++i) {
      assertThat(actual.get(i).mappingKey()).isEqualTo(expected.get(i).mappingKey());
      assertThat(actual.get(i).conversionType()).isEqualTo(expected.get(i).conversionType());
    }
  }

  private static StringFormat.ReplacementField field(String conversionType) {
    return new StringFormat.ReplacementField(conversionType, null);
  }

  private static StringFormat.ReplacementField field(String name, String conversionType) {
    return new StringFormat.ReplacementField(conversionType, name);
  }

}
