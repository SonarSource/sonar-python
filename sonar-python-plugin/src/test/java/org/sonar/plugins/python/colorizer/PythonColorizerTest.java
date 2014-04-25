/*
 * SonarQube Python Plugin
 * Copyright (C) 2011 SonarSource and Waleri Enns
 * dev@sonar.codehaus.org
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
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02
 */
package org.sonar.plugins.python.colorizer;

import org.junit.Test;
import org.sonar.colorizer.CodeColorizer;

import java.io.StringReader;

import static org.fest.assertions.Assertions.assertThat;

public class PythonColorizerTest {

  private PythonColorizer pythonColorizer = new PythonColorizer();
  private CodeColorizer codeColorizer = new CodeColorizer(pythonColorizer.getTokenizers());

  private String colorize(String sourceCode) {
    return codeColorizer.toHtml(new StringReader(sourceCode));
  }

  @Test
  public void increase_coverage_for_fun() {
    assertThat(pythonColorizer.getTokenizers()).isSameAs(pythonColorizer.getTokenizers());
  }

  @Test
  public void should_colorize_keywords() {
    assertThat(colorize("False")).contains("<span class=\"k\">False</span>");
  }

  @Test
  public void should_colorize_comments() {
    assertThat(colorize("# comment \n new line")).contains("<span class=\"cd\"># comment </span>");
  }

  @Test
  public void should_colorize_shortstring_literals() {
    assertThat(colorize("\"string\"")).contains("<span class=\"s\">\"string\"</span>");
  }

  @Test
  public void should_colorize_longstring_literals() {
    assertThat(colorize("\"\"\"string\"\"\"")).contains("<span class=\"s\">\"\"\"string\"\"\"</span>");
  }

}
