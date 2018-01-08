/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2018 SonarSource SA
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
package org.sonar.python.toolkit;

import java.util.Arrays;
import org.sonar.colorizer.KeywordsTokenizer;
import org.sonar.colorizer.Tokenizer;
import org.sonar.python.api.PythonKeyword;
import org.sonar.sslr.toolkit.Toolkit;

import java.util.List;

public final class PythonToolkit {

  private PythonToolkit() {
  }

  public static void main(String[] args) {
    Toolkit toolkit = new Toolkit("SSLR :: Python :: Toolkit", new PythonConfigurationModel());
    toolkit.run();
  }

  public static List<Tokenizer> getPythonTokenizers() {
    return Arrays.asList(
      (Tokenizer) new KeywordsTokenizer("<span class=\"k\">", "</span>", PythonKeyword.keywordValues()));
  }

}
