/*
 * Sonar Python Plugin
 * Copyright (C) 2011 Waleri Enns
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

package org.sonar.plugins.python;

import java.util.ArrayList;
import java.util.List;

import org.sonar.api.web.CodeColorizerFormat;
import org.sonar.colorizer.KeywordsTokenizer;
import org.sonar.colorizer.StringTokenizer;
import org.sonar.colorizer.Tokenizer;

public final class PythonColorizer extends CodeColorizerFormat {
  private List<Tokenizer> tokenizers;

  public PythonColorizer() {
    super(Python.KEY);
  }

  @Override
  public List<Tokenizer> getTokenizers() {
    if (tokenizers == null) {
      tokenizers = new ArrayList<Tokenizer>();
      tokenizers.add(new KeywordsTokenizer("<span class=\"k\">", "</span>", Python.KEYWORDS));
      tokenizers.add(new StringTokenizer("<span class=\"s\">", "</span>"));
      tokenizers.add(new PythonDocTokenizer("<span class=\"cd\">", "</span>"));
      tokenizers.add(new PythonDocStringTokenizer("<span class=\"s\">", "</span>"));

      // the following tokenizers don't work, for some reason.
      // tokens.add(new KeywordsTokenizer("<span class=\"c\">", "</span>", CONSTANTS));
      // tokens.add(new KeywordsTokenizer("<span class=\"h\">", "</span>", BUILTINS));

      // TODO:
      // use regexptokenizer to match functions or classes
    }
    return tokenizers;
  }
}
