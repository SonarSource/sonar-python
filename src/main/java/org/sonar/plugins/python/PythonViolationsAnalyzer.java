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

import java.util.LinkedList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class PythonViolationsAnalyzer {

  private static final String PYLINT = "pylint -i y -f parseable -r n ";
  private static final Pattern PATTERN = Pattern.compile("([^:]+):([0-9]+): \\[(.*)\\] (.*)");

  public List<Issue> analyze(String path) {
    return parseOutput(Utils.callCommand(PYLINT + " " + path));
  }

  protected List<Issue> parseOutput(List<String> lines) {
    // Parse the output of pylint. Example of the format:
    //
    // complexity/code_chunks.py:62: [W0104, list_compr] Statement seems to have no effect
    // complexity/code_chunks.py:64: [C0111, list_compr_filter] Missing docstring
    // ...

    List<Issue> issues = new LinkedList<Issue>();

    int linenr;
    String filename = null;
    String ruleid = null;
    String objname = null;
    String descr = null;

    if ( !lines.isEmpty()) {
      for (String line : lines) {
        Matcher m = PATTERN.matcher(line);
        if (m.matches() && m.groupCount() == 4) {
          filename = m.group(1);
          linenr = Integer.valueOf(m.group(2));
          String[] parts = m.group(3).split(",");
          ruleid = parts[0].trim();
          if (parts.length == 2) {
            objname = parts[1].trim();
          }

          descr = m.group(4);
          issues.add(new Issue(filename, linenr, ruleid, objname, descr));
        }
      }
    }

    return issues;
  }
}
