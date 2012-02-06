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

import java.util.HashMap;
import java.util.Map;

import org.sonar.api.resources.Directory;
import org.sonar.api.resources.Language;
import org.sonar.api.resources.Qualifiers;
import org.sonar.api.resources.Resource;
import org.sonar.api.utils.WildcardPattern;
import org.apache.commons.lang.StringUtils;

/** A class that represents a Python package in Sonar */
public final class PythonPackage extends Directory {
  private static Map<String, PythonPackage> packages
    = new HashMap<String, PythonPackage>();
    
  private PythonPackage(String key) {
    super(key);
  }

  public static PythonPackage create(String key) {
    String packageKey = StringUtils.replace(key, "/", ".");
    PythonPackage pypackage = packages.get(packageKey);
    if (pypackage == null) {
      pypackage = new PythonPackage(packageKey);
      packages.put(packageKey, pypackage);
    }
    
    return pypackage;
  }
  
  @Override
  public Language getLanguage() {
    return Python.INSTANCE;
  }

  @Override
  public String getScope() {
    return Resource.SCOPE_SPACE;
  }

  @Override
  public String getQualifier() {
    return Qualifiers.PACKAGE;
  }

  @Override
  public Resource getParent() {
    // Interesting: all the Language implementations i saw so far
    // dont implement nested resources, they just show them in
    // a flat list. We follow this strategy (at least for now)
    return null;
  }

  public boolean matchFilePattern(String antPattern) {
    String patternWithoutFileSuffix = StringUtils.substringBeforeLast(antPattern, ".");
    WildcardPattern matcher = WildcardPattern.create(patternWithoutFileSuffix, ".");
    return matcher.match(getKey());
  }
}
