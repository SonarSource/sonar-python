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

import org.sonar.api.resources.Language;
import org.sonar.api.resources.Resource;
import org.sonar.api.resources.Qualifiers;
import org.sonar.api.utils.WildcardPattern;
import org.sonar.api.resources.Directory;

/** A class that represents a Python package in Sonar */
public class PythonPackage extends Directory {

  public PythonPackage(String key) {
    super(key);
  }

  public Language getLanguage() {
    return Python.INSTANCE;
  }

  public String getScope() {
    return Resource.SCOPE_SPACE;
  }

  public String getQualifier() {
    return Qualifiers.PACKAGE;
  }

  public Resource getParent() {
    // Interesting: all the Language implementations i saw so far
    // dont implement nested resources, they just show them in
    // a flat list. We follow this strategy (at least for now)
    return null;
  }

  public boolean matchFilePattern(String antPattern) {
    WildcardPattern matcher = WildcardPattern.create(antPattern, "/");
    return matcher.match(getKey());
  }
}
