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

class Issue {

  public final String filename;
  public final int line;
  public final String ruleId;
  public final String objname;
  public final String descr;

  Issue(String filename, int line, String ruleId, String objname, String descr) {
    this.filename = filename;
    this.line = line;
    this.ruleId = ruleId;
    this.objname = objname;
    this.descr = descr;
  }

  @Override
  public String toString() {
    return "(" + filename + ", " + line + ", " + ruleId + ", " + objname + ", " + descr + ")";
  }
}
