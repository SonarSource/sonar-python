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
package org.sonar.python;

import java.nio.charset.Charset;

public class PythonConfiguration {

  private Charset charset;
  private boolean stopSquidOnException = false;
  private boolean ignoreHeaderComments;

  public PythonConfiguration(Charset charset) {
    this.charset = charset;
  }

  // TODO this method seems to be unused, should we create plugin property to ignore header comments?
  public void setIgnoreHeaderComments(boolean ignoreHeaderComments) {
    this.ignoreHeaderComments = ignoreHeaderComments;
  }

  public boolean getIgnoreHeaderComments() {
    return ignoreHeaderComments;
  }

  public Charset getCharset() {
    return charset;
  }

  public void setStopSquidOnException(boolean stopSquidOnException) {
    this.stopSquidOnException = stopSquidOnException;
  }

  public boolean stopSquidOnException() {
    return stopSquidOnException;
  }

}
