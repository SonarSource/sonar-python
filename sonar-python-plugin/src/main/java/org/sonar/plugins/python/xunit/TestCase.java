/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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
package org.sonar.plugins.python.xunit;

import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.apache.commons.lang.StringEscapeUtils;

/**
 * Represents a unit test case. Has a couple of data items like name, status, time etc. associated. Reports testcase details in
 * sonar-conform XML
 */
public class TestCase {

  public static final String STATUS_OK = "ok";
  public static final String STATUS_ERROR = "error";
  public static final String STATUS_FAILURE = "failure";
  public static final String STATUS_SKIPPED = "skipped";

  private final String name;
  private final String status;
  private final String stackTrace;
  private final String errorMessage;
  private final int time;
  private final String file;
  private final String testClassname;

  /**
   * Constructs a testcase instance out of following parameters
   *
   * @param name
   *          The name of this testcase
   * @param time
   *          The execution time in milliseconds
   * @param status
   *          The execution status of the testcase
   * @param stack
   *          The stack trace occurred while executing of this testcase; pass "" if the testcase passed/skipped.
   * @param msg
   *          The error message associated with this testcase of the execution was erroneous; pass "" if not.
   * @param file
   *          The optional file to which this test case applies.
   * @param testClassname
   *          The classname of the test.
   */
  public TestCase(String name, int time, String status, String stack, String msg, @Nullable String file, @Nullable String testClassname) {
    this.name = name;
    this.time = time;
    this.stackTrace = stack;
    this.errorMessage = msg;
    this.status = status;
    this.file = file;
    this.testClassname = testClassname;
  }

  /**
   * Returns true if this testcase is an error, false otherwise
   */
  public boolean isError() {
    return STATUS_ERROR.equals(status);
  }

  /**
   * Returns true if this testcase is a failure, false otherwise
   */
  public boolean isFailure() {
    return STATUS_FAILURE.equals(status);
  }

  /**
   * Returns true if this testcase has been skipped, failure, false otherwise
   */
  public boolean isSkipped() {
    return STATUS_SKIPPED.equals(status);
  }

  public int getTime() {
    return time;
  }

  @CheckForNull
  public String getFile() {
    return file;
  }

  @CheckForNull
  public String getTestClassname() {
    return testClassname;
  }

  /**
   * Returns execution details as sonar-conform XML
   */
  public String getDetails() {
    StringBuilder details = new StringBuilder();
    details.append("<testcase status=\"").append(status).append("\" time=\"").append(time).append("\" name=\"").append(name).append("\"");
    if (isError() || isFailure()) {
      details.append(">").append(isError() ? "<error message=\"" : "<failure message=\"").append(StringEscapeUtils.escapeXml(errorMessage))
          .append("\">").append("<![CDATA[").append(StringEscapeUtils.escapeXml(stackTrace)).append("]]>")
          .append(isError() ? "</error>" : "</failure>").append("</testcase>");
    } else {
      details.append("/>");
    }

    return details.toString();
  }
}
