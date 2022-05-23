package org.sonar.python.quickfix;

import org.sonar.plugins.python.api.IssueLocation;
import org.sonar.plugins.python.api.LocationInFile;

public class PythonTextEdit {

  public final IssueLocation issueLocation;

  public PythonTextEdit(LocationInFile location, String addition) {
    this.issueLocation = IssueLocation.preciseLocation(location, addition);
  }

  public static PythonTextEdit insertAtPosition(IssueLocation issueLocation, String addition) {
    LocationInFile location = atBeginningOfIssue(issueLocation);
    return new PythonTextEdit(location, addition);
  }

  private static LocationInFile atBeginningOfIssue(IssueLocation issue) {
    return new LocationInFile(issue.fileId(), issue.startLine(), issue.startLineOffset(), issue.startLine(), issue.startLineOffset());
  }

  public String message() {
    return issueLocation.message();
  }
}
