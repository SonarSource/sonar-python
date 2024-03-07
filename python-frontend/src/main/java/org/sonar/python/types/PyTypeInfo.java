package org.sonar.python.types;

import com.google.gson.annotations.SerializedName;
import java.util.Optional;
import org.sonar.plugins.python.api.types.InferredType;

public class PyTypeInfo {
  private final String text;
  @SerializedName("start_line")
  private final int startLine;
  @SerializedName("start_col")
  private final int startCol;
  @SerializedName("syntax_role")
  private final String syntaxRole;
  private final PyTypeDetailedInfo type;
  @SerializedName("short_type")
  private final String shortType;

  public PyTypeInfo(String text, int startLine, int startCol, String syntaxRole, PyTypeDetailedInfo type, String shortType) {
    this.text = text;
    this.startLine = startLine;
    this.startCol = startCol;
    this.syntaxRole = syntaxRole;
    this.type = type;
    this.shortType = shortType;
  }

  public String text() {
    return text;
  }

  public int startLine() {
    return startLine;
  }

  public int startCol() {
    return startCol;
  }

  public String syntaxRole() {
    return syntaxRole;
  }

  public PyTypeDetailedInfo type() {
    return type;
  }

  public String shortType() {
    return shortType;
  }

  public InferredType inferredType() {
    return Optional.of(this)
      .map(PyTypeInfo::type)
      .map(PyTypeDetailedInfo::inferredType)
      .orElseGet(InferredTypes::anyType);
  }
}
