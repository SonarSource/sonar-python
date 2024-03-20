package org.sonar.python.types.v2;

import java.util.List;

public class UnionType implements PythonType {
    List<PythonType> candidates;

    @Override
    public boolean isPrimitive() {
      // TODO Auto-generated method stub
      throw new UnsupportedOperationException("Unimplemented method 'isPrimitive'");
    }

    @Override
    public boolean isUnknown() {
      // TODO Auto-generated method stub
      throw new UnsupportedOperationException("Unimplemented method 'isUnknown'");
    }

    @Override
    public boolean isNone() {
      // TODO Auto-generated method stub
      throw new UnsupportedOperationException("Unimplemented method 'isNone'");
    }
}
