from genericExceptionImported import CustomException

def no_issue_on_method_annotated_with_Self_return_type():
    raise CustomException().with_traceback("foo")


def issue():
    raise BaseException()  # Noncompliant
