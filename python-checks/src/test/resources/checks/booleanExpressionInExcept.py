try:
    foo()
except ValueError:
    pass
except ValueError or TypeError:  # Noncompliant
    pass
except ValueError and TypeError:  # Noncompliant
    pass
except (ValueError or TypeError) as exception:  # Noncompliant
    pass
except (ValueError, TypeError) as exception:
    pass
except (ValueError):
    pass
except:
    pass
