from typing import List, Tuple


def calculate_average(numbers):
    return sum(numbers) / len(numbers)

def get_name_and_age():
    name= "John Doe"
    age = 30
    return name, age

def greet(name):
    return f"Hello, {name}!"

def main():
    numbers = [1, 2, 3, 4, 5]
    print(calculate_average(numbers))

    bar = numbers[2]

    name, age = get_name_and_age()
    print(f"Name: {name}, Age: {age}")

    greeting = greet(name)
    print(greeting)


if __name__ == "__main__":
    main()


###########################################################################
###########################################################################
###########################################################################


#
#
# from typing import List, Tuple
#
# def calculate_average(numbers: List[int]) -> float:
#     return sum(numbers) / len(numbers)
#
# def get_name_and_age() -> Tuple[str, int]:
#     name: str = "John Doe"
#     age: int = 30
#     return name, age
#
# def greet(name: str) -> str:
#     return f"Hello, {name}!"
#
# def main() -> None:
#     numbers: List[int] = [1, 2, 3, 4, 5]
#     print(calculate_average(numbers))
#
#     name, age = get_name_and_age()
#     print(f"Name: {name}, Age: {age}")
#
#     greeting: str = greet(name)
#     print(greeting)
#
# if __name__ == "__main__":
#     main()
