from typing import cast


class Animal:
    def shout(self):
        print("Hello. I'm an animal.")


class Dog(Animal):
    def shout(self, times: int = 1):
        print(" ".join(["Bark!"] * times))


class Cat(Animal):
    def shout(self, times: int = 1):
        print(" ".join(["Miaou!"] * times))


def animal_shout(animal: Animal):
    animal.shout()


def test_dog(dog: Dog):
    dog.shout()


def test_cat(cat):
    animal_shout(cat)


def f2():
    a_dog: Dog = Dog()
    a_dog_annotated_as_animal: Animal = Dog()
    a_dog_casted_as_cat = cast(Cat, Dog())
    a_cat: Cat = Cat()

    test_dog(a_dog)
    test_dog(a_dog_annotated_as_animal)
    test_cat(a_dog_casted_as_cat)
