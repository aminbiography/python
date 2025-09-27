Live URL: https://aminbiography.github.io/python/  
  
---

![python.py](https://i.pinimg.com/736x/58/a5/bd/58a5bdd98f2669cdc2ee1496565fc8df.jpg)

---
  
# Essential Python Concepts with Examples and Outputs

## Python Basics
```
print, input, variables, if-else, loops, functions, lists, dictionaries, import, try-except
```

---

## 01: Printing and Input
```python
print("Hello, World!")
```
```
Output:
Hello, World!
```

```python
name = "Alice"  # Simulated input
print("Hello,", name)
```
```
Output:
Hello, Alice
```

---

## 02: Variables and Data Types
```python
age = 25
name = "Alice"
pi = 3.14
is_active = True
print(age, name, pi, is_active)
```
```
Output:
25 Alice 3.14 True
```

```python
print(type(name))
```
```
Output:
<class 'str'>
```

---

## 03: Conditional Statements
```python
age = 15
if age >= 18:
    print("Adult")
elif age > 12:
    print("Teen")
else:
    print("Child")
```
```
Output:
Teen
```

---

## 04: Loops
```python
for i in range(5):
    print(i)
```
```
Output:
0
1
2
3
4
```

```python
count = 0
while count < 5:
    print(count)
    count += 1
```
```
Output:
0
1
2
3
4
```

---

## 05: Functions
```python
def greet(name):
    return f"Hello, {name}!"

print(greet("Alice"))
```
```
Output:
Hello, Alice!
```

---

## 06: Lists
```python
fruits = ["apple", "banana", "cherry"]
print(fruits[0])
```
```
Output:
apple
```

```python
fruits.append("orange")
fruits.remove("banana")
print(fruits)
```
```
Output:
['apple', 'cherry', 'orange']
```

---

## 07: Dictionaries
```python
person = {"name": "Bob", "age": 30}
print(person["name"])
```
```
Output:
Bob
```

```python
person["city"] = "New York"
del person["age"]
print(person)
```
```
Output:
{'name': 'Bob', 'city': 'New York'}
```

---

## 08: Importing Modules
```python
import math
print(math.sqrt(16))
```
```
Output:
4.0
```

```python
from math import pi
print(pi)
```
```
Output:
3.141592653589793
```

---

## 09: File Handling
```python
with open("file.txt", "w") as f:
    f.write("Hello, file!")

with open("file.txt", "r") as f:
    content = f.read()
    print(content)
```
```
Output:
Hello, file!
```

---

## 10: Exception Handling
```python
try:
    num = 5
    print(10 / num)
except ZeroDivisionError:
    print("Cannot divide by zero.")
except ValueError:
    print("Invalid input.")
```
```
Output:
2.0
```

---

## 11: Classes and Objects
```python
class Person:
    def __init__(self, name):
        self.name = name
    def greet(self):
        print(f"Hello, my name is {self.name}.")

p = Person("Alice")
p.greet()
```
```
Output:
Hello, my name is Alice.
```

---

## 12: List Comprehensions
```python
numbers = [x**2 for x in range(5)]
print(numbers)
```
```
Output:
[0, 1, 4, 9, 16]
```

---

## 13: Lambda Functions
```python
square = lambda x: x**2
print(square(4))
```
```
Output:
16
```

---

## 14: Map, Filter, Reduce
```python
nums = [1, 2, 3, 4]
print(list(map(lambda x: x*2, nums)))
```
```
Output:
[2, 4, 6, 8]
```

```python
print(list(filter(lambda x: x%2==0, nums)))
```
```
Output:
[2, 4]
```

```python
from functools import reduce
print(reduce(lambda a, b: a+b, nums))
```
```
Output:
10
```

---

## 15: Virtual Environments
```
python -m venv env
source env/bin/activate   # Linux/Mac
env\Scripts\activate      # Windows
```

---

![python.py](https://i.pinimg.com/736x/58/a5/bd/58a5bdd98f2669cdc2ee1496565fc8df.jpg)


