Live URL: https://aminbiography.github.io/python/

![python.py](https://i.pinimg.com/736x/58/a5/bd/58a5bdd98f2669cdc2ee1496565fc8df.jpg)

  
<h1>Here are some essential Python concepts and commands that every Python programmer should know.</h1>

<p>Python Basics:</p>

```
print, input, variables, if-else, loops, functions, lists, dictionaries, import, try-except
```

<h2>01: Printing and Input</h2>

<p>print()</p>
<p>Displays output to the console.</p>

```
print("Hello, World!")
```

<p>input()</p>
<p>Reads user input as a string.</p>

```
name = input("Enter your name: ")
print("Hello,", name)
```

<h2>02: Variables and Data Types</h2>

<p>Assigning values</p>
<p>Variables store data of various types.</p>

```
age = 25
name = "Alice"
pi = 3.14
is_active = True
```

<p>type()</p>
<p>Checks the type of a variable.</p>

```
print(type(name))
```

<h2>03: Conditional Statements</h2>

<p>if, elif, else</p>
<p>Executes code blocks based on conditions.</p>

```
if age >= 18:
    print("Adult")
elif age > 12:
    print("Teen")
else:
    print("Child")
```

<h2>04: Loops</h2>

<p>for loop</p>
<p>Iterates over a sequence.</p>

```
for i in range(5):
    print(i)
```

<p>while loop</p>
<p>Repeats while a condition is true.</p>

```
count = 0
while count < 5:
    print(count)
    count += 1
```

<h2>05: Functions</h2>

<p>Defining functions</p>
<p>Reusable blocks of code.</p>

```
def greet(name):
    return f"Hello, {name}!"
print(greet("Alice"))
```

<h2>06: Lists</h2>

<p>Creating lists</p>
<p>Ordered, mutable collections.</p>

```
fruits = ["apple", "banana", "cherry"]
print(fruits[0])
```

<p>List methods</p>

```
fruits.append("orange")
fruits.remove("banana")
```

<h2>07: Dictionaries</h2>

<p>Key-value pairs</p>

```
person = {"name": "Bob", "age": 30}
print(person["name"])
```

<p>Adding and removing items</p>

```
person["city"] = "New York"
del person["age"]
```

<h2>08: Importing Modules</h2>

<p>import</p>
<p>Brings in external modules.</p>

```
import math
print(math.sqrt(16))
```

<p>from ... import</p>

```
from math import pi
print(pi)
```

<h2>09: File Handling</h2>

<p>Opening files</p>

```
with open("file.txt", "r") as f:
    content = f.read()
    print(content)
```

<p>Writing to files</p>

```
with open("file.txt", "w") as f:
    f.write("Hello, file!")
```

<h2>10: Exception Handling</h2>

<p>try, except</p>

```
try:
    num = int(input("Enter a number: "))
    print(10 / num)
except ZeroDivisionError:
    print("Cannot divide by zero.")
except ValueError:
    print("Invalid input.")
```

<h2>11: Classes and Objects</h2>

<p>Defining classes</p>

```
class Person:
    def __init__(self, name):
        self.name = name
    def greet(self):
        print(f"Hello, my name is {self.name}.")

p = Person("Alice")
p.greet()
```

<h2>12: List Comprehensions</h2>

```
numbers = [x**2 for x in range(5)]
print(numbers)
```

<h2>13: Lambda Functions</h2>

```
square = lambda x: x**2
print(square(4))
```

<h2>14: Map, Filter, Reduce</h2>

```
nums = [1, 2, 3, 4]
print(list(map(lambda x: x*2, nums)))
print(list(filter(lambda x: x%2==0, nums)))

from functools import reduce
print(reduce(lambda a,b: a+b, nums))
```

<h2>15: Virtual Environments</h2>

```
python -m venv env
source env/bin/activate   # Linux/Mac
env\Scripts\activate      # Windows
```
```


![python.py](https://i.pinimg.com/736x/58/a5/bd/58a5bdd98f2669cdc2ee1496565fc8df.jpg)


