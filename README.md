

Live URL: https://aminbiography.github.io/python/

![python.py](https://i.pinimg.com/736x/58/a5/bd/58a5bdd98f2669cdc2ee1496565fc8df.jpg)


<h1>Python Overview</h1>

```
# 1. Setting Up Python
# Before you start coding in Python, make sure you have Python installed on your computer.
# Download Python from the official website: https://www.python.org/downloads/

# After installing Python, you can test it in the command line or terminal:
# - Type `python` or `python3` to ensure it is installed correctly.

# 2. Hello World Program
# A basic Python program to print a message:
print("Hello, World!")

# 3. Variables and Data Types

# Variables store data that you can use in your program.
# Python is dynamically typed, meaning you don’t need to declare variable types explicitly.
my_variable = 10  # Integer
my_string = "Hello, Python!"  # String
my_float = 10.5  # Float
my_boolean = True  # Boolean

# Printing variables
print(my_variable)
print(my_string)
print(my_float)
print(my_boolean)

# 4. Basic Math Operations
# You can perform mathematical operations like addition, subtraction, multiplication, and division in Python.

addition = 5 + 3
subtraction = 10 - 4
multiplication = 7 * 3
division = 12 / 4

# Print the results
print("Addition:", addition)
print("Subtraction:", subtraction)
print("Multiplication:", multiplication)
print("Division:", division)

# 5. Lists (Arrays in Python)
# A list is a collection of items that can be of different types.

my_list = [1, 2, 3, "Python", 5.5]

# Accessing list items
print(my_list[0])  # First item
print(my_list[-1])  # Last item

# Modifying a list
my_list.append(100)
print(my_list)

# 6. If-Else Statements
# Conditional statements allow you to execute different code depending on conditions.

age = 18

if age >= 18:
    print("You are an adult.")
else:
    print("You are a minor.")

# 7. Loops
# Python has two types of loops: `for` and `while`.

# For loop example
for i in range(5):
    print("Iteration:", i)

# While loop example
counter = 0
while counter < 5:
    print("Counter:", counter)
    counter += 1

# 8. Functions
# Functions in Python allow you to group reusable code together.

def greet(name):
    print("Hello, " + name)

# Calling a function
greet("Alice")
greet("Bob")

# 9. Dictionaries (Key-Value Pairs)
# A dictionary in Python is a collection of key-value pairs.

my_dict = {
    "name": "John",
    "age": 25,
    "location": "New York"
}

# Accessing values in a dictionary
print(my_dict["name"])
print(my_dict["age"])

# Adding or updating items
my_dict["age"] = 26
my_dict["job"] = "Engineer"
print(my_dict)

# 10. Classes and Objects (Object-Oriented Programming)
# Python supports Object-Oriented Programming (OOP). You can create classes and objects.

class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def greet(self):
        print("Hello, my name is " + self.name + " and I am " + str(self.age) + " years old.")

# Creating an object of the Person class
person1 = Person("Alice", 30)
person1.greet()

# 11. Exception Handling
# Exception handling in Python helps you deal with errors gracefully using try-except blocks.

try:
    result = 10 / 0  # This will cause a ZeroDivisionError
except ZeroDivisionError:
    print("Cannot divide by zero!")

# 12. File Handling
# Python can handle file reading and writing operations.

# Writing to a file
with open("example.txt", "w") as file:
    file.write("Hello, this is a test file.")

# Reading from a file
with open("example.txt", "r") as file:
    content = file.read()
    print(content)

# 13. Lambda Functions
# Lambda functions are small anonymous functions defined using the `lambda` keyword.

# Example of a lambda function:
multiply = lambda x, y: x * y
print(multiply(3, 4))  # Output: 12

# 14. List Comprehensions
# List comprehensions provide a concise way to create lists by applying expressions to each element of an iterable.

# Example of list comprehension:
squares = [x**2 for x in range(5)]
print(squares)  # Output: [0, 1, 4, 9, 16]

# 15. Modules and Packages
# Modules are Python files that contain code and can be imported into other Python scripts.

# Example of using the math module:
import math

print(math.sqrt(16))  # Output: 4.0
print(math.pi)        # Output: 3.141592653589793

# You can also create your own modules by saving Python code into a `.py` file and importing it in other scripts.

# 16. Working with Dates and Times
# The `datetime` module allows working with dates and times.

from datetime import datetime

# Get the current date and time
now = datetime.now()
print(now)  # Output: Current date and time

# Formatting date and time
formatted_date = now.strftime("%Y-%m-%d %H:%M:%S")
print(formatted_date)  # Output: Formatted current date and time

# 17. Regular Expressions (Regex)
# The `re` module in Python allows you to use regular expressions to search and manipulate strings.

import re

# Searching for a pattern in a string
pattern = r"\d+"  # Pattern to find digits
text = "There are 100 apples."
match = re.search(pattern, text)
if match:
    print("Match found:", match.group())  # Output: Match found: 100

# 18. Iterators and Generators
# Iterators allow you to traverse through all the elements of a collection. Generators provide a more efficient way to create iterators.

# Example of an iterator:
my_list = [1, 2, 3, 4]
my_iterator = iter(my_list)
print(next(my_iterator))  # Output: 1
print(next(my_iterator))  # Output: 2

# Example of a generator:
def my_generator():
    yield 1
    yield 2
    yield 3

gen = my_generator()
for value in gen:
    print(value)  # Output: 1 2 3

# 19. Decorators
# Decorators allow you to modify the behavior of functions or methods without changing their code.

# Example of a simple decorator:
def decorator_function(func):
    def wrapper():
        print("Before the function is called.")
        func()
        print("After the function is called.")
    return wrapper

@decorator_function
def say_hello():
    print("Hello!")

say_hello()
# Output:
# Before the function is called.
# Hello!
# After the function is called.

# 20. Working with APIs
# Python allows you to interact with APIs (Application Programming Interfaces) using libraries like `requests`.

import requests

# Sending a GET request to an API
response = requests.get("https://jsonplaceholder.typicode.com/posts")
print(response.status_code)  # Output: 200 (OK)
print(response.json())       # Output: JSON data returned from the API

# 21. Multithreading and Multiprocessing
# Python's `threading` module allows running tasks in parallel (multithreading).
# The `multiprocessing` module allows creating separate processes to take advantage of multiple CPU cores.

import threading

# Example of multithreading:
def print_numbers():
    for i in range(5):
        print(i)

# Creating two threads
thread1 = threading.Thread(target=print_numbers)
thread2 = threading.Thread(target=print_numbers)

# Starting the threads
thread1.start()
thread2.start()

# 22. Web Development with Flask
# Flask is a lightweight web framework in Python that allows you to build web applications easily.

from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, Flask!"

if __name__ == '__main__':
    app.run(debug=True)

# To run this Flask app, save it in a `.py` file and run it from the command line with:
# $ python filename.py
# Then open your browser and visit http://127.0.0.1:5000/

# 23. Working with Databases (SQLite)
# Python provides built-in support for interacting with databases like SQLite.

import sqlite3

# Connecting to a SQLite database
conn = sqlite3.connect('example.db')
cursor = conn.cursor()

# Creating a table
cursor.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT)''')

# Inserting data
cursor.execute("INSERT INTO users (name) VALUES ('Alice')")
cursor.execute("INSERT INTO users (name) VALUES ('Bob')")
conn.commit()

# Querying data
cursor.execute("SELECT * FROM users")
print(cursor.fetchall())  # Output: [(1, 'Alice'), (2, 'Bob')]

# Closing the connection
conn.close()

# 24. Unit Testing
# Python's `unittest` module allows you to test your code by writing test cases.

import unittest

# Example of a simple test case
def add(a, b):
    return a + b

class TestMathOperations(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)  # Test that 2 + 3 equals 5

if __name__ == '__main__':
    unittest.main()

# 25. Virtual Environments
# A virtual environment allows you to create isolated Python environments for your projects.

# Creating a virtual environment:
# $ python -m venv myenv

# Activating the virtual environment (Windows):
# $ myenv\Scripts\activate

# Activating the virtual environment (Mac/Linux):
# $ source myenv/bin/activate

# 26. Python Packages
# You can install and use third-party libraries using `pip`, the Python package manager.

# Installing a package using pip:
# $ pip install requests

# Importing and using the installed package:
import requests
response = requests.get("https://jsonplaceholder.typicode.com/posts")
print(response.status_code)

# 27. Working with JSON
# Python's `json` module helps you work with JSON data (JavaScript Object Notation).

import json

# Convert a Python dictionary to JSON
data = {"name": "Alice", "age": 25}
json_data = json.dumps(data)
print(json_data)  # Output: '{"name": "Alice", "age": 25}'

# Convert JSON data back to Python dictionary
python_data = json.loads(json_data)
print(python_data)  # Output: {'name': 'Alice', 'age': 25}

# 28. File Handling
# Python provides built-in functions to work with files, including reading, writing, and managing files.

# Writing to a file:
with open("sample.txt", "w") as file:
    file.write("Hello, Python!\n")
    file.write("This is a sample file.")

# Reading from a file:
with open("sample.txt", "r") as file:
    content = file.read()
    print(content)  # Output: Hello, Python!\nThis is a sample file.

# Appending to a file:
with open("sample.txt", "a") as file:
    file.write("\nAppended content.")

# 29. Sorting and Filtering
# Python provides several ways to sort and filter data, such as using the `sorted()` function and `filter()`.

# Sorting a list of numbers:
numbers = [4, 1, 3, 2, 5]
sorted_numbers = sorted(numbers)
print(sorted_numbers)  # Output: [1, 2, 3, 4, 5]

# Filtering a list of numbers (keep only even numbers):
even_numbers = filter(lambda x: x % 2 == 0, numbers)
print(list(even_numbers))  # Output: [4, 2]

# 30. Mapping Functions
# The `map()` function allows you to apply a function to every item in an iterable.

# Example of using `map()` to square numbers:
numbers = [1, 2, 3, 4, 5]
squared_numbers = map(lambda x: x**2, numbers)
print(list(squared_numbers))  # Output: [1, 4, 9, 16, 25]

# 31. Global and Local Variables
# Variables declared inside a function are local, while variables declared outside a function are global.

# Example:
x = 10  # Global variable

def foo():
    global x  # Accessing the global variable
    x = 20    # Modifying the global variable
    print(x)

foo()  # Output: 20
print(x)  # Output: 20

# 32. Recursion
# Recursion is a technique where a function calls itself to solve a problem.

# Example of a recursive function to calculate factorial:
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)

print(factorial(5))  # Output: 120 (5 * 4 * 3 * 2 * 1)

# 33. Map, Filter, and Reduce
# The `reduce()` function (from the `functools` module) is used to apply a function cumulatively to the items of an iterable.

from functools import reduce

# Example of `reduce()` to sum a list:
numbers = [1, 2, 3, 4, 5]
total = reduce(lambda x, y: x + y, numbers)
print(total)  # Output: 15

# 34. Handling Exceptions
# Python uses `try`, `except`, `else`, and `finally` blocks to handle exceptions.

# Example of handling exceptions:
try:
    value = int(input("Enter a number: "))
except ValueError:
    print("Invalid input! Please enter a valid number.")
else:
    print(f"You entered: {value}")
finally:
    print("This will always be executed.")

# 35. Working with JSON Files
# Python makes it easy to read and write JSON data, which is commonly used in APIs and data exchange.

import json

# Writing a Python object to a JSON file:
data = {"name": "Alice", "age": 25}
with open("data.json", "w") as json_file:
    json.dump(data, json_file)

# Reading data from a JSON file:
with open("data.json", "r") as json_file:
    loaded_data = json.load(json_file)
    print(loaded_data)  # Output: {'name': 'Alice', 'age': 25}

# 36. Decorators with Arguments
# Decorators can also accept arguments to modify their behavior.

# Example of a decorator with arguments:
def repeat(n):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(n):
                func(*args, **kwargs)
        return wrapper
    return decorator

@repeat(3)
def say_hello():
    print("Hello!")

say_hello()  # Output: Hello! Hello! Hello!

# 37. Python's `os` module
# The `os` module allows you to interact with the operating system, such as file handling, directory manipulation, etc.

import os

# Get the current working directory:
print(os.getcwd())

# List files in a directory:
print(os.listdir("."))

# Create a new directory:
os.mkdir("new_directory")

# 38. Python's `sys` module
# The `sys` module provides access to system-specific parameters and functions, including command-line arguments.

import sys

# Get command-line arguments:
print(sys.argv)  # Output: List of arguments passed to the script

# Exit the program:
sys.exit("Exiting the program.")

# 39. Python's `time` module
# The `time` module provides functions for working with time-related tasks, such as measuring performance.

import time

# Example of time delay:
print("Start")
time.sleep(2)  # Sleep for 2 seconds
print("End")  # This will print after 2 seconds

# Measure execution time:
start_time = time.time()
# Some code to measure time for
end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds")

# 40. Python's `collections` module
# The `collections` module provides specialized container datatypes like `namedtuple`, `deque`, `Counter`, etc.

from collections import Counter

# Example of using `Counter` to count occurrences in a list:
word_list = ['apple', 'banana', 'apple', 'orange', 'banana', 'apple']
word_count = Counter(word_list)
print(word_count)  # Output: Counter({'apple': 3, 'banana': 2, 'orange': 1})

# 41. Object-Oriented Programming (OOP)
# Python supports Object-Oriented Programming (OOP), allowing you to define classes and objects.

# Example of creating a class:
class Car:
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model

    def display(self):
        print(f"Car Brand: {self.brand}, Model: {self.model}")

# Creating an object of the Car class:
my_car = Car("Toyota", "Corolla")
my_car.display()  # Output: Car Brand: Toyota, Model: Corolla

# 42. Static Methods and Class Methods
# In addition to regular methods, you can define static methods and class methods in Python classes.

class MyClass:
    @staticmethod
    def static_method():
        print("This is a static method.")
    
    @classmethod
    def class_method(cls):
        print("This is a class method.")

# Calling the static and class methods:
MyClass.static_method()
MyClass.class_method()

# 43. Context Managers
# Context managers allow you to manage resources (like files) with the `with` statement, ensuring proper cleanup.

# Example of a context manager with file handling:
with open("example.txt", "w") as file:
    file.write("Hello, Python with context manager!")

# 44. Python's `shutil` module
# The `shutil` module provides high-level file operations like copying and moving files.

import shutil

# Copying a file:
shutil.copy("source.txt", "destination.txt")

# Moving a file:
shutil.move("source.txt", "new_directory/source.txt")

# 45. Python's `sqlite3` module
# Python's `sqlite3` module allows you to interact with SQLite databases.

import sqlite3

# Connect to SQLite database
connection = sqlite3.connect('my_database.db')

# Create a cursor object
cursor = connection.cursor()

# Create a table in SQLite database
cursor.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT)''')

# Insert data into the table
cursor.execute("INSERT INTO users (name) VALUES ('Alice')")
connection.commit()

# Query data from the table
cursor.execute("SELECT * FROM users")
print(cursor.fetchall())  # Output: [(1, 'Alice')]

# Close the connection
connection.close()


# 46. Regular Expressions (regex)
# Python's `re` module allows you to work with regular expressions for pattern matching in strings.

import re

# Example: Searching for a pattern in a string
text = "The quick brown fox jumps over the lazy dog."
pattern = r"\b\w{5}\b"  # Find words with exactly 5 characters

matches = re.findall(pattern, text)
print(matches)  # Output: ['quick', 'brown', 'jumps']

# 47. Lambda Functions
# Lambda functions are small anonymous functions defined with the `lambda` keyword.

# Example of a lambda function:
multiply = lambda x, y: x * y
print(multiply(3, 4))  # Output: 12

# 48. Python's `itertools` module
# The `itertools` module provides functions for creating iterators for efficient looping.

import itertools

# Example of using `itertools.combinations` to generate combinations from a list:
numbers = [1, 2, 3, 4]
combinations = itertools.combinations(numbers, 2)
print(list(combinations))  # Output: [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]

# 49. Data Structures - Stacks and Queues
# A stack is a collection where elements are added and removed from one end (LIFO).
# A queue is a collection where elements are added from one end and removed from the other (FIFO).

# Example of a stack:
stack = []
stack.append(1)  # Push 1
stack.append(2)  # Push 2
print(stack.pop())  # Pop 2 (Output: 2)

# Example of a queue using `collections.deque`:
from collections import deque
queue = deque()
queue.append(1)  # Enqueue 1
queue.append(2)  # Enqueue 2
print(queue.popleft())  # Dequeue 1 (Output: 1)

# 50. Python's `socket` module
# The `socket` module allows you to implement networking, including client-server communication.

import socket

# Example of creating a simple server:
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(("localhost", 12345))
server_socket.listen(1)
print("Server is waiting for a connection...")
client_socket, client_address = server_socket.accept()
print("Client connected:", client_address)

# Example of creating a client:
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(("localhost", 12345))
client_socket.sendall(b"Hello, Server!")

# 51. Working with CSV Files
# Python's `csv` module provides tools for reading and writing CSV files.

import csv

# Writing data to a CSV file:
header = ["Name", "Age", "Country"]
rows = [["Alice", 25, "USA"], ["Bob", 30, "Canada"]]
with open("people.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(rows)

# Reading data from a CSV file:
with open("people.csv", mode="r") as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)  # Output: ['Name', 'Age', 'Country'], ['Alice', 25, 'USA'], ['Bob', 30, 'Canada']

# 52. Working with Databases (MySQL, PostgreSQL, etc.)
# Python supports working with databases like MySQL, PostgreSQL, SQLite, etc.

import mysql.connector

# Example of connecting to a MySQL database and executing queries:
connection = mysql.connector.connect(
    host="localhost", user="root", password="password", database="test_db"
)

cursor = connection.cursor()
cursor.execute("SELECT * FROM users")
result = cursor.fetchall()
print(result)

cursor.close()
connection.close()

# 53. Python's `pandas` Library
# Pandas is a powerful library for data analysis and manipulation. It provides data structures like `DataFrame` and `Series`.

import pandas as pd

# Example of creating a DataFrame:
data = {"Name": ["Alice", "Bob"], "Age": [25, 30], "Country": ["USA", "Canada"]}
df = pd.DataFrame(data)
print(df)

# Example of reading data from a CSV file using pandas:
df = pd.read_csv("people.csv")
print(df)

# 54. Python's `matplotlib` and `seaborn` Libraries
# `matplotlib` is used for creating static visualizations, and `seaborn` is built on top of matplotlib for statistical graphics.

import matplotlib.pyplot as plt
import seaborn as sns

# Example of plotting a simple line chart with matplotlib:
x = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 25]
plt.plot(x, y)
plt.title("Simple Line Chart")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

# Example of plotting a bar chart with seaborn:
sns.barplot(x=["A", "B", "C"], y=[10, 20, 30])
plt.title("Bar Chart")
plt.show()

# 55. Python's `numpy` Library
# NumPy is a library for numerical computing. It provides support for large multi-dimensional arrays and matrices, as well as a collection of mathematical functions.

import numpy as np

# Example of creating a NumPy array:
arr = np.array([1, 2, 3, 4])
print(arr)  # Output: [1 2 3 4]

# Example of performing mathematical operations with NumPy:
arr2 = np.array([5, 6, 7, 8])
sum_arr = np.add(arr, arr2)
print(sum_arr)  # Output: [6 8 10 12]

# 56. Web Development with Flask
# Flask is a micro web framework for Python that allows you to build web applications quickly.

from flask import Flask

# Example of a basic Flask application:
app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, Flask!"

if __name__ == "__main__":
    app.run()

# 57. Web Development with Django
# Django is a high-level web framework that encourages rapid development and clean, pragmatic design.

# Example of a simple Django view:
from django.http import HttpResponse

def my_view(request):
    return HttpResponse("Hello, Django!")

# 58. Python's `asyncio` module
# The `asyncio` module allows you to write asynchronous programs, which can be useful for I/O-bound and high-level structured network code.

import asyncio

# Example of an asynchronous function:
async def say_hello():
    print("Hello")
    await asyncio.sleep(1)  # Asynchronous sleep (non-blocking)
    print("World")

# Running an event loop:
asyncio.run(say_hello())

# 59. Python's `pytest` Library
# `pytest` is a testing framework for Python that makes it easy to write simple and scalable test cases.

# Example of a simple test case using pytest:
def test_addition():
    assert 1 + 1 == 2

# To run the test, you would run `pytest` in the command line.

# 60. Python's `tkinter` Library
# `tkinter` is the standard Python interface to the Tk GUI toolkit.

import tkinter as tk

# Example of creating a simple window with a button:
root = tk.Tk()
button = tk.Button(root, text="Click me", command=lambda: print("Button clicked"))
button.pack()
root.mainloop()

# 61. Python's `threading` module
# The `threading` module allows you to run multiple threads concurrently, which can be useful for parallel tasks.

import threading

# Example of using threading to run two functions simultaneously:
def task1():
    print("Task 1 running")

def task2():
    print("Task 2 running")

# Creating and starting threads:
thread1 = threading.Thread(target=task1)
thread2 = threading.Thread(target=task2)

thread1.start()
thread2.start()

# Wait for both threads to complete:
thread1.join()
thread2.join()

# 62. Python's `multiprocessing` module
# The `multiprocessing` module allows you to create multiple processes, making it possible to take full advantage of multi-core systems.

import multiprocessing

# Example of using multiprocessing:
def worker():
    print("Worker process started")

# Creating and starting a process:
process = multiprocessing.Process(target=worker)
process.start()
process.join()


# 63. Python's `subprocess` module
# The `subprocess` module allows you to spawn new processes, connect to their input/output/error pipes, and obtain their return codes.

import subprocess

# Example of running a shell command and capturing its output:
result = subprocess.run(['echo', 'Hello, subprocess!'], capture_output=True, text=True)
print(result.stdout)  # Output: Hello, subprocess!

# 64. Python's `os` and `sys` Modules
# The `os` module provides a way of using operating system-dependent functionality, and `sys` allows you to interact with the Python runtime environment.

import os
import sys

# Example of using `os` to get the current working directory:
current_directory = os.getcwd()
print(current_directory)

# Example of using `sys` to interact with command-line arguments:
print(sys.argv)  # List of command-line arguments

# 65. Python's `logging` module
# The `logging` module provides a way to log messages from your application, useful for debugging and tracking application behavior.

import logging

# Example of configuring and using the logging module:
logging.basicConfig(level=logging.INFO)
logging.info("This is an info message")
logging.warning("This is a warning message")
logging.error("This is an error message")

# 66. Python's `json` module
# The `json` module allows you to work with JSON data, both reading and writing.

import json

# Example of converting a Python object to a JSON string:
data = {"name": "Alice", "age": 25}
json_string = json.dumps(data)
print(json_string)  # Output: {"name": "Alice", "age": 25}

# Example of converting a JSON string back to a Python object:
json_data = '{"name": "Bob", "age": 30}'
python_data = json.loads(json_data)
print(python_data)  # Output: {'name': 'Bob', 'age': 30}

# 67. Python's `hashlib` module
# The `hashlib` module allows you to create secure hash values using different algorithms like SHA-256.

import hashlib

# Example of generating a SHA-256 hash:
text = "Hello, world!"
hashed_text = hashlib.sha256(text.encode()).hexdigest()
print(hashed_text)  # Output: A long SHA-256 hash

# 68. Python's `functools` module
# The `functools` module provides higher-order functions that operate on or return other functions.

from functools import lru_cache

# Example of using the Least Recently Used (LRU) cache to optimize a function:
@lru_cache(maxsize=3)
def expensive_function(x):
    print(f"Computing {x}...")
    return x * x

print(expensive_function(2))  # Output: Computing 2... 4
print(expensive_function(2))  # Output: 4 (cached result)

# 69. Python's `collections` module
# The `collections` module includes useful data structures like `defaultdict`, `deque`, `namedtuple`, and `Counter`.

from collections import defaultdict

# Example of using a defaultdict:
default_dict = defaultdict(int)  # Default value is 0
default_dict["apple"] += 1
print(default_dict["apple"])  # Output: 1

# 70. Python's `memoryview` object
# The `memoryview` object allows you to access the internal data of objects like bytes and bytearrays without making copies.

# Example of using memoryview with a bytearray:
data = bytearray(b"Hello, world!")
view = memoryview(data)
print(view[0])  # Output: 72 (ASCII code of 'H')

# 71. Python's `weakref` module
# The `weakref` module allows you to create weak references to objects, which do not prevent garbage collection.

import weakref

# Example of creating a weak reference to an object:
class MyClass:
    def __del__(self):
        print("MyClass instance deleted")

obj = MyClass()
weak_ref = weakref.ref(obj)
del obj  # Output: MyClass instance deleted

# 72. Python's `pickle` module
# The `pickle` module allows you to serialize and deserialize Python objects, enabling saving and loading of objects.

import pickle

# Example of pickling an object:
data = {"name": "Alice", "age": 25}
with open("data.pickle", "wb") as f:
    pickle.dump(data, f)

# Example of unpickling an object:
with open("data.pickle", "rb") as f:
    loaded_data = pickle.load(f)
print(loaded_data)  # Output: {'name': 'Alice', 'age': 25}

# 73. Python's `contextlib` module
# The `contextlib` module provides utilities for working with context managers, which simplify resource management.

from contextlib import contextmanager

# Example of using contextlib to create a custom context manager:
@contextmanager
def my_context_manager():
    print("Entering context")
    yield
    print("Exiting context")

with my_context_manager():
    print("Inside context")

# Output:
# Entering context
# Inside context
# Exiting context

# 74. Python's `uuid` module
# The `uuid` module generates universally unique identifiers (UUIDs).

import uuid

# Example of generating a random UUID:
generated_uuid = uuid.uuid4()
print(generated_uuid)  # Output: e.g., 2f3c14f8-c3d1-4b7b-91c4-076073dfb320

# 75. Python's `inspect` module
# The `inspect` module allows introspection of live objects, including functions and classes.

import inspect

# Example of getting information about a function:
def my_function(a, b):
    return a + b

print(inspect.signature(my_function))  # Output: (a, b)

# 76. Python's `dataclasses` module
# The `dataclasses` module provides a decorator and functions for automatically adding special methods to user-defined classes.

from dataclasses import dataclass

# Example of using a dataclass:
@dataclass
class Person:
    name: str
    age: int

person = Person("Alice", 25)
print(person)  # Output: Person(name='Alice', age=25)

# 77. Python's `threading` vs. `multiprocessing`
# `threading` is for I/O-bound tasks, and `multiprocessing` is for CPU-bound tasks.

import threading
import multiprocessing

# Example of using multiprocessing for CPU-bound tasks:
def cpu_bound_task():
    print("CPU-bound task")

if __name__ == "__main__":
    # Using threading
    thread = threading.Thread(target=cpu_bound_task)
    thread.start()
    thread.join()

    # Using multiprocessing
    process = multiprocessing.Process(target=cpu_bound_task)
    process.start()
    process.join()

# 78. Python's `asyncio` vs. `threading`
# `asyncio` is better suited for I/O-bound tasks that require high concurrency, while `threading` is for concurrent execution of I/O-bound and CPU-bound tasks.

import asyncio

# Example of asynchronous programming with asyncio:
async def async_task():
    print("Starting async task...")
    await asyncio.sleep(1)
    print("Async task finished")

# Running the async task
asyncio.run(async_task())

# 79. Python's `warnings` module
# The `warnings` module provides a way to issue warning messages from your code.

import warnings

# Example of issuing a warning:
warnings.warn("This is a warning", UserWarning)

# 80. Python's `sqlite3` module
# The `sqlite3` module allows you to interact with SQLite databases, which are lightweight, self-contained databases.

import sqlite3

# Example of creating a database and inserting data:
conn = sqlite3.connect("example.db")
cursor = conn.cursor()

# Create a table
cursor.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)''')

# Insert data
cursor.execute("INSERT INTO users (name, age) VALUES ('Alice', 25)")
conn.commit()

# Retrieve data
cursor.execute("SELECT * FROM users")
print(cursor.fetchall())  # Output: [(1, 'Alice', 25)]

# Close the connection
conn.close()


# 81. Python's `asyncio` and `await`
# The `asyncio` library is used for writing single-threaded concurrent code using async/await syntax.

import asyncio

# Example of using async/await for concurrent tasks:
async def task_1():
    print("Task 1 started")
    await asyncio.sleep(2)  # Simulate I/O-bound operation
    print("Task 1 finished")

async def task_2():
    print("Task 2 started")
    await asyncio.sleep(1)  # Simulate I/O-bound operation
    print("Task 2 finished")

# Running the tasks concurrently:
async def main():
    await asyncio.gather(task_1(), task_2())

asyncio.run(main())

# 82. Python's `with` statement for resource management
# The `with` statement simplifies exception handling and resource management, especially when dealing with files or network connections.

# Example of using `with` to manage file I/O:
with open("file.txt", "w") as f:
    f.write("Hello, file!")

# No need to explicitly close the file, the `with` statement handles it automatically.

# 83. Python's `itertools` module
# The `itertools` module provides efficient tools for working with iterators, including generating permutations, combinations, and infinite sequences.

import itertools

# Example of using `itertools.permutations`:
data = [1, 2, 3]
perms = itertools.permutations(data)
print(list(perms))  # Output: [(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)]

# 84. Python's `subprocess.Popen` for advanced process control
# `subprocess.Popen` provides more advanced process control compared to `subprocess.run`.

import subprocess

# Example of using `Popen` to handle a running process:
process = subprocess.Popen(["echo", "Hello, World!"], stdout=subprocess.PIPE)
stdout, stderr = process.communicate()  # Retrieve output and error
print(stdout.decode())  # Output: Hello, World!

# 85. Python's `shutil` module
# The `shutil` module offers a higher-level interface for file operations, such as copying files and directories.

import shutil

# Example of copying a file:
shutil.copy("source.txt", "destination.txt")

# 86. Python's `enum` module
# The `enum` module allows you to create enumerated constants, which can be useful for code readability and avoiding magic numbers.

from enum import Enum

# Example of creating an enumeration:
class Status(Enum):
    PENDING = 1
    IN_PROGRESS = 2
    COMPLETED = 3

# Accessing the enumeration:
print(Status.PENDING)  # Output: Status.PENDING
print(Status.PENDING.name)  # Output: PENDING
print(Status.PENDING.value)  # Output: 1

# 87. Python's `pathlib` module for file paths
# The `pathlib` module provides an object-oriented approach to handling file system paths.

from pathlib import Path

# Example of using `pathlib` to manipulate paths:
path = Path("folder/subfolder/file.txt")
print(path.exists())  # Check if the path exists
print(path.is_file())  # Check if it's a file
print(path.parent)  # Output: folder/subfolder

# 88. Python's `abc` module (Abstract Base Classes)
# The `abc` module defines abstract base classes for defining common interfaces and enforcing them in subclasses.

from abc import ABC, abstractmethod

# Example of defining an abstract class:
class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14 * self.radius ** 2

circle = Circle(5)
print(circle.area())  # Output: 78.5

# 89. Python's `time` module and performance testing
# The `time` module provides various time-related functions, including performance testing with `time.time()` and `time.perf_counter()`.

import time

# Example of performance testing using `time.perf_counter()`:
start_time = time.perf_counter()
# Code to measure time for
for _ in range(1000000):
    pass
end_time = time.perf_counter()

print(f"Execution time: {end_time - start_time} seconds")

# 90. Python's `traceback` module for error reporting
# The `traceback` module helps capture and display error traceback information.

import traceback

# Example of catching and displaying exceptions with traceback:
try:
    1 / 0
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()

# 91. Python's `pytest` for unit testing
# `pytest` is a powerful tool for writing unit tests in Python, allowing for easier testing and assertions.

# Example of using `pytest` to write a simple test:
def test_addition():
    assert 1 + 1 == 2

# To run the test, you'd use the command `pytest` from the terminal.

# 92. Python's `mock` module for testing
# The `unittest.mock` module allows you to mock objects in your unit tests, so you can isolate the functionality you're testing.

from unittest.mock import MagicMock

# Example of mocking a function:
mock = MagicMock()
mock.return_value = 42
print(mock())  # Output: 42

# 93. Python's `os` module for environment variables
# The `os` module allows you to interact with environment variables in your Python code.

import os

# Example of getting and setting environment variables:
os.environ["MY_VAR"] = "value"
print(os.environ["MY_VAR"])  # Output: value

# 94. Python's `tkinter` for creating graphical user interfaces (GUIs)
# `tkinter` is the standard Python library for creating simple GUIs.

import tkinter as tk

# Example of creating a simple GUI with `tkinter`:
root = tk.Tk()
label = tk.Label(root, text="Hello, Tkinter!")
label.pack()
root.mainloop()

# 95. Python's `requests` library for HTTP requests
# The `requests` library is a popular third-party library for making HTTP requests.

import requests

# Example of making a simple GET request:
response = requests.get("https://jsonplaceholder.typicode.com/posts")
print(response.status_code)  # Output: 200 (OK)

# 96. Python's `pandas` for data manipulation and analysis
# The `pandas` library is a powerful tool for working with data frames and performing data analysis.

import pandas as pd

# Example of using pandas to create and manipulate a DataFrame:
data = {"name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]}
df = pd.DataFrame(data)
print(df)

# 97. Python's `numpy` for numerical computing
# `numpy` is a powerful library for numerical operations, especially with large arrays and matrices.

import numpy as np

# Example of using numpy to perform mathematical operations:
array = np.array([1, 2, 3, 4, 5])
print(np.sum(array))  # Output: 15

# 98. Python's `scipy` for scientific computing
# `scipy` builds on `numpy` to provide advanced scientific and technical computing functions.

from scipy import stats

# Example of using scipy to perform a statistical test:
data = [2.3, 2.9, 3.1, 3.6, 4.2]
mean = np.mean(data)
std_dev = np.std(data)
print(f"Mean: {mean}, Std Dev: {std_dev}")

# 99. Python's `sympy` for symbolic mathematics
# `sympy` provides capabilities for symbolic mathematics and algebraic manipulation.

from sympy import symbols, Eq, solve

# Example of solving an equation:
x = symbols('x')
equation = Eq(x**2 + 3*x - 4, 0)
solutions = solve(equation, x)
print(solutions)  # Output: [-4, 1]

# 100. Python's `multiprocessing` for parallel computing
# The `multiprocessing` module allows you to create multiple processes, which is useful for parallel computing.

import multiprocessing

# Example of using multiprocessing to parallelize tasks:
def square(n):
    return n * n

if __name__ == "__main__":
    with multiprocessing.Pool(4) as pool:
        results = pool.map(square, [1, 2, 3, 4, 5])
    print(results)  # Output: [1, 4, 9, 16, 25]

# 101. Python's `logging` module for logging messages
# The `logging` module allows for efficient logging of messages to track the application's state, errors, and events.

import logging

# Example of setting up logging:
logging.basicConfig(level=logging.DEBUG)

logging.debug("Debug message")
logging.info("Info message")
logging.warning("Warning message")
logging.error("Error message")
logging.critical("Critical message")

# 102. Python's `pickle` module for serializing objects
# The `pickle` module serializes Python objects into byte streams and can deserialize them back into objects.

import pickle

# Example of using `pickle` to serialize and deserialize objects:
data = {"name": "Alice", "age": 25}

# Serialize (pickling):
with open("data.pickle", "wb") as f:
    pickle.dump(data, f)

# Deserialize (unpickling):
with open("data.pickle", "rb") as f:
    loaded_data = pickle.load(f)
    print(loaded_data)  # Output: {'name': 'Alice', 'age': 25}

# 103. Python's `zip` function for iterating over multiple iterables
# The `zip` function combines multiple iterables into an iterator of tuples.

# Example of using `zip`:
names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 35]

for name, age in zip(names, ages):
    print(f"{name} is {age} years old.")

# 104. Python's `collections` module
# The `collections` module provides alternative container datatypes, such as `namedtuple`, `defaultdict`, `deque`, and `Counter`.

from collections import defaultdict, deque, Counter

# Example of using `defaultdict`:
d = defaultdict(int)
d["apple"] += 1
print(d)  # Output: defaultdict(<class 'int'>, {'apple': 1})

# Example of using `deque` (double-ended queue):
q = deque([1, 2, 3])
q.appendleft(0)
print(q)  # Output: deque([0, 1, 2, 3])

# Example of using `Counter`:
counter = Counter(["apple", "banana", "apple", "orange"])
print(counter)  # Output: Counter({'apple': 2, 'banana': 1, 'orange': 1})

# 105. Python's `functools` module for functional programming tools
# The `functools` module provides higher-order functions to manipulate other functions, such as `lru_cache`, `partial`, and `reduce`.

from functools import lru_cache, partial, reduce

# Example of using `lru_cache` for memoization:
@lru_cache(maxsize=None)
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print(fibonacci(30))  # Cached result for faster computation

# Example of using `partial` to create a new function with a default argument:
def multiply(x, y):
    return x * y

double = partial(multiply, 2)
print(double(5))  # Output: 10

# Example of using `reduce` to apply a function cumulatively:
result = reduce(lambda x, y: x + y, [1, 2, 3, 4])
print(result)  # Output: 10

# 106. Python's `memoryview` for memory-efficient data handling
# The `memoryview` object provides a way to access internal data of objects like `bytes` without copying.

# Example of using `memoryview`:
data = bytearray(b"Hello, World!")
view = memoryview(data)

print(view[0])  # Output: 72 (ASCII value of 'H')
view[0] = 88  # Modify the first byte (change 'H' to 'X')
print(data)  # Output: bytearray(b'Xello, World!')

# 107. Python's `namedtuple` for creating simple classes
# The `namedtuple` function provides a quick way to define a class with named fields, making it easier to access attributes by name.

from collections import namedtuple

# Example of using `namedtuple`:
Point = namedtuple('Point', ['x', 'y'])
point = Point(1, 2)
print(point.x)  # Output: 1
print(point.y)  # Output: 2

# 108. Python's `__slots__` for memory optimization in classes
# The `__slots__` attribute can be used to define a fixed set of attributes for a class, saving memory by preventing the creation of a `__dict__`.

class MyClass:
    __slots__ = ['x', 'y']

    def __init__(self, x, y):
        self.x = x
        self.y = y

# Example of memory optimization using `__slots__`:
obj = MyClass(1, 2)
print(obj.x)  # Output: 1

# 109. Python's `__del__` method for object destruction
# The `__del__` method is invoked when an object is about to be destroyed (garbage collected).

class MyClass:
    def __del__(self):
        print("Object is being destroyed!")

# Example of using `__del__`:
obj = MyClass()
del obj  # Output: Object is being destroyed!

# 110. Python's `assert` statement for debugging
# The `assert` statement tests a condition and raises an `AssertionError` if the condition is false.

# Example of using `assert` for debugging:
x = 5
assert x == 5  # This passes
assert x == 6  # This raises an AssertionError

# 111. Python's `contextlib` module for context managers
# The `contextlib` module provides utilities for creating context managers, including `contextmanager` and `closing`.

from contextlib import contextmanager

# Example of using `contextmanager`:
@contextmanager
def my_context():
    print("Entering the context")
    yield
    print("Exiting the context")

with my_context():
    print("Inside the context")

# Output:
# Entering the context
# Inside the context
# Exiting the context

# 112. Python's `timeit` module for performance measurement
# The `timeit` module allows you to measure the execution time of small code snippets.

import timeit

# Example of using `timeit`:
execution_time = timeit.timeit("x = 2 + 2", number=1000000)
print(f"Execution time: {execution_time} seconds")

# 113. Python's `os` module for directory and file manipulation
# The `os` module provides tools for interacting with the operating system, including working with directories and files.

import os

# Example of using `os` for directory manipulation:
os.mkdir("new_directory")
os.rmdir("new_directory")

# 114. Python's `dataclasses` for automatic class creation
# The `dataclasses` module provides a decorator to automatically add special methods to a class, such as `__init__`, `__repr__`, and `__eq__`.

from dataclasses import dataclass

# Example of using `dataclass`:
@dataclass
class Person:
    name: str
    age: int

person = Person("Alice", 25)
print(person)  # Output: Person(name='Alice', age=25)

# 115. Python's `pdb` module for debugging
# The `pdb` module is the built-in Python debugger, useful for setting breakpoints and inspecting variables during runtime.

# Example of using `pdb`:
import pdb

def my_function(x):
    pdb.set_trace()  # Set a breakpoint
    return x * 2

result = my_function(5)  # Debugger will pause at the breakpoint
```


