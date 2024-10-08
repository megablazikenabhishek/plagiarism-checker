import os
import random

# Folder to store the C++ code files
dataset_folder = "dataset"
if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)

# Base C++ code snippets
cpp_code_snippets = [
    """#include <iostream>
int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}""",
    """#include <iostream>
using namespace std;
int main() {
    cout << "Sum: " << (5 + 3) << endl;
    return 0;
}""",
    """#include <iostream>
void greet() {
    std::cout << "Greetings!" << std::endl;
}
int main() {
    greet();
    return 0;
}""",
    """#include <iostream>
int factorial(int n) {
    return (n <= 1) ? 1 : n * factorial(n - 1);
}
int main() {
    int num = 5;
    std::cout << "Factorial of " << num << " is " << factorial(num) << std::endl;
    return 0;
}""",
    """#include <iostream>
int main() {
    for(int i = 0; i < 10; ++i) {
        std::cout << i << " ";
    }
    return 0;
}"""
]

# Function to add random modifications to the base C++ code snippets
def add_modification_to_code(code, variation):
    if variation == 1:
        return code.replace("std::cout", "printf")
    elif variation == 2:
        return code.replace("int main()", "int main(int argc, char** argv)")
    elif variation == 3:
        return code.replace("5", str(random.randint(6, 10)))
    else:
        return code

# Number of files to generate
num_files = 100  # Increase this number for larger datasets

# Generate the C++ code files with variations
for i in range(1, num_files + 1):
    base_code = random.choice(cpp_code_snippets)  # Randomly select a base snippet
    modified_code = add_modification_to_code(base_code, random.randint(1, 4))  # Apply a random modification
    
    # Write the generated code to a new file
    with open(f"{dataset_folder}/file{i}.cpp", "w") as file:
        file.write(modified_code)

print(f"Generated {num_files} C++ files in the 'dataset' folder.")
