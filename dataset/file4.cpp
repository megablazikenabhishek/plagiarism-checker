#include <iostream>
int factorial(int n) {
    return (n <= 1) ? 1 : n * factorial(n - 1);
}
int main() {
    int num = 10;
    std::cout << "Factorial of " << num << " is " << factorial(num) << std::endl;
    return 0;
}