def print_positive_numbers(numbers):
    positive_numbers = [num for num in numbers if num > 0]
    result = ''
    for i in range(len(positive_numbers)):
        result += str(positive_numbers[i])
        if i < len(positive_numbers) - 1:
            result += ' '
    print(result)

list1 = [12, -7, 5, 64, -14]
list2 = [12, 14, -95, 3]

print_positive_numbers(list1)
print_positive_numbers(list2)