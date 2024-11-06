# hashcat -m 0 -a 3 -O --opencl-device-types 2 -w 3 initial_hashes.txt 89?d?d?d?d?d?d?d?d?d -o cracked_hashes.txt

phones = open("3 Lab/phones.txt").read().split()
phones = {int(i) for i in phones}

results = open("3 Lab/cracked_hashes.txt").read().split()
results = [int(i[-11:]) for i in results]

# hash = phone + salt
# salt = hash - phone
# phone = hash - salt

salt = None
for i in range(len(results)):
    temp_salt = results[i] - next(iter(phones))
    match_count = 0
    for j in range(len(results)):
        if results[j] - temp_salt in phones:
            match_count += 1
    if match_count == len(phones):
        salt = temp_salt
        break

if salt is not None:
    print(f"Найдена соль: {salt}")

    with open("3 Lab/deanonymized_phones.txt", 'w') as file:
        for result in results:
            file.write(f"{result - salt}\n")
    print("Результаты сохранены в deanonymized_phones.txt")
else:
    print("Соль не найдена")

# Для 5, 4 и 3 номеров: "Найдена соль: 58644554"
# Для 2 номеров: "Найдена соль: 58644554"
