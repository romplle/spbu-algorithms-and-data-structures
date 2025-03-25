import hashlib

salt = ["1", "123"]

with open("3 Lab/deanonymized_phones.txt", 'r') as file:
    with open("3 Lab/md5_small_salt_hashes.txt", 'w') as md5_small_file, \
         open("3 Lab/sha1_small_salt_hashes.txt", 'w') as sha1_small_file, \
         open("3 Lab/blake2b_small_salt_hashes.txt", 'w') as blake2b_small_file, \
         open("3 Lab/md5_large_salt_hashes.txt", 'w') as md5_large_file, \
         open("3 Lab/sha1_large_salt_hashes.txt", 'w') as sha1_large_file, \
         open("3 Lab/blake2b_large_salt_hashes.txt", 'w') as blake2b_large_file:

        for line in file:
            phone_number = line.strip()

            salted_phone_small = (phone_number + salt[0]).encode()
            md5_small_hash = hashlib.md5(salted_phone_small).hexdigest()
            sha1_small_hash = hashlib.sha1(salted_phone_small).hexdigest()
            blake2b_small_hash = hashlib.blake2b(salted_phone_small).hexdigest()

            md5_small_file.write(f'{md5_small_hash}\n')
            sha1_small_file.write(f'{sha1_small_hash}\n')
            blake2b_small_file.write(f'{blake2b_small_hash}\n')

            salted_phone_large = (phone_number + salt[1]).encode()
            md5_large_hash = hashlib.md5(salted_phone_large).hexdigest()
            sha1_large_hash = hashlib.sha1(salted_phone_large).hexdigest()
            blake2b_large_hash = hashlib.blake2b(salted_phone_large).hexdigest()

            md5_large_file.write(f'{md5_large_hash}\n')
            sha1_large_file.write(f'{sha1_large_hash}\n')
            blake2b_large_file.write(f'{blake2b_large_hash}\n')

print("Хэшированные данные сохраняются в соответствующие файлы с маленькой и крупной солями.")

# Тестирование
# hashcat -m 0 -a 3 -O --opencl-device-types 2 -w 3 md5_small_salt_hashes.txt 89?d?d?d?d?d?d?d?d?d?d -o md5_cracked_small_salt_hashes.txt
# hashcat -m 100 -a 3 -O --opencl-device-types 2 -w 3 sha1_small_salt_hashes.txt 89?d?d?d?d?d?d?d?d?d?d -o sha1_cracked_small_salt_hashes.txt
# hashcat -m 600 -a 3 -O --opencl-device-types 2 -w 3 blake2b_small_salt_hashes.txt 89?d?d?d?d?d?d?d?d?d?d -o blake2b_cracked_small_salt_hashes.txt

# hashcat -m 0 -a 3 -O --opencl-device-types 2 -w 3 md5_large_salt_hashes.txt 89?d?d?d?d?d?d?d?d?d?d?d?d -o md5_cracked_large_salt_hashes.txt
# hashcat -m 100 -a 3 -O --opencl-device-types 2 -w 3 sha1_large_salt_hashes.txt 89?d?d?d?d?d?d?d?d?d?d?d?d -o sha1_cracked_large_salt_hashes.txt
# hashcat -m 600 -a 3 -O --opencl-device-types 2 -w 3 blake2b_large_salt_hashes.txt 89?d?d?d?d?d?d?d?d?d?d?d?d -o blake2b_cracked_large_salt_hashes.txt