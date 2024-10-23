import pandas as pd

from anonymization_functions import anonymize_fullname, anonymize_passport, anonymize_snils, anonymize_symptoms, anonymize_doctor, anonymize_analyses, anonymize_cost, anonymize_card

df = pd.read_csv("1 Lab/dataset.csv")

fieldnames = ["ФИО", "Паспортные данные", "СНИЛС", "Симптомы", "Выбор врача", "Дата посещения врача", "Анализы", "Дата получения анализов", "Стоимость анализов", "Карта оплаты"]

def input_quasi_identifiers():
    quasi_identifiers = []
    for field in fieldnames:
        while True:
            answer = str(input(f"Нужно ли обезличивать {field}? y/n: "))
            if answer.lower() in ['y', 'д']:
                quasi_identifiers.append(field)
                break
            elif answer.lower() in ['n', 'н']:
                break   
            else:
                print("Нужно ввести y или n!")
    return quasi_identifiers

def anonymize_data(quasi_identifiers):
    if not quasi_identifiers:
        print("Нет полей для обезличивания.")
        return

    for field in quasi_identifiers:
        if field == 'ФИО':
            df['ФИО'] = df['ФИО'].apply(anonymize_fullname)
        elif field == 'Паспортные данные':
            df['Паспортные данные'] = df['Паспортные данные'].apply(anonymize_passport)
        elif field == 'СНИЛС':
            df['СНИЛС'] = df['СНИЛС'].apply(anonymize_snils)
        elif field == 'Симптомы':
            df['Симптомы'] = df['Симптомы'].apply(anonymize_symptoms)
        elif field == 'Выбор врача':
            df['Выбор врача'] = df['Выбор врача'].apply(anonymize_doctor)
        elif field == 'Дата посещения врача':
            df['Дата посещения врача'] = pd.to_datetime(df['Дата посещения врача']).dt.year
        elif field == 'Анализы':
            df['Анализы'] = df['Анализы'].apply(anonymize_analyses)
        elif field == 'Дата получения анализов':
            df['Дата получения анализов'] = pd.to_datetime(df['Дата получения анализов']).dt.year
        elif field == 'Стоимость анализов':
            df['Стоимость анализов'] = df['Стоимость анализов'].apply(anonymize_cost)
        elif field == 'Карта оплаты':
            df['Карта оплаты'] = df['Карта оплаты'].apply(anonymize_card)

def calculate_k_anonymity():
    grouped = df.groupby(fieldnames).size().reset_index(name='count')

    record_count = len(df)
    if record_count <= 51000:
        acceptable_k = 10
    elif record_count <= 105000:
        acceptable_k = 7
    else:
        acceptable_k = 5

    bad_k_values = grouped[grouped['count'] < acceptable_k]
    good_k_values = grouped[grouped['count'] >= acceptable_k]

    return bad_k_values, good_k_values

def remove_bad_k_anonymity_records(df, bad_k_values):
    merged_df = df.merge(bad_k_values, on=fieldnames, how='left', indicator=True)
    df_cleaned = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge'])
    removed_records = len(df) - len(df_cleaned)

    print(f"Удалено записей: {removed_records}, что составляет {removed_records / len(df) * 100:.2f}% от общего количества")
    
    return df_cleaned

def print_k_values(bad_k_values, good_k_values):
    record_count = len(df)

    if bad_k_values.empty:
        unique_k_values = good_k_values['count'].unique()
        sorted_unique_k_values = sorted(unique_k_values)
        min_k_values = sorted_unique_k_values[:5]

        for k_value in min_k_values:
            count = good_k_values[good_k_values['count'] == k_value].count().values[0]
            percentage = (count / record_count) * 100
            print(f'K-anonymity: {k_value} ({percentage:.4f})')

    unique_k_values = bad_k_values['count'].unique()
    sorted_unique_k_values = sorted(unique_k_values)
    min_k_values = sorted_unique_k_values[:5]

    for k_value in min_k_values:
        count = bad_k_values[bad_k_values['count'] == k_value].count().values[0]
        percentage = (count / record_count) * 100
        print(f'K-anonymity: {k_value} ({percentage:.4f})')

if __name__ == "__main__":
    quasi_identifiers = input_quasi_identifiers()
    anonymize_data(quasi_identifiers)

    bad_k_values, good_k_values = calculate_k_anonymity()
    bad_k_values.to_csv('2 Lab/bad_k_values.csv', index=False, encoding='utf-8')
    good_k_values.to_csv('2 Lab/good_k_values.csv', index=False, encoding='utf-8')

    if (0 < len(bad_k_values) < len(df) * 0.05):
        df = remove_bad_k_anonymity_records(df, bad_k_values)
        bad_k_values, good_k_values = calculate_k_anonymity()
    
    print_k_values(bad_k_values, good_k_values)

    df.to_csv('2 Lab/anon_dataset.csv', index=False, encoding='utf-8')
    print("Данные успешно сохранены в файл 'anon_dataset.csv'.")
