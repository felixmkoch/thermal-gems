import csv
import os

def combination_exists_dict(output_csv_name, d):

    try: 
        with open(output_csv_name, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:

                    found = True

                    for k, v in d.items():
                        if not row.get(k) == v:
                            found = False
                            break
                    
                    if found:
                        return True
                    
    except FileNotFoundError:
        return False
                
    return False



def writerow(output_csv_name, csv_cols, result):
    file_exists = os.path.isfile(output_csv_name)

    with open(output_csv_name, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_cols)

        # If the file is new, write the header first
        if not file_exists:
            writer.writeheader()

        # Write the row
        writer.writerow(result)
