import openpyxl as op

wb = op.load_workbook("socrativeQuizTemplate.xlsx")

sh = wb.get_sheet_by_name('Quick Quiz')

sh['B3'] = "This is our quiz"

sh['A7'] = "Multiple choice"
sh['B7'] = "Who is this?"
sh['C7'] = "Einstein"
sh['D7'] = "Curie"
sh['E7'] = "Newton"
sh['F7'] = "Pauli"
sh['G7'] = "Darwin"
sh['H7'] = "A"

wb.save(filename = "test.xlsx")

