import os
import shutil
import openpyxl
import pandas

# Paths and filenames
path = r'C:\Users\gsass\Documents\My Tableau Repository\Datasources\fall_risk_assessments\cohort_1_fall_risk_metrics.xlsx'
folder = r'C:\Users\gsass\Documents\My Tableau Repository\Datasources\fall_risk_assessments'
new_file_path = r'C:\Users\gsass\Documents\My Tableau Repository\Datasources\fall_risk_assessments\NEW_cohort_1_fall_risk_metrics.xlsx'

# Create copy of the spreadsheet
shutil.copy(path, new_file_path)
# Open the new copy and modify the sheet values
wb = openpyxl.load_workbook(new_file_path)
user_uuids = ['abcd-1234', 'efgh-5678']
# Index the sheet names with UUIDs
ws = wb[user_uuids[1]]
ws.cell(row=2,column=5).value = 22

# data = pandas.read_excel(new_file_path, engine='openpyxl')
# with open(new_file_path, 'w') as xl_file:
#     data = pandas.read_excel(xl_file)
# Save off the modifications made to the document
wb.save(new_file_path)
# Publish the dashboard as a PDF