import os
import re
import shutil
import openpyxl
import pandas


class ReportGenerator:
    def __init__(self):
        pass

    def convert_wb(self, old_wb_path, template_wb_path, output_wb_path):
        # Open existing workbook with data
        old_wb = openpyxl.load_workbook(old_wb_path)
        # Create copy of template for data
        shutil.copy(template_wb_path, output_wb_path)
        # Copy data from existing workbook into the new template wb
        output_wb = openpyxl.load_workbook(output_wb_path)


    def append_values(self, ws: openpyxl.worksheet.worksheet.Worksheet, values):
        """
        Adds values to the bottom of the provided sheet
        :param ws:
        :param values:
        :return:
        """
        ws.append(values)

    def create_sheet(self, wb: openpyxl.workbook.Workbook, title, ix):
        wb.create_sheet(title, ix)

    def modify_cell(self, sheet: openpyxl.worksheet.worksheet.Worksheet,
                    m, n, value):
        """
        https://openpyxl.readthedocs.io/en/stable/api/openpyxl.worksheet.worksheet.html?highlight=iter_cols#openpyxl.worksheet.worksheet.Worksheet.iter_cols
        1-based row and column indexing
        :param sheet:
        :param m:
        :param n:
        :param value:
        :return:
        """
        # 1-based indexing
        sheet.cell(row=m, column=n).value = value

    def get_coord_from_str(self, ix_str):
        xy = openpyxl.utils.cell.coordinate_from_string(ix_str)
        col = openpyxl.utils.cell.column_index_from_string(xy[0])
        row = xy[1]
        return col, row

    def get_sheet_dims(self, ws: openpyxl.worksheet.worksheet.Worksheet):
        x, y = re.split(':', ws.calculate_dimension())
        col_1, row_1 = self.get_coord_from_str(x)
        col_2, row_2 = self.get_coord_from_str(y)
        return [col_1, row_1], [col_2, row_2]


def main():
    rg = ReportGenerator()
    # Paths and filenames
    path = r'C:\Users\gsass\Documents\My Tableau Repository\Datasources\fall_risk_assessments\cohort_1_fall_risk_metrics.xlsx'
    folder = r'C:\Users\gsass\Documents\My Tableau Repository\Datasources\fall_risk_assessments'
    new_file_path = r'C:\Users\gsass\Documents\My Tableau Repository\Datasources\fall_risk_assessments\NEW_cohort_1_fall_risk_metrics.xlsx'

    # Create copy of the spreadsheet
    shutil.copy(path, new_file_path)
    # Open the new copy and modify the sheet values
    wb = openpyxl.load_workbook(new_file_path)
    user_uuids = ['data_abcd-1234']
    # Index the sheet names with UUIDs
    ws = wb[user_uuids[0]]
    ws.cell(row=2, column=11).value = 15
    dim1, dim2 = rg.get_sheet_dims(ws)
    for i in ws.columns:
        print(i)
    # start creating basic functions to copy columns, modify values etc to prep for the conversion of MR stuff

    # data = pandas.read_excel(new_file_path, engine='openpyxl')
    # with open(new_file_path, 'w') as xl_file:
    #     data = pandas.read_excel(xl_file)
    # Save off the modifications made to the document
    wb.save(new_file_path)
    # Publish the dashboard as a PDF


if __name__ == '__main__':
    main()