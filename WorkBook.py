import os
import sys
import pandas as pds

class WorkBook(object):
    def __init__(self, num_exp, fraction_list):
        first_column = [i+1 for i in range(num_exp)]
        first_column.append("Average")
        row_num = len(first_column)
        first_row = [first_column[0]]
        for i in fraction_list:
            first_row.append(i)
        column_num = len(first_row)
        self.write_data = {'实验序号/选择比例': first_column}
        for i in range(1, column_num):
            self.write_data[first_row[i]] = ['' for j in range(row_num)]
    def to_excel(self, path='./data.xlsx'):
        # os.remove(path)
        df = pds.DataFrame(self.write_data)
        df.to_excel(path, index=False)
    def append(self, column_id, index, data):
        self.write_data[column_id][index] = data
if __name__ == '__main__':

    wb = WorkBook(20, [1.0, 2.0])
    wb.append(1.0, 0, 0.99)
    wb.to_excel()
    print("run end")