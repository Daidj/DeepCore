import os
import sys
import pandas as pds

class WorkBook(object):
    def __init__(self, num_exp):
        first_column = [i+1 for i in range(num_exp)]
        # first_column.append("Average")
        row_num = len(first_column)
        # first_row = [first_column[0]]
        # for i in fraction_list:
        #     first_row.append(i)
        # column_num = len(first_row)
        self.write_data = {
            '实验序号': first_column,
            '样本数量': ['' for i in range(row_num)],
            '准确度': ['' for i in range(row_num)],
            '总时间': ['' for i in range(row_num)],
            '算法时间': ['' for i in range(row_num)],
            'MMD时间': ['' for i in range(row_num)],
            'MMD距离': ['' for i in range(row_num)],
            '备注': ['' for i in range(row_num)]
        }
        # for i in range(1, column_num):
        #     self.write_data[first_row[i]] = ['' for j in range(row_num)]
    def to_excel(self, path='./data.xlsx'):
        # os.remove(path)
        df = pds.DataFrame(self.write_data)
        df.to_excel(path, index=False)
    def append(self, column_id, index, data):
        self.write_data[column_id][index] = data
if __name__ == '__main__':

    wb = WorkBook(20)
    wb.append('样本数量', 0, 60000)
    wb.append('备注', 0, "测试")
    wb.to_excel('./excel/test.xlsx')
    print("run end")