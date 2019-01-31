import tabulate
import csv
from plot_info import get_git_metadata, showAndSave
import numpy as np
import copy

class TableBuilder(object):
    def __init__(self):
        self.rows = []

        self.lower_header = None
        self.upper_header = None
        self.title = "No title"

    def set_lower_header(self, lower_header):
        self.lower_header = copy.deepcopy(lower_header)

    def set_upper_header(self, upper_header):
        self.upper_header = copy.deepcopy(upper_header)


    def set_header(self, header):
        self.set_upper_header(header)

    def add_row(self, row):
        self.rows.append(copy.deepcopy(row))


    def print_table(self, outname):

        if TableBuilder.disable_print:
            return
        data = []
        if self.upper_header is not None:
            data.append(copy.deepcopy(self.upper_header))
        if self.lower_header is not None:
            data.append(copy.deepcopy(self.lower_header))

        for r in self.rows:
            data.append(copy.deepcopy(r))

        multicolumn = self.lower_header is not None

        print_comparison_table(outname, data, multicolumn, self.title)

    def set_title(self, title):
        self.title = copy.deepcopy(title)

TableBuilder.disable_print =  'MACHINE_LEARNING_DO_NOT_PRINT_TABLES' in os.environ and os.environ['MACHINE_LEARNING_DO_NOT_PRINT_TABLES'].lower() == 'on'

def format_latex(d):

    if type(d) == str:
        return d
    elif type(d) == int:
        return "{}".format(d)
    else:
        return "{:.5f}".format(d)
def make_classical_table_multicolumn(data, start):
    table = ''

    data = copy.deepcopy(data)
    for r in data:
        for k in range(len(r)):
            if type(r[k]) == str:
                pass
                #r[k] = r[k].replace("_", "\_")



    for k in range(start, len(data[0])):
        data[0][k] = '\\multicolumn{2}{c|}{\\textbf{%s}}' % data[0][k]

    table += '\\begin{tabular}{%s%s|}\n'% ('|l'*start, '|c'*(len(data[1])-start))
    table += '\\hline\n'
    for k in range(0, start):
        table += '%s &' % data[0][k]

    table += '&'.join([data[0][k] for k in range(start, len(data[0]))])

    table += "\\\\ \n"
    cmidrules = [2]
    for k in range(len(data[0][start+1:])):
        cmidrules.append(cmidrules[k]+3)

    table += '\\hline\n'
    table += "\n"
    for r in range(1, len(data)):
        table += '%s &' % data[r][0]
        table += '&'.join(['&'.join([format_latex(data[r][k]), format_latex(data[r][k+1])]) for k in range(start, len(data[r]), 2)])
        table += "\\\\ \n"

        table += "\\hline\n"
    table += "\\end{tabular}\n"


    return table

def make_classical_table(data, start):
    table = ''

    data = copy.deepcopy(data)
    for r in data:
        for k in range(len(r)):
            if type(r[k]) == str:
                r[k] = r[k].replace("_", "\_")




    table += '\\begin{tabular}{%s%s|}\n'% ('|l'*start, '|c'*(len(data[1])-start))
    table += '\\hline\n'
    for k in range(0, start):
        table += '%s &' % data[0][k]

    table += '&'.join([data[0][k] for k in range(start, len(data[0]))])

    table += "\\\\ \n"
    cmidrules = [2]
    for k in range(len(data[0][start+1:])):
        cmidrules.append(cmidrules[k]+3)

    table += '\\hline\n'
    table += "\n"
    for r in range(1, len(data)):
        table += " & ".join([format_latex(d) for d in data[r]])
        table += "\\\\ \n"

        table += "\\hline\n"
    table += "\\end{tabular}\n"
    return table


def make_booktabs_table_multicolumn(data, start):
        table = ''

        data = copy.deepcopy(data)
        for r in data:
            for k in range(len(r)):
                if type(r[k]) == str:
                    r[k] = r[k].replace("_", "\_")



        for k in range(start, len(data[0])):
            data[0][k] = '\\multicolumn{2}{c}{\\textbf{%s}}' % data[0][k]

        table += '\\begin{tabular}{%s%s}\n'% ('l'*start, 'c'*2*(len(data[1])-start))
        table += '\\toprule\n'
        for k in range(0, start):
            table += '%s &' % data[0][k]

        table += '&&'.join([data[0][k] for k in range(start, len(data[0]))])

        table += "\\\\ \n"
        cmidrules = [2]
        for k in range(len(data[0][start+1:])):
            cmidrules.append(cmidrules[k]+3)

        table += " ".join(['\\cmidrule(r){%d-%d}' % (cmidrules[k], cmidrules[k]+1) for k in range(len(data[0][start+1:]))])
        table += "\n"
        for r in range(1, len(data)):
            table += '%s &' % data[r][0]
            table += '&&'.join(['&'.join([format_latex(data[r][k]), format_latex(data[r][k+1])]) for k in range(start, len(data[r]), 2)])
            table += "\\\\ \n"
            if r == 1:
                table += "\\midrule\n"
        table += "\\bottomrule\n"
        table += "\\end{tabular}\n"
        return table



def print_comparison_table(outname, data, multicolumn = False, title= "No title"):
    outname = outname.lower()
    if showAndSave.prefix != '':
        outname = '%s_%s' % (showAndSave.prefix, outname)
    outname.replace(" ", "_")
    outname = ''.join(ch for ch in outname if ch.isalnum() or ch =='_')

    start = 0
    data = copy.deepcopy(data)

    if ( not multicolumn and len(data[1]) % len(data[0]) != 0) \
        or (multicolumn and len(data[2]) % len(data[1]) != 0):
        start  = 1
        data[0] = [""] + data[0]

        if multicolumn:
            data[1] = [""] + data[1]


    if multicolumn:
        latex = make_booktabs_table_multicolumn(data, start)
    else:
        latex = tabulate.tabulate(data[1:], headers=data[0],tablefmt="latex_booktabs")




    with open('tables/%s.tex' % outname, 'w') as f:

        f.write("%% INCLUDE THE COMMENTS AT THE END WHEN COPYING\n")
        f.write("%%%%%%%%%%%%%TITLE%%%%%%%%%%%%%%%%%\n")
        for line in title.splitlines():
            f.write("%% {}\n".format(line))
        f.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
        f.write(latex)
        git_metadata = get_git_metadata()
        f.write("\n")
        f.write("%% ALWAYS INCLUDE THE COMMENTS WHEN COPYING THIS TABLE\n")
        f.write("%% DO NOT REMOVE THE COMMNENTS BELOW!\n")
        for k in git_metadata.keys():

            f.write("%% GIT {} : {}\n".format(k, git_metadata[k]))

    if print_comparison_table.callback is not None:
        print_comparison_table.callback('tables/%s.tex' % outname, title)

    if not multicolumn:
        latex_classical = make_classical_table(data, start)
    else:
        latex_classical = make_classical_table_multicolumn(data, start)

    with open('tables/%s_classical.tex' % outname, 'w') as f:
        f.write("%% INCLUDE THE COMMENTS AT THE END WHEN COPYING\n")
        f.write("%%%%%%%%%%%%%TITLE%%%%%%%%%%%%%%%%%\n")
        for line in title.splitlines():
            f.write("%% {}\n".format(line))
        f.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
        f.write(latex_classical)
        git_metadata = get_git_metadata()
        f.write("\n")
        f.write("%% ALWAYS INCLUDE THE COMMENTS WHEN COPYING THIS TABLE\n")
        f.write("%% DO NOT REMOVE THE COMMNENTS BELOW!\n")
        for k in git_metadata.keys():

            f.write("%% GIT {} : {}\n".format(k, git_metadata[k]))

    if multicolumn:
        new_header = []
        for h in data[0][:start]:
            new_header.append(h)
        for h in data[0][start:]:
            new_header.append(h)
            new_header.append(h)



        data[0] = new_header


    github  = tabulate.tabulate(data, tablefmt='github')
    if not print_comparison_table.silent:
        print("")
        print("")

        print('#'*(len(outname)+2))
        print('#%s#' % outname)
        print('#'*(len(outname)+2))
        print(github)
        print('#'*(len(outname)+2))
        print("")
        print("")
    with open('tables/%s.github' % outname,'w') as f:
        f.write(github)


    with open('tables/%s.csv' % outname, 'w') as f:
        writer = csv.writer(f)
        for r in data:
            writer.writerow(r)

print_comparison_table.silent = False
print_comparison_table.callback = None

def print_keras_model_as_table(outname, model):
    data = [["Layer", "Size", "Parameters"]]

    sum_parameters = 0
    for (n, l) in enumerate(model.layers):
        input_dim = int(np.prod(l.input_shape[1:]))
        output_dim = int(np.prod(l.output_shape[1:]))


        parameters = input_dim*output_dim + output_dim
        sum_parameters+= parameters
        data.append([n, output_dim, parameters])

    data.append(["Sum", "", sum_parameters])
    print_comparison_table(outname, data)
