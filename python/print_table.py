import tabulate
import csv
from plot_info import get_git_metadata, showAndSave
import numpy as np
def print_comparison_table(outname, data, multicolumn = False):
    if showAndSave.prefix != '':
        outname = '%s_%s' % (showAndSave.prefix, outname)
    
    latex = tabulate.tabulate(data, tablefmt="latex_booktabs")

    if multicolumn:
        for k in data[0]:
            latex = latex.replace(k, '\\\multicolumn{2}{c}{\\textbf{%s}}' % k)


    with open('tables/%s.tex' % outname, 'w') as f:
        f.write(latex)
        git_metadata = get_git_metadata()
        for k in git_metadata.keys():
            f.write("%% GIT {} : {}\n".format(k, git_metadata[k]))

    if multicolumn:
        new_header = []
        for h in data[0]:
            new_header.append(h)
            new_header.append(h)



        data[0] = new_header
        

    github  = tabulate.tabulate(data, tablefmt='github')
    print(github)
    with open('tables/%s.github' % outname,'w') as f:
        f.write(github)
        

    with open('tables/%s.csv' % outname, 'w') as f:
        writer = csv.writer(f)
        for r in data:
            writer.writerow(r)
    
    
    
def print_keras_model_as_table(outname, model):
    data = [["Layer", "Size", "Parameters"]]

    for (n, l) in enumerate(model.layers):
        input_dim = int(np.prod(l.input_shape[1:]))
        output_dim = int(np.prod(l.output_shape[1:]))
        

        parameters = input_dim*output_dim + output_dim

        data.append([n, output_dim, parameters])
    print_comparison_table(outname, data)
