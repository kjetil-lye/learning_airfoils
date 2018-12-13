
import numpy
import glob
import sys

import matplotlib
matplotlib.rcParams['savefig.dpi']=600
from numpy import *
import matplotlib.pyplot as plt

import matplotlib2tikz
import PIL
import socket
import os
import os.path
import datetime
import traceback
from IPython.core.display import display, HTML
try:
    import git
    def get_git_metadata():
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        modified= repo.is_dirty()
        activeBranch = repo.active_branch
        url = repo.remotes.origin.url

        return {'git_commit': str(sha),
                'git_repo_modified':str(modified),
                'git_branch' : str(activeBranch),
                'git_remote_url' : str(url)}


    def add_git_information(filename):
        writeMetadata(filename, get_git_metadata())

except:
    def add_git_information(filename):
        pass

    def get_git_metadata():
        return {'git_commit': 'unknown',
                'git_repo_modified':'unknown',
                'git_branch' : 'unknown',
                'git_remote_url' : 'unknown'}

# From https://stackoverflow.com/a/6796752
class RedirectStdStreams(object):
    def __init__(self, stdout=None, stderr=None):
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush(); self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush(); self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr


class RedirectStdStreamsToNull(object):
    def __init__(self):
        self._devnull = open(os.devnull, 'w')
        self._redirect_stream = RedirectStdStreams(self._devnull, self._devnull)

    def __enter__(self):
        self._redirect_stream.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        self._redirect_stream.__exit__(exc_type, exc_value, traceback)
        self._devnull.close()





def writeMetadata(filename, data):
    im = PIL.Image.open(filename)

    meta = PIL.PngImagePlugin.PngInfo()

    for key in data.keys():
        meta.add_text(key, str(data[key]))
    im.save(filename, "png", pnginfo=meta)

def only_alphanum(s):
    return ''.join(ch for ch in s if ch.isalnum() or ch =='_')
def savePlot(name):
    name = showAndSave.prefix + name
    name = ''.join(ch for ch in name if ch.isalnum() or ch =='_')


    fig = plt.gcf()
    ax = plt.gca()
    gitMetadata = get_git_metadata()
    informationText = 'By Kjetil Lye@ETHZ <kjetil.o.lye@gmail.com>\nand Siddhartha Mishra@ETHZ <siddhartha.mishra@sam.math.ethz.ch>\nand Deep Ray@EPFL<deep.ray@gmail.com>\nCommit: %s\nRepo: %s\nHostname: %s' % (gitMetadata['git_commit'], gitMetadata['git_remote_url'], socket.gethostname())

    ax.text(0.95, 0.01, informationText,
         fontsize=3, color='gray',
         ha='right', va='bottom', alpha=0.5, transform=ax.transAxes)

    # We don't want all the output from matplotlib2tikz

    with RedirectStdStreamsToNull():
        matplotlib2tikz.save('img_tikz/' + name + '.tikz',
           figureheight = '\\figureheight',
           figurewidth = '\\figurewidth',
           show_info = False)


    savenamepng = 'img/' + name + '.png'
    plt.savefig(savenamepng, bbox_inches='tight')

    writeMetadata(savenamepng, {'Copyright' : 'Copyright, Deep Ray@EPFL<deep.ray@gmail.com> and Kjetil Lye@ETHZ <kjetil.o.lye@gmail.com>'
                               })

    add_git_information(savenamepng)
    writeMetadata(savenamepng, {'working_directory': os.getcwd(),
                                'hostname':socket.gethostname(),
                                'generated_on_date': str(datetime.datetime.now())})
def showAndSave(name):
    savePlot(name)
    if not showAndSave.silent:
        plt.show()
    plt.close()

showAndSave.prefix=''
showAndSave.silent=False


def legendLeft():
    ax = plt.gca()
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

import inspect
def console_log_show(x):
    try:
        x = "{} (in {}): {}".format(str(datetime.datetime.now()), inspect.stack()[1][3], x)
    except:
        x = "{} (in unknown function): {}".format(str(datetime.datetime.now()), x)
    console_log(x)
    print(x)

def console_log(x):
    """Simple hack to write to stdout from a notebook"""

    x=str(x)
    with open('/dev/stdout', 'w') as f:
        f.write("DEBUG: %s\n"%x)
        f.flush()
