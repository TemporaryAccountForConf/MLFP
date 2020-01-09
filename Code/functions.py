
#################
#               #
# Data recovery #
#               #
#################

def data_recovery(opt, date):
    import pandas as pd
    import numpy as np
    import os

    if opt.dataset in ['abalone8', 'abalone17', 'abalone20']:
        data = pd.read_csv("../Datasets/abalone.data", header=None)

        data = pd.get_dummies(data, dtype=float)

        if opt.dataset in ['abalone8']:
            y = np.array([1 if elt == 8 else 0 for elt in data[8]])
        elif opt.dataset in ['abalone17']:
            y = np.array([1 if elt == 17 else 0 for elt in data[8]])
        elif opt.dataset in ['abalone20']:
            y = np.array([1 if elt == 20 else 0 for elt in data[8]])
        X = np.array(data.drop([8], axis=1))

    elif opt.dataset in ['autompg']:
        data = pd.read_csv("../Datasets/auto-mpg.data", header=None,
                           sep=r'\s+')

        data = data.replace('?', np.nan)
        data = data.dropna()
        data = data.drop([8], axis=1)

        y = np.array([1 if elt in [2, 3] else 0 for elt in data[7]])
        X = np.array(data.drop([7], axis=1))

    elif opt.dataset in ['balance']:
        data = pd.read_csv("../Datasets/balance-scale.data", header=None)

        y = np.array([1 if elt in ['L'] else 0 for elt in data[0]])
        X = np.array(data.drop([0], axis=1))

    elif opt.dataset in ['german']:
        data = pd.read_csv("../Datasets/german.data-numeric", header=None,
                           sep=r'\s+')

        y = np.array([1 if elt == 2 else 0 for elt in data[24]])
        X = np.array(data.drop([24], axis=1))

    elif opt.dataset in ['glass']:
        data = pd.read_csv("../Datasets/glass.data", header=None, index_col=0)

        y = np.array([1 if elt == 1 else 0 for elt in data[10]])
        X = np.array(data.drop([10], axis=1))

    elif opt.dataset in ['hayes']:
        data = pd.read_csv("../Datasets/hayes-roth.data", header=None)

        y = np.array([1 if elt in [3] else 0 for elt in data[5]])
        X = np.array(data.drop([0, 5], axis=1))

    elif opt.dataset in ['iono']:
        data = pd.read_csv("../Datasets/ionosphere.data", header=None)

        y = np.array([1 if elt in ['b'] else 0 for elt in data[34]])
        X = np.array(data.drop([34], axis=1))

    elif opt.dataset in ['libras']:
        data = pd.read_csv("../Datasets/movement_libras.data", header=None)

        y = np.array([1 if elt in [1] else 0 for elt in data[90]])
        X = np.array(data.drop([90], axis=1))

    elif opt.dataset in ['pageblocks']:
        data = pd.read_csv("../Datasets/page-blocks.data", header=None,
                           sep=r'\s+')

        y = np.array([1 if elt in [2, 3, 4, 5] else 0 for elt in data[10]])
        X = np.array(data.drop([10], axis=1))

    elif opt.dataset in ['pima']:
        with open("../Datasets/pima.dat", 'r') as f:
            content = f.read()
            content = content.split('\n')
            content = content[13:]
            content = '\n'.join(content)
        with open("../Datasets/temp.data", 'w') as f:
            f.write(content)

        data = pd.read_csv("../Datasets/temp.data", header=None)
        os.remove("../Datasets/temp.data")

        y = np.array([1 if elt in ['positive'] else 0 for elt in data[8]])
        X = np.array(data.drop([8], axis=1))

    elif opt.dataset in ['satimage']:
        with open("../Datasets/sat.trn", 'r') as f:
            content = f.read()
        with open("../Datasets/temp.data", 'w') as f:
            f.write(content)
        with open("../Datasets/sat.tst", 'r') as f:
            content = f.read()
        with open("../Datasets/temp.data", 'a') as f:
            f.write(content)

        data = pd.read_csv("../Datasets/temp.data", header=None, sep=r'\s+')
        os.remove("../Datasets/temp.data")

        y = np.array([1 if elt in [4] else 0 for elt in data[36]])
        X = np.array(data.drop([36], axis=1))

    elif opt.dataset in ['segmentation']:
        with open("../Datasets/segmentation.data", 'r') as f:
            content = f.read()
            content = content.split('\n')
            content = content[5:]
            content = '\n'.join(content)
        with open("../Datasets/temp.data", 'w') as f:
            f.write(content)
        with open("../Datasets/segmentation.test", 'r') as f:
            content = f.read()
            content = content.split('\n')
            content = content[5:]
            content = '\n'.join(content)
        with open("../Datasets/temp.data", 'a') as f:
            f.write(content)

        data = pd.read_csv("../Datasets/temp.data", header=None)
        os.remove("../Datasets/temp.data")

        y = np.array([1 if elt == 'WINDOW' else 0 for elt in data[0]])
        X = np.array(data.drop([0], axis=1))

    elif opt.dataset in ['vehicle']:
        with open("../Datasets/xaa.dat", 'r') as f:
            content = f.read()
        with open("../Datasets/temp.data", 'w') as f:
            f.write(content)
        for file in ['xab', 'xac', 'xad', 'xae', 'xaf', 'xag', 'xah', 'xai']:
            with open(f"../Datasets/{file}.dat", 'r') as f:
                content = f.read()
            with open("../Datasets/temp.data", 'a') as f:
                f.write(content)

        data = pd.read_csv("../Datasets/temp.data", header=None, sep=r'\s+')
        os.remove("../Datasets/temp.data")

        y = np.array([1 if elt in ['van'] else 0 for elt in data[18]])
        X = np.array(data.drop([18], axis=1))

    elif opt.dataset in ['wine']:
        data = pd.read_csv("../Datasets/wine.data", header=None)

        y = np.array([1 if elt == 1 else 0 for elt in data[0]])
        X = np.array(data.drop([0], axis=1))

    elif opt.dataset in ['wine4']:
        data = pd.read_csv("../Datasets/winequality-red.csv", sep=';')

        y = np.array([1 if elt in [4] else 0 for elt in data.quality])
        X = np.array(data.drop(["quality"], axis=1))

    elif opt.dataset in ['yeast3', 'yeast6']:
        data = pd.read_csv("../Datasets/yeast.data", header=None, sep=r'\s+')
        data = data.drop([0], axis=1)
        print(data)

        if opt.dataset == 'yeast3':
            y = np.array([1 if elt == 'ME3' else 0 for elt in data[8]])
        elif opt.dataset == 'yeast6':
            y = np.array([1 if elt == 'EXC' else 0 for elt in data[8]])
        X = np.array(data.drop([8], axis=1))

    dim = len(X[0])
    return (X, y, dim)
