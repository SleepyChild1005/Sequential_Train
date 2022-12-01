import pandas as pd
import matplotlib.pyplot as plt


def plot_csvfile(df):

    plt.rcParams["figure.figsize"] = [7.50, 3.50]
    plt.rcParams["figure.autolayout"] = True

    df.plot()
    plt.show()


def plot_exp(exp_name):
    # df = pd.read_csv(f'varnet_results/{exp_name}/CSV_log/TrainingLoss.csv', header=None)
    # df2 = pd.read_csv(f'varnet_results/{exp_name}/CSV_log/ValLoss.csv', header=None)
    df = pd.read_csv('/Users/kangdong/Activity/OMSCS/Project Idea/ValLoss.csv', header=None)


    plot_csvfile(df)
    # plot_csvfile(df2)


# headers = ['Training Loss']
exp_name = 'varnet_overfit_E100_F1_lr2_221129_1614'
plot_exp(exp_name=exp_name)