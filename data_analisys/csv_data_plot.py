#read and plot csv file
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv
import os

class PlotData:
    def __init__(self, img_num = 0, img_size = 0.0, g = [], times = [] ,alg_types = [], errors = []):
        self.img_num = img_num
        self.img_size = img_size
        self.g = g
        self.times = times
        self.alg_types = alg_types
        self.errors = errors

def plot(data_plots, file_name):
    #check if there is anything to plot before further operation
    plot_id = 0
    start = 1
    #create img_x array  - x axis for image size plot
    step = 0.1
    img_x = np.linspace(0.0, 2 * len(data_plots), num = int((2 * len(data_plots)) / step) + 1 ).tolist()
    best_plots = []
    avg_plots = []
    fig_list  = set()
    color = ['r', 'g', 'b', 'k']
    for data in data_plots:
        g_x = range(0, len(data.g))
        step = 2
        g_xn = []
        idx = 0
        while idx < step * len(data.g):
            g_xn += [idx]
            idx += step
         #make it generated according to number of algorithms for passed data object list
        best_plot = []
        avg_plot = []
        for i in xrange(len(data.times[0]) - 1, -1, -1):
            t_y = [x[i] for x in data.times]
            best_plot += [(max(t_y), np.argmax(np.array(t_y)))]
            avg_plot += [sum([int(val) for val in t_y])/ len(t_y)]
            fig_list.add(plt.figure(plot_id))
            plt.figure(plot_id)
            plt.subplot(len(data.times[0]) * 100 + 10 + i + 1)
            plt.bar(g_x, t_y, color = color[i])
            lgnd = mpatches.Patch(color = color[i], label = data.alg_types[i])
            plt.legend(handles = [lgnd])
            plt.ylabel('Time[ms]')
            if i == 0:
                plt.title('Execution time for image size ' + str(int(float(data.img_size) * 1024.0)) + ' KB')
            elif i == len(data.times[0]) - 1:
                plt.xlabel('Neighbor range')
            plt.xticks(g_x, data.g)
            plt.grid()

            fig_list.add(plt.figure(plot_id + 100))
            plt.figure(plot_id + 100)
            x_i = [n + (0.4 * i) for n in g_xn]
            plt.bar(x_i, t_y, color = color[i], label = data.alg_types[i], width = 0.4)
            plt.xlabel('Neighbor range')
            plt.ylabel('Time[ms]')
            plt.title('Execution time for image size ' + str(int(float(data.img_size)* 1024.0)) + ' KB')
            plt.xticks(g_xn, data.g)
            plt.legend()
            plt.grid()
        plt.figure(plot_id, figsize = (10, 10))
        plt.subplots_adjust(hspace = 0.52)
        best_plots += [best_plot]
        avg_plots += [avg_plot]
        plot_id += 1
        start += 1
    # avg and best time plots
    sizes = [(int(float(data.img_size)* 1024.0)) for data in data_plots]
    step = 1
    bar_width = 60
    start = min(sizes) - int(len(data_plots[0].g) * bar_width / 2)
    end = max(sizes) + (len(data_plots[0].g) * bar_width / 2)
    size_x = np.linspace(start, end, num = ((end - start) / step)).tolist()
    init_arr = [0] * len(size_x)
    fig_list.add(plt.figure(1000))
    plt.figure(1000)
    for i in xrange(0, len(avg_plots[0])):
        y = init_arr[:]
        avg = [x[i] for x in avg_plots]
        dist = 0
        for n in xrange(0, len(sizes)):
            try:
                idx = int(((sizes[n] - start) / step) - (i * step * bar_width)) + 100#100 - magic number - moves in x all bars, make it so the middle bar is in sizes[n]
                y[idx] = avg[n]
            except:
                print 'error'
        plt.bar(size_x, y, color=color[len(data_plots[0].alg_types) - i - 1], label=data_plots[0].alg_types[len(data_plots[0].alg_types) - i - 1], width=bar_width)

    plt.xlabel('Image size[KB]')
    plt.ylabel('Time[ms]')
    plt.title('Average Execution Time')
    plt.xticks(sizes, [str(int(float(d.img_size) * 1024.0)) for d in data_plots])
    plt.grid()
    plt.legend()

    fig_list.add(plt.figure(1001))
    plt.figure(1001)
    for i in xrange(0, len(best_plots[0])):
        y = init_arr[:]
        best = [x[i] for x in best_plots]
        best_g = ''
        for n in xrange(0, len(sizes)):
            idx = int(((sizes[n] - start) / step) - (i * step * bar_width)) + 100
            y[idx] = best[n][0]
            best_g = data.g[best[n][1]]
        plt.bar(size_x, y, color=color[len(data_plots[0].alg_types) - i - 1], label=data_plots[0].alg_types[len(data_plots[0].alg_types) - i - 1] + ' for g ' + best_g, width=bar_width)
    plt.xlabel('Image size[KB]')
    plt.ylabel('Time[ms]')
    plt.title('Best Case Execution Time')
    plt.xticks(sizes, [str(int(float(d.img_size) * 1024.0)) for d in data_plots])
    plt.grid()
    plt.legend()

    #error plot - for future - first generate this parameter
    #plt.show()
    #save all plots
    i = 0
    dot = file_name.index('.')

    dir = file_name[:dot] + "_plots/"
    if not os.path.exists(dir):
        os.makedirs(dir)
    for fig in fig_list:
        fig.savefig( dir + "plot" + str(i), dpi = fig.dpi)
        i += 1


#main script
file_name = 'test_results.csv'#raw_input("State results file name : ")
plot_data = open(file_name, 'r')
c_reader = csv.reader(plot_data, delimiter=',', quotechar='|')
plots = []
i = 0
prev_img = -1
pData = PlotData()
added_plot = False
for row in c_reader:
    if i > 0:
        current_img = row[0]
        if current_img != prev_img:
            if added_plot:
                plots += [pData]
                pData = PlotData()
            else:
                added_plot = True
            pData.img_num = current_img
            pData.img_size = row[1]
            pData.g = []
            pData.times = []
            pData.errors = []
            prev_img = current_img
        pData.g += [(row[2] + ':' +row[3])]
        pData.errors += [row[4]]
        time = []
        for i in xrange(5, len(row)):
            time += [row[i]]
        pData.times += [time]
    else:
        for i in xrange(5, len(row)):
            pData.alg_types += [row[i]]
    i += 1
plots += [pData]
print plots
plot(plots, file_name)
