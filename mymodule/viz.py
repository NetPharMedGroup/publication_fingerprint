import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def make_plot_data(c, z, l, b, h, index=None):
    '''
    css,zip,loewe,bliss,hsa,
    ind should be a multilevel index matching 3 levels of row index and 2 levels of col index
    '''
    assert index is not None 
    holder = dict()
    holder_melt = dict()
    for y in ['r', 'sd', 'range', 'no_norm', 'iqr']:
        holder[y] = pd.DataFrame(data=c[y].values(), 
                                 index=pd.MultiIndex.from_arrays(index), 
                                 columns=["css "+y,'css CI95'])
        holder[y]['zip '+y] = [x[0] for x in z[y].values()]
        holder[y]['zip CI95'] = [x[1] for x in z[y].values()]
        holder[y]['loewe '+y] = [x[0] for x in l[y].values()]
        holder[y]['loewe CI95'] = [x[1] for x in l[y].values()]
        holder[y]['bliss '+y] = [x[0] for x in b[y].values()]
        holder[y]['bliss CI95'] = [x[1] for x in b[y].values()]
        holder[y]['hsa '+y] = [x[0] for x in h[y].values()]
        holder[y]['hsa CI95'] = [x[1] for x in h[y].values()]

        midx = pd.MultiIndex(
        levels=[['CSS_RI', 'Synergy ZIP','Synergy Loewe', 'Synergy Bliss', 'Synergy HSA'], 
                [y,'']],
        codes=[
            [0,0,1,1,2,2,3,3,4,4],
            [0,1,0,1,0,1,0,1,0,1]
        ])
        holder[y].columns = midx

        plot1 = holder[y].T
        plot1 = plot1.droplevel(2, axis=1)
        plot1 = plot1.droplevel(1, axis=0)

        temp = ['CSS', 'CSS']
        temp.extend([x[8:] for x in plot1.index if 'Synergy' in x])
        plot1.index = temp

        new_ind =[]
        for ind, x in enumerate(plot1.index):
            if ind%2 == 0: #vals
                new_ind.append(x)
            else: #CIs
                x = 'ci_'+x
                new_ind.append(x)
        
        plot1.index = new_ind
        plot1_corr = plot1.iloc[range(0,10,2),:]\
        .melt(ignore_index=False)\
        .reset_index(drop=False) 
        plot1_corr['ci'] = plot1.iloc[range(1,11,2),:]\
        .melt(ignore_index=False)\
        .reset_index(drop=False)['value'] #??don't need CSS CI
        
        if y == 'r':
            to_name = 'mean_corr'
        else:
            to_name = "rmse_"+y
        plot1_corr.columns = ['metric', 'model', 'size', to_name, 'ci']
        plot1_corr['model'] = plot1_corr['model']+ "_"+ plot1_corr['size'].astype('str')
        plot1_corr.drop(columns='size', inplace=True)
        plot1_corr = plot1_corr.sort_values(by=to_name, ascending=True)
        holder_melt[y] = plot1_corr

    return holder, holder_melt



def space_invaders(name_plot='r', ci='t', include_css=False, holder_melt=None):
    '''['r', 'sd', 'iqr', 'range', 'no_norm'], t vs bs '''
    assert holder_melt is not None
    df = holder_melt.copy()
    sns.set_theme(context='poster', 
                  style='darkgrid',
                  font='sans-serif',
                  color_codes=True)
    plt.rcParams["figure.figsize"] = (16,12)

    
    name_plot = name_plot
    name_y = 'rmse_'+ name_plot
    include_css = include_css

    if name_plot == 'r':
        name_y = 'mean_corr'

    if include_css:
        errors = df[name_plot].ci
        data = df[name_plot]
    else:
        errors = df[name_plot].loc[df[name_plot]['metric'] !='CSS','ci']
        data = df[name_plot].loc[df[name_plot]['metric'] !='CSS', :]

    #plt.rcParams["errorbar.capsize"] = 0.05
    #colors = ['#a6cee3','#1f78b4','#b2df8a','#ffff99','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#33a02c']
    ax = sns.pointplot(x='metric', 
                       y=name_y, 
                       hue='model', 
                       style='metric',
                       data=data, 
                       dodge=0.6, 
                       join=False, 
                       ci=None,
                       scale =1,
                       palette=sns.color_palette('Paired', data.shape[0])
                       #palette = sns.color_palette("Paired", 13)
                      )
    

    # Find the x,y coordinates for each point
    x_coords = []
    y_coords = []
    for point_pair in ax.collections:
        for x, y in point_pair.get_offsets():
            x_coords.append(x)
            y_coords.append(y)

    # Calculate the type of error to plot as the error bars
    # Make sure the order is the same as the points were looped over

    ax.errorbar(x_coords, 
                y_coords, 
                yerr=errors, 
                fmt='none',
                c='black', 
                elinewidth=4,
                markeredgewidth=4,
                zorder=-1, 
                capsize=10)
    
    ax.set_xlabel('')
    ax.set_ylabel('')

    plt.title('rmse normed by '+name_plot +' confidence calculated using '+ci)

    if name_plot == 'r':
        plt.title('r '+'confidence calculated using '+ci)
    elif name_plot=='no_norm':
        plt.title('rmse not normed ' + 'confidence calculated using ' +ci)

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.1)

    return plt



def highlight_max(s):
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]

def highlight_min(s):
    is_min = s == s.min()
    return ['background-color: yellow' if v else '' for v in is_min]

def color_second(s):
    f = lambda x: x.nlargest(3)
    return ['color: red' if v in f(s).values else 'color: black' for v in s]

def color_second_min(s):
    f = lambda x: x.nsmallest(3)
    return ['color: red' if v in f(s).values else 'color: black' for v in s]

def color_lowest(s):
    f = lambda x: x.nsmallest(1)
    return ['color: green' if v in f(s).values else 'color: black' for v in s]

def color_highest_error(s):
    is_max = s == s.max()
    return ['color: blue' if v else 'color: black' for v in is_max]

def color_mean(s):
    return ['background-color: lawngreen' for v in s]

def colors(name='r', holder=None):
    '''
    name any of r - sd - iqr - no_norm - range
    '''
    assert holder is not None

    df = holder[name].copy()

    if name != 'r':
        f1 = highlight_min
        f2 = color_second_min
        f3 = color_lowest
    elif name == 'r':
        f1 = highlight_max
        f2 = color_second
        f3 = color_highest_error

    rows = [x for x in df.index]
    rows.append(('mean','',''))
    df.loc[('mean','','')] = round(df.mean(),4)
    subset1=[x for x in df.columns if name in x]
    subset2=[x for x in df.columns if name not in x]

    return df.style.apply(f1, subset=subset1).apply(f2,subset=subset1).apply(f3, subset=subset2).apply(color_mean,axis=1,subset=rows[-1] ).set_precision(4).format({k: lambda v: f"±{v}" for k in subset2})