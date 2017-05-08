#!/home/boulgakov/anaconda2/bin/python


"""
Useful plotting functions. Core code & design by Angela Bardo.
"""


import plotly.plotly
import plotly.offline
from plotly import graph_objs
import numpy as np

colors = {405:'GnBu', 488:'YIOrRd', 561:'YIOrRd', 647:'YIGnBu'}

def plot_histogram(plot_target, title, yaxis_title, xaxis_title,
                   log_yaxis, filepath):
    plot_range = (np.amin(plot_target) - 1, np.amax(plot_target) + 1)
    layout = graph_objs.Layout(title=title,
                               yaxis=dict(type=('log' if log_yaxis else ''),
                                          title=yaxis_title),
                               xaxis=dict(range=plot_range,
                                          title=xaxis_title))
    data = [graph_objs.Histogram(x=plot_target)]
    fig = graph_objs.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename=filepath, auto_open=False)


def single_drops_heatmap(signals, num_mocks, num_edmans, num_mocks_omitted,
                         peptide_string, wavelength, zmin, zmax, filepath,
                         plot_multidrops=False, plot_remainders=False):
    num_mocks -= num_mocks_omitted
    total_cycles = num_mocks + num_edmans
    if plot_remainders:
        heatmap_array_size = total_cycles + 1
    else:
        heatmap_array_size = total_cycles
    heatmap_array = np.array([[0 for y in range(heatmap_array_size)]])
    for (signal, is_zero), count in signals.iteritems():
        if len(signal) != 1:
            continue
        if signal == (('A', 0),):
            if not plot_remainders:
                continue
            if is_zero:
                continue
            else:
                x, y = 0, heatmap_array_size - 1
        else:
            if not is_zero:
                continue
            else:
                x, y = 0, signal[0][1] - 1
        heatmap_array[x, y] += count
    color_channel = wavelength
    if color_channel not in colors:
        raise("Exception: Invalid wavelength.")
    cs = colors[color_channel] #cs = "color space"
    cycles_header = (["M" + str(i + 1 + num_mocks_omitted)
                      for i in range(num_mocks)] +
                     ["E" + str(i + 1) for i in range(num_edmans)] +
                     ["R"])
    annotations = []
    text_limit = np.amax(heatmap_array)
    for (y, x), count in np.ndenumerate(heatmap_array):
        annotations.append(dict(text=str(count),
                                x=cycles_header[x],
                                y="C",
                                font=dict(color=('white'
                                                 if count > (text_limit * 0.75)
                                                 else 'black')),
                                showarrow=False))
    layout = graph_objs.Layout(title=("Single Drops (" + str(color_channel) +
                                      " Channel) Total: " +
                                      str(np.sum(heatmap_array)) + " - " +
                                      peptide_string),
                               annotations=annotations,
                               titlefont=dict(size=16),
                               yaxis=dict(title="",
                                          titlefont=dict(size=14),
                                          ticks="",
                                          autorange='reversed'),
                               xaxis=dict(title="Drop Position",
                                          titlefont=dict(size=16),
                                          ticks="",
                                          side='top'),
                               margin=graph_objs.Margin(l=50, r=50, b=100,
                                                        t=150, pad=2),
                               width=700,
                               height=325,
                               autosize=False)
    data = [graph_objs.Heatmap(z=heatmap_array,
                               x=cycles_header,
                               y=["C", ""],
                               colorscale=cs,
                               reversescale=True,
                               zmin=(np.amin(heatmap_array)
                                     if zmin is None else zmin),
                               zmax=(np.amax(heatmap_array)
                                     if zmax is None else zmax))]
    fig = graph_objs.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename=filepath, auto_open=False)


def double_drops_heatmap(signals, num_mocks, num_edmans, num_mocks_omitted,
                         peptide_string, wavelength, zmin, zmax, filepath,
                         plot_multidrops=False, plot_remainders=False):
    num_mocks -= num_mocks_omitted
    total_cycles = num_mocks + num_edmans
    if plot_remainders:
        heatmap_array_size_x = total_cycles
        heatmap_array_size_y = total_cycles + 1
    else:
        heatmap_array_size_y = heatmap_array_size_x = total_cycles
    heatmap_array = np.array([[0 for y in range(heatmap_array_size_y)]
                              for x in range(heatmap_array_size_x)])
    for (signal, is_zero), count in signals.iteritems():
        if len(signal) == 1:
            if signal == (('A', 0),):
                continue
            elif plot_remainders and not is_zero:
                x, y = signal[0][1] - 1, heatmap_array_size_y - 1
            else:
                continue
        elif len(signal) == 2:
            if not plot_multidrops and len(signal) > len(set(signal)):
                continue
            elif is_zero:
                x, y = signal[0][1] - 1, signal[1][1] - 1
            else:
                continue
        elif len(signal) > 2:
            continue
        heatmap_array[x, y] += count
    color_channel = wavelength
    if color_channel not in colors:
        raise("Exception: Invalid wavelength.")
    cs = colors[color_channel] #cs = "color space"
    y_cycles_header = (["M" + str(i + 1 + num_mocks_omitted)
                        for i in range(num_mocks)] +
                       ["E" + str(i + 1) for i in range(num_edmans)])
    if plot_remainders:
        x_cycles_header = y_cycles_header + ["R"]
    else:
        x_cycles_header = y_cycles_header
    annotations = []
    text_limit = np.amax(heatmap_array)
    for (y, x), count in np.ndenumerate(heatmap_array):
        annotations.append(dict(text=str(count),
                                x=x_cycles_header[x],
                                y=y_cycles_header[y],
                                font=dict(color=('white'
                                                 if count > (text_limit * 0.75)
                                                 else 'black')),
                                showarrow=False))
    layout = graph_objs.Layout(title=("Double Drops (" + str(color_channel) +
                                      " Channel) Total: " +
                                      str(np.sum(heatmap_array)) + " - " +
                                      peptide_string),
                               annotations=annotations,
                               titlefont=dict(size=16),
                               yaxis=dict(title="First Drop",
                                          titlefont=dict(size=16),
                                          ticks="",
                                          autorange='reversed'),
                               xaxis=dict(title="Second Drop",
                                          titlefont=dict(size=16),
                                          ticks="",
                                          side='top'),
                               margin=graph_objs.Margin(l=50, r=50, b=100,
                                                        t=150, pad=4),
                               width=700,
                               height=735,
                               autosize=False)
    data = [graph_objs.Heatmap(z=heatmap_array,
                               x=x_cycles_header,
                               y=y_cycles_header,
                               colorscale=cs,
                               reversescale=True,
                               zmin=(np.amin(heatmap_array)
                                     if zmin is None else zmin),
                               zmax=(np.amax(heatmap_array)
                                     if zmax is None else zmax))]
    fig = graph_objs.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename=filepath, auto_open=False)


def single_drops_heatmap_v2(signals, num_mocks, num_edmans, num_mocks_omitted,
                            peptide_string, wavelength, zmin, zmax, filepath,
                            plot_remainders=False):
    num_mocks -= num_mocks_omitted
    total_cycles = num_mocks + num_edmans
    if plot_remainders:
        heatmap_array_size = total_cycles + 1
    else:
        heatmap_array_size = total_cycles
    heatmap_array = np.array([[0 for y in range(heatmap_array_size)]])
    for (signal, is_zero, starting_intensity), count in signals.iteritems():
        if starting_intensity > 1:
            continue
        if len(signal) != 1:
            continue
        if signal == (('A', 0),):
            if not plot_remainders:
                continue
            if is_zero:
                continue
            else:
                x, y = 0, heatmap_array_size - 1
        else:
            if not is_zero:
                continue
            else:
                x, y = 0, signal[0][1] - 1
        heatmap_array[x, y] += count
    color_channel = wavelength
    if color_channel not in colors:
        raise("Exception: Invalid wavelength.")
    cs = colors[color_channel] #cs = "color space"
    cycles_header = (["M" + str(i + 1 + num_mocks_omitted)
                      for i in range(num_mocks)] +
                     ["E" + str(i + 1) for i in range(num_edmans)] +
                     ["R"])
    annotations = []
    text_limit = np.amax(heatmap_array)
    for (y, x), count in np.ndenumerate(heatmap_array):
        annotations.append(dict(text=str(count),
                                x=cycles_header[x],
                                y="C",
                                font=dict(color=('white'
                                                 if count > (text_limit * 0.75)
                                                 else 'black')),
                                showarrow=False))
    layout = graph_objs.Layout(title=("Single Drops (" + str(color_channel) +
                                      " Channel) Total: " +
                                      str(np.sum(heatmap_array)) + " - " +
                                      peptide_string),
                               annotations=annotations,
                               titlefont=dict(size=16),
                               yaxis=dict(title="",
                                          titlefont=dict(size=14),
                                          ticks="",
                                          autorange='reversed'),
                               xaxis=dict(title="Drop Position",
                                          titlefont=dict(size=16),
                                          ticks="",
                                          side='top'),
                               margin=graph_objs.Margin(l=50, r=50, b=100,
                                                        t=150, pad=2),
                               width=700,
                               height=325,
                               autosize=False)
    data = [graph_objs.Heatmap(z=heatmap_array,
                               x=cycles_header,
                               y=["C", ""],
                               colorscale=cs,
                               reversescale=True,
                               zmin=(np.amin(heatmap_array)
                                     if zmin is None else zmin),
                               zmax=(np.amax(heatmap_array)
                                     if zmax is None else zmax))]
    fig = graph_objs.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename=filepath, auto_open=False)


def double_drops_heatmap_v2(signals, num_mocks, num_edmans, num_mocks_omitted,
                            peptide_string, wavelength, zmin, zmax, filepath,
                            plot_multidrops=False, plot_remainders=False):
    num_mocks -= num_mocks_omitted
    total_cycles = num_mocks + num_edmans
    if plot_remainders:
        heatmap_array_size_x = total_cycles
        heatmap_array_size_y = total_cycles + 1
    else:
        heatmap_array_size_y = heatmap_array_size_x = total_cycles
    heatmap_array = np.array([[0 for y in range(heatmap_array_size_y)]
                              for x in range(heatmap_array_size_x)])
    for (signal, is_zero, starting_intensity), count in signals.iteritems():
        if starting_intensity > 2:
            continue
        if len(signal) == 1:
            if signal == (('A', 0),):
                continue
            elif plot_remainders and not is_zero:
                x, y = signal[0][1] - 1, heatmap_array_size_y - 1
            else:
                continue
        elif len(signal) == 2:
            if not plot_multidrops and len(signal) > len(set(signal)):
                continue
            elif is_zero:
                x, y = signal[0][1] - 1, signal[1][1] - 1
            else:
                continue
        elif len(signal) > 2:
            continue
        heatmap_array[x, y] += count
    color_channel = wavelength
    if color_channel not in colors:
        raise("Exception: Invalid wavelength.")
    cs = colors[color_channel] #cs = "color space"
    y_cycles_header = (["M" + str(i + 1 + num_mocks_omitted)
                        for i in range(num_mocks)] +
                       ["E" + str(i + 1) for i in range(num_edmans)])
    if plot_remainders:
        x_cycles_header = y_cycles_header + ["R"]
    else:
        x_cycles_header = y_cycles_header
    annotations = []
    text_limit = np.amax(heatmap_array)
    for (y, x), count in np.ndenumerate(heatmap_array):
        annotations.append(dict(text=str(count),
                                x=x_cycles_header[x],
                                y=y_cycles_header[y],
                                font=dict(color=('white'
                                                 if count > (text_limit * 0.75)
                                                 else 'black')),
                                showarrow=False))
    layout = graph_objs.Layout(title=("Double Drops (" + str(color_channel) +
                                      " Channel) Total: " +
                                      str(np.sum(heatmap_array)) + " - " +
                                      peptide_string),
                               annotations=annotations,
                               titlefont=dict(size=16),
                               yaxis=dict(title="First Drop",
                                          titlefont=dict(size=16),
                                          ticks="",
                                          autorange='reversed'),
                               xaxis=dict(title="Second Drop",
                                          titlefont=dict(size=16),
                                          ticks="",
                                          side='top'),
                               margin=graph_objs.Margin(l=50, r=50, b=100,
                                                        t=150, pad=4),
                               width=700,
                               height=735,
                               autosize=False)
    data = [graph_objs.Heatmap(z=heatmap_array,
                               x=x_cycles_header,
                               y=y_cycles_header,
                               colorscale=cs,
                               reversescale=True,
                               zmin=(np.amin(heatmap_array)
                                     if zmin is None else zmin),
                               zmax=(np.amax(heatmap_array)
                                     if zmax is None else zmax))]
    fig = graph_objs.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename=filepath, auto_open=False)
