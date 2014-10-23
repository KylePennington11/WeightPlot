from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

plot_params = {'margin': {'left': 0.13,
                          'right': 0.05,
                          'top': 0.07,
                          'bottom': 0.15},
                'keepAxis': ['left', 'bottom'],
                'markersize': 3,
                'dimensions': {'width': 345},
                'fontsize': 10.0,
                'ratio': 1.6813}


def format(subplot=111, style='thesis'):
    global plot_params, plt

    rc('font', **{'family': 'serif', 'serif': ['Palatino']})

    params = {'backend': 'ps',
              'axes.labelsize': plot_params['fontsize'],
              'text.fontsize': plot_params['fontsize'],
              'legend.fontsize': plot_params['fontsize'] - 2,
              'axes.linewidth': 0.5,
              'xtick.labelsize': plot_params['fontsize'],
              'ytick.labelsize': plot_params['fontsize'],
              'axes.formatter.limits': '-3, 3',
              'text.usetex': True,
              'text.latex.unicode': True,
              'lines.markersize': plot_params['markersize']}

    if style.lower() == 'ieee':
        plot_params['margin']['bottom'] += 0.02
        plot_params['margin']['left'] += 0.02
        plot_params['markersize'] /= 3.0
        plot_params['dimensions']['width'] = 255.0

    if subplot == 111:

        g_scale_left = plot_params['margin']['left']
        g_scale_right = 1.0 - plot_params['margin']['right']
        g_scale_top = 1.0 - plot_params['margin']['top']
        g_scale_bottom = plot_params['margin']['bottom']

        width_pt = plot_params['dimensions']['width']

        inch_per_pt = 1.0 / 72.27
        ratio = plot_params['ratio']
        width_total = width_pt * inch_per_pt
        width_graph = width_total * (g_scale_right - g_scale_left)
        height_graph = width_graph / ratio
        height_total = height_graph / (g_scale_top - g_scale_bottom)

        dimensions = [width_total, height_total]
        params['figure.figsize'] = dimensions


    plt.rcParams.update(params)

    fig = plt.figure()

    # turn off ticks where there is no spine
    fig.gca().xaxis.set_ticks_position('bottom')
    fig.gca().yaxis.set_ticks_position('left')

    for loc, spine in plt.gca().spines.items():
        if loc in plot_params['keepAxis']:
            spine.set_position(('outward', 0))  # outward by 10 points
        elif loc in ['right', 'top']:
            spine.set_color('none')  # don't draw spine

    if subplot == 111:
        fig.subplots_adjust(left=g_scale_left,
                            bottom=g_scale_bottom,
                            right=g_scale_right,
                            top=g_scale_top,
                            wspace=0.2,
                            hspace=0.2)

    return fig.add_subplot(subplot)


def get_heatPair(num=5):

    return [generate_colours(list(np.linspace(12, 44, num)), 90, 100),
            generate_colours(list(np.linspace(224, 192, num)), 90, 100)]


def get_blues(num):
    return generate_colours(list(np.linspace(225, 210, num)),
                            list(np.linspace(100, 100, num)),
                            list(np.linspace(55, 98, num)))


def get_yellows(num):
    return generate_colours(list(np.linspace(34, 44, num)),
                            list(np.linspace(90, 90, num)),
                            list(np.linspace(90, 100, num)))

def get_purples(num):
    return generate_colours(list(np.linspace(263, 283, num)),
                            list(np.linspace(90, 90, num)),
                            list(np.linspace(90, 100, num)))

def get_reds(num):
    return generate_colours(list(np.linspace(353, 360, num)),
                            list(np.linspace(90, 100, num)),
                            list(np.linspace(75, 100, num)))


def format_labels(axis, ys, debug=False):
    """
    Takes a matplotlib axes and formats the labels according to the values
    containind in it. This function returns the quantifier eg. 'm' or '\eta'
    to indicate any scaling that was applied to the labels
    """
    if debug:
        print("formatting axis " + str(axis))
    # print ys
    miny = abs(min(ys))
    maxy = abs(max(ys))
    if miny > maxy:
        tmp = miny
        miny = maxy
        maxy = tmp
        del(tmp)
    if debug:
        print ('miny = ' + str(miny))
        print ('maxy = ' + str(maxy))

    quantifiers = ['n', '\mu ', 'm', '', 'k', 'M', 'G']
    quantifiers.reverse()
    magnitudes = [1e9, 1e6, 1e3, 1, 1e-3, 1e-6, 1e-9]

    # Find the lowest quantifier
    for mag_low_index, mag_low_value in enumerate(magnitudes):
        if miny >= mag_low_value:
            break
    if debug:
        print ('lowest magnitude = ' + str(magnitudes[mag_low_index]))

    # Find the maximum quantifier
    for mag_high_index, mag_high_value in enumerate(magnitudes):
        if maxy >= mag_high_value:
            break

    # Detect if min was zero, if it was just use the max for both
    if miny == 0:
        mag_low_index = mag_high_index
        if debug:
            print('lowest magnitude adjusted to '
                  + str(magnitudes[mag_low_index])
                  + ' because of being equal to 0')

    if debug:
        print ('largest magnitude = ' + str(magnitudes[mag_high_index]))

    mag_mid_index = int(((mag_high_index - mag_low_index) / 2.0) + mag_low_index)

    multiplyer = 1 / magnitudes[mag_mid_index]
    if debug:
        print ('selected magnitude = ' + str(multiplyer))

    quantifier = quantifiers[mag_mid_index]
    if debug:
        print ('selected quantifier = ' + quantifier)

    decimalPlaces = 2
    if debug:
        print ('max number on y after scaling = ', (maxy * multiplyer))
    if maxy * multiplyer > 10:
        decimalPlaces = 1
    if maxy * multiplyer > 100:
        decimalPlaces = 0

    formattingString = '%0.' + str(decimalPlaces) + 'f'
    if debug:
        print ('format string = ' + formattingString)
    print (multiplyer)
    axis.set_major_formatter(FuncFormatter(lambda x, pos: formattingString % (x * multiplyer)))

    return quantifier


def get_standardColours(num, sets=1, palette='new'):
    """
    Returns the standard colour arrays for use with creating the graphs
    """
    offset = 360.0 / sets
    colours = []

    if palette == 'new':
        saturation = 90
        value = 100
        for i in range(sets):
            hues = list(np.linspace(34 + i * offset,
                                    44 + i * offset,
                                    num))
            saturation = list(np.linspace(90, 100, num))
            value = list(np.linspace(90, 100, num))
            colours.append(generate_colours(hues, saturation, value))

    elif palette == 'old':
        saturation = 100
        value = list(np.linspace(50, 100, num))
        hue = 34
        for i in range(sets):
            colours.append(generate_colours(hue + i * offset,
                                           saturation,
                                           value))

    return colours


def generate_colours(hue, saturation, value):
    """
    Hue is 360 degrees
    Saturation is percent
    Vale is percent

    Any of the fields can be arrays
    """
    import colorsys

    num = 1
    if type(hue) == list:
        if len(hue) > num:
            num = len(hue)

    if type(saturation) == list:
        if len(saturation) > num:
            num = len(saturation)

    if type(value) == list:
        if len(value) > num:
            num = len(value)

    tuples = []
    for i in xrange(num):
        if type(hue) == list:
            h = hue[i] / 360.0
        else:
            h = hue / 360.0

        if type(saturation) == list:
            s = saturation[i] / 100.0
        else:
            s = saturation / 100.0

        if type(value) == list:
            v = value[i] / 100.0
        else:
            v = value / 100.0

        tuples.append((h, s, v))

    return list(map(lambda x: colorsys.hsv_to_rgb(*x), tuples))
