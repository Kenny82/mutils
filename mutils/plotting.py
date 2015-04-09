# -*- coding:utf-8 -*-
"""

.. module:: mutils.plotting

.. moduleauthor:: Moritz Maus <mmaus@sport.tu-darmstadt.de>

plotting.py - preformatted scientific plotting
==============================================

This module provides an adapted figure-subclass
:py:class:`mfig` for plotting data in a pre-formatted way.

typical usage::

    fig = myfigure('a5', 'my figure title')
    fig.set_layout(plotting.layouts['2x2-l']) # this is mandatory!
    fig.add_to_legend({'color' : 'r'}, 'item 1')
    fig.add_to_legend({'color' : 'k'}, 'item 2')

LE: 18.11.2012 Moritz Maus

LE: 14.01.2013 Moritz Maus - modified docstrings for sphinx

LE: 23.01.2013 Moritz Maus - modified docstrings for sphinx

"""

__version__ = '0.1.2'

from pylab import figure, mod, plot, text, arange, fill_between
from matplotlib.figure import Figure
from numpy import array, sqrt, linspace, sin, cos, vstack, pi
from numpy.linalg import eig

# color definitions
colorset_distinct = ['#000000', '#106aa4', '#43bf3c', '#ff7f00', '#ef0a28', '#5f2e82',
'#8f8f8f', '#92bee3', '#a1df80', '#fdaf5f', '#fb8987', '#baa2c5',]

# dictionary of linestyles - handy shortcuts
# usage: e.g.:
# ls = lineStyles['bw']
# plot(x, y, **ls[0])
# plot(x, y, **ls[1])

# each entry of lineStyles is a list
# each list entry is either:
# (a) a dictionary, containing line style information
# (b) a list of 2 dictionaries, containing related line style information each
# Note: in the case (b), this is usefull when e.g. plotting a line with
# confidence intervals
lineStyles = { 'bw' : 
    [
     {'linestyle' : '-', 'color' : '#000000', 'linewidth' : 1},
     {'linestyle' : '-', 'color' : '#626262', 'linewidth' : 1},
     {'linestyle' : '-', 'color' : '#aeaeae', 'linewidth' : 1},
     {'linestyle' : '-.', 'color' : '#000000', 'linewidth' : 1},
     {'linestyle' : '-.', 'color' : '#626262', 'linewidth' : 1},
     {'linestyle' : '-.', 'color' : '#aeaeae', 'linewidth' : 1},
    ],
    'bw+-' : 
    [
     [{'linestyle' : '-', 'color' : '#000000', 'linewidth' : 2},
      {'linestyle' : '--', 'color' : '#000000', 'linewidth' : 1}],
     [{'linestyle' : '-', 'color' : '#626262', 'linewidth' : 2},
      {'linestyle' : '--', 'color' : '#626262', 'linewidth' : 1}],
     [{'linestyle' : '-', 'color' : '#aeaeae', 'linewidth' : 2},
      {'linestyle' : '--', 'color' : '#aeaeae', 'linewidth' : 1}],
     [{'linestyle' : '-.', 'color' : '#000000', 'linewidth' : 2},
      {'linestyle' : ':', 'color' : '#000000', 'linewidth' : 1}],
     [{'linestyle' : '-.', 'color' : '#626262', 'linewidth' : 2},
       {'linestyle' : ':', 'color' : '#626262', 'linewidth' : 1}],
     [{'linestyle' : '-.', 'color' : '#aeaeae', 'linewidth' : 2},
       {'linestyle' : ':', 'color' : '#aeaeae', 'linewidth' : 1}],
    ]
    }


# dictionary of paper sizes
paperFormats = {
    'a0' : (1189, 841), 
    'b0' : (1414, 1000),
    'a1' : (841, 594), 
    'b1' : (1000, 707),
    'a2' : (594, 420), 
    'b2' : (707, 500),
    'a3' : (420, 297), 
    'b3' : (500, 353),
    'a4' : (297, 210), 
    'b4' : (353, 250),
    'a5' : (210, 148), 
    'b5' : (250, 176),
    'a6' : (148, 105), 
    'b6' : (176, 125),
    'a7' : (105, 74), 
    'b7' : (125, 88),
    'a8' : (74, 52), 
    'b8' : (88, 62),
    'a9' : (52, 37), 
    'b9' : (62, 44),
    'a10' : (37, 26), 
    'b10' : (44, 31),
    }

"""
A layout definition goes as follows:
    (...)

    subplots:
        First, a grid is defined. The boxes of the grid are numbered as you
        would 'read' them: 0 is top-left, 1 is top-(left + 1), ..., and the
        numbers continue each line.
        Then, a tuple of integers or 2-tuples defines the span of each subplot.
        Ideally, but not necessarily, these subplots cover each grid box once.

"""

layouts = { 
    '2x2-l' : { 
        'name' : '2x2-l',
        'description' : '2 by 2 subplots, with an additional top frame for ' +
        'the legend',
        'legend_frame' : True,
        'legend_location' : 'top', # currently: ignored,
        'legend_rows' : 3, # how many rows in the legend frame (top or bottom)
        'legend_cols' : None, # how many columns in the legend frame
        'legend_fontsize' : 10, # fontsize in pt (1/72 inch)
        'fontsize' : 10,
        'margins' : (15, 5, 10, 5), # bottom, top, left, right [mm]
        'spacing' : (2, 5, 8), # width, height between subplots, 
        # distance to legend [mm]
        'subplot_grid_horiz' : 2,
        'subplot_grid_vert' : 2,
        'subplots' : (0, 1, 2, 3),
        'ticklabels_outer_only' : True,
        },
    '1x1-l' : {
        'name' : '1x1-l',
        'description' : 'just a single frame with legend frame on top',
        'legend_frame' : True,
        'legend_location' : 'top',
        'legend_rows' : 2,
        'legend_cols' : 2,
        'legend_margins' : (2, 2, 10, 10),
        'legend_fontsize' : 9,
        'legend_linelength' : 8.,
        'fontsize' : 9,
        'margins' : (15, 5, 15, 5),
        'spacing' : (2, 7, 2),
        'subplot_grid_horiz' : 1,
        'subplot_grid_vert' : 1,
        'subplots' : (0, ),
        },
    '4x3-l' : {
        'name' : '4x3-l',
        'description' : '4 by 3 subplots incl. legend',
        'legend_frame' : True,
        'legend_location' : 'top', # currently: ignored,
        'legend_rows' : 3, # how many rows in the legend frame (top or bottom)
        'legend_cols' : 3, # how many columns in the legend frame
        'legend_margins' : (2, 2, 10, 10), # bottom, top, left, right [mm]
        'legend_fontsize' : 10, # fontsize in pt (1/72 inch)
        'legend_linelength' : 8, # [mm] length of a line in the legend
        'legend_line_text_sep' : 1.2, # [mm] separation between line and text
        'legend_textvspace' : 1.2, # fraction of vert. text spacing to text
        # height
        'fontsize' : 10,
        'margins' : (15, 5, 15, 2), # bottom, top, left, right [mm]
        'spacing' : (2, 7, 9), # width, height between subplots, 
        # distance to legend [mm]
        'subplot_grid_horiz' : 3,
        'subplot_grid_vert' : 4,
        'subplots' : (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
        'ticklabels_outer_only' : True,
        'cxlabel_margin' : 3., # mm distance from common xlabel to bottom
        'cylabel_margin' : 2., # mm distance from common ylabel to left
        },
    'plain' : { 
        'name' : 'plain',
        'description' : 'plain image - good for e.g. animations',
        'legend_frame' : False,
        'legend_location' : 'top', # currently: ignored,
        'legend_rows' : 3, # how many rows in the legend frame (top or bottom)
        'legend_cols' : None, # how many columns in the legend frame
        'legend_fontsize' : 10, # fontsize in pt (1/72 inch)
        'fontsize' : 10,
        'margins' : (0, 0, 0, 0), # bottom, top, left, right [mm]
        'spacing' : (0, 0, 0), # width, height between subplots, 
        # distance to legend [mm]
        'subplot_grid_horiz' : 1,
        'subplot_grid_vert' : 1,
        'subplots' : (0,),
        },
    '3x3-test' : { 
        'name' : '3x3-test',
        'description' : '3 by 3 test' +
        'the legend',
        'legend_frame' : True,
        'legend_location' : 'top', # currently: ignored,
        'legend_rows' : 3, # how many rows in the legend frame (top or bottom)
        'legend_cols' : None, # how many columns in the legend frame
        'legend_fontsize' : 10, # fontsize in pt (1/72 inch)
        'fontsize' : 10,
        'margins' : (15, 5, 10, 5), # bottom, top, left, right [mm]
        'spacing' : (2, 5, 8), # width, height between subplots, 
        # distance to legend [mm]
        'subplot_grid_horiz' : 3,
        'subplot_grid_vert' : 3,
        'subplots' : ((0, 4), 1, 2, (4, 5), 6),
        },
    '2+1-l' : { 
        'name' : '2+1-l',
        'description' : '2 subplots left, one subplot right, with an ' + 
        'additional top frame for the legend',
        'legend_frame' : True,
        'legend_location' : 'top', # currently: ignored,
        'subplot_grid_horiz' : 2,
        'subplot_grid_vert' : 2,
        'subplots' : (0, 2, (1, 3)),
        }
 }


def plotband(x, y, ups, downs, color='#0067ea', alpha=.25, **kwargs):
    """
    plots a line surrounded by a ribbon of the same color, but semi-transparent
    
    :args:
        x (N-by-1): positions on horizontal axis
        y (N-by-1): corresponding vertical values of the (center) line
        ups (N-by-1): upper edge of the ribbon
        downs (N-by-1): lower edge of the ribbon
        
    :returns:
        [line, patch] returns of underlying "plot" and "fill_between" function
    """
    pt1 = plot(x, y, color, **kwargs )
    pt2 = fill_between(x, ups, downs, color='None', facecolor=color, lw=0, alpha=alpha)
    return [pt1, pt2]

def plot_cov_ellispis(ref, cv, ax_sel, color='k'):
    """
    Plots the covariance ellipse on the current axes.    

    @param ref (N-by-1 array): reference point (origin)
    @param cv (N-by-N array): the covariance matrix
    @ax_sel (2-by-1 array): the axes of the data to be plotted, e.g. [0, 1]
    @color (color code): color in which the ellipsis should be drawn

    @return Returns None
    """
    ax_sel_ = array(ax_sel)
    ev, ew = eig(cv[ax_sel_, :][:, ax_sel_])
    ref_ = array(ref)[ax_sel]
    a = sqrt(ev[0])
    b = sqrt(ev[1])
    vec1 = a * ew[:,0]
    vec2 = b * ew[:,1]
    # edges
    t = linspace(0, 2*pi, 9)
    all_p = ref_ + vstack([(cos(tx) * vec1 + sin(tx)*vec2) for tx in t])
    #all_p = ref + ()
    
    p1 = ref_ + vec1
    p2 = ref_ - vec1
    p3 = ref_ + vec2
    p4 = ref_ - vec2
    # 45-deg points
    p1a = ref_ + (0.707 * vec1 + 0.707 * vec2)
    p2a = ref_ + (0.707 * vec1 - 0.707 * vec2)
    p3a = ref_ + (-0.707 * vec1 + 0.707 * vec2)
    p4a = ref_ + (-0.707 * vec1 - 0.707 * vec2)
    
    # plot axes
    #plot([p1[0], p2[0]], [p1[1], p2[1]], 'rd-', lw=4)
    #plot([p3[0], p4[0]], [p3[1], p4[1]], 'md-', lw=4)
    plot(all_p[[0,4], 0], all_p[[0,4], 1], '-', color=color, lw=0.5)
    plot(all_p[[2,6], 0], all_p[[2,6], 1], '-', color=color, lw=0.5)
    
    # plot curve
    plot(all_p[:,0], all_p[:,1], '-', color=color, lw=1)
    

class mfig(object):
    """
    This class provides convenient formatting of plots.

    TODO: write docstring!
    """

    def __init__(self, figformat, title=None, clear=True, **kwargs):
        """ docstring """
        self.fig = figure(num=title, **kwargs)
        if clear:
            self.fig.clf()
        self.set_format(figformat)
        self.axes = []
        self.legend_entries = []


    def set_format(self, width, height=None):
        """
        sets the format of the figure
        
        Parameters
        ==========
        width : *float* **or** *string*
            sets the width of the paper in mm. if *width* is a string, a
            predefined format will be set.
            An optional '.T' may be placed at the end of the string to
            transpose the paper (default: landscape)
        height : *float* **or** *None*
            the height of the figure in mm. Ignored if 'width' is a string

        Returns
        =======
            (None)

        """
        if isinstance(width, str):
            if width.lower().endswith('.t'):
                h, w = paperFormats.get(width[:-2].lower(), (7 * 25.4, 7 * 25.4))
            else:
                w, h = paperFormats.get(width.lower(), (7 * 25.4, 7 * 25.4))
            self.width = w
            self.height = h
        else:
            self.width = width
            self.height = height
        self.fig.set_size_inches(self.width / 25.4, self.height / 25.4, forward
                = True)
        self.fig.set_size_inches(self.width / 25.4, self.height / 25.4)

    def set_layout(self, layout):
        """
        sets the layout of the subplots

        ===========
        Parameters:
        ===========
            layout : *dict*
                a dictionary that defines the layout. Example dictionaries are
                collected in the module's dictionary 'layouts'
                
        ========
        Returns:
        ========
            (None)
        """
        self.layout = layout
        self.mainframe = self.fig.add_axes([0, 0, 1, 1], frameon=False)
        self.mainframe.set_xticks([])
        self.mainframe.set_yticks([])
        lm = layout.get('legend_margins', (2., 2., 10., 10.))
        # define subplots
        # walk through all subplots 
        lmargin = float(layout['margins'][2]) / float(self.width)
        rmargin = 1. - float(layout['margins'][3]) / float(self.width)
        topmargin = 1. - float(layout['margins'][1]) / float(self.height)
        bottommargin = float(layout['margins'][0]) / float(self.height)
# 1.2 * fontheight -> spacing between rows
        lsp = layout.get('legend_textvspace', 1.2)
        legendheight = ( layout['legend_rows'] * layout['legend_fontsize'] / 
                72. * 25.4 * lsp + float(lm[0] + lm[1])) / float(self.height) 
        if layout['legend_frame'] and layout['legend_location'] == 'top':
            topmargin -= (float(layout['spacing'][2]) / self.height 
                    + legendheight) 

        haxes = rmargin - lmargin
        vaxes = topmargin - bottommargin
        hspacing = float(layout['spacing'][0]) / self.height
        vspacing = float(layout['spacing'][1]) / self.width


        gridboxwidth = (haxes - float(layout['subplot_grid_horiz'] - 1) *
                hspacing )/ float(layout['subplot_grid_horiz']) 
        gridboxheight = (vaxes - float(layout['subplot_grid_vert'] - 1) *
                vspacing ) / float(layout['subplot_grid_vert'])


        rect=[0, 0, 0, 0] # dummy, will be overwritten
        for elem in layout['subplots']:
            draw_xticks = True
            draw_yticks = True
            if isinstance(elem, int):
            # a plain number -> a single grid point
                nleft = mod(elem, layout['subplot_grid_horiz'])
                nbtm = elem // layout['subplot_grid_horiz']
            # this has only an effect if layout['ticklabels_outer_only'] is set
                if nleft > 0:
                    draw_yticks = False
                if nbtm != layout['subplot_grid_vert'] - 1:
                    draw_xticks = False
                rect = [lmargin + nleft * gridboxwidth + nleft * hspacing,
                        topmargin - (nbtm + 1) * gridboxheight - 
                        nbtm * vspacing,
                        gridboxwidth, gridboxheight]
            elif isinstance(elem, tuple):
            # a tuple defining first and last box
                nleft0 = mod(elem[0], layout['subplot_grid_horiz'])
                nbtm0 = elem[0] // layout['subplot_grid_horiz']
                nleft1 = mod(elem[1], layout['subplot_grid_horiz'])
                nbtm1 = elem[1] // layout['subplot_grid_horiz']
                if nleft0 > 0:
                    draw_yticks = False
                if nbtm1 != layout['subplot_grid_vert'] - 1:
                    draw_xticks = False
                rect = [lmargin + nleft0 * gridboxwidth + nleft0 * hspacing,
                        topmargin - (nbtm1 + 1) * gridboxheight - 
                        nbtm1 * vspacing,
                        gridboxwidth * (nleft1 - nleft0 + 1) + 
                        hspacing * (nleft1 - nleft0),
                        gridboxheight * (nbtm1 - nbtm0 + 1) + 
                        vspacing * (nbtm1 - nbtm0)]
            else:
                raise ValueError('invalid subplot configuration - only ' + 
                    'integers and tuples are allowed')
            self.axes.append(self.fig.add_axes(rect))
            if layout.get('ticklabels_outer_only', False):
                if not draw_xticks:
                    self.axes[-1].set_xticklabels([])
                if not draw_yticks:
                    self.axes[-1].set_yticklabels([])
        
        self.haxes = haxes
        self.vaxes = vaxes
        self.bottommargin = bottommargin
        self.lmargin = lmargin
        self.legendheight = legendheight

        if layout['legend_frame']:
            if layout['legend_location'] != 'top':
                raise NotImplementedError('other positions than "top" not ' +
                        'yet implemented!')
            # define legend frame
            rect = [lmargin, 1. - float(layout['margins'][1]) /
                    float(self.height) - legendheight, haxes, legendheight]
            self.legendax = self.fig.add_axes(rect)
            self.legendax.set_xticks([])
            self.legendax.set_yticks([])

    def set_cxlabel(self, text):
        """
        sets a common xlabel for all axes

        ===========
        Parameters:
        ===========
            text : *string*
            the text of the label. (formatting is in the layout dictionary)

        ========
        Returns:
        ========
            (None) 
                (the label is stored in <object>.cxlabel)

        """
        x = self.lmargin + self.haxes * .5
        y = float(self.layout.get('cxlabel_margin', 2.)) / self.height 
        #print 'x = ', x, 'y = ', y
        self.cxlabel = self.mainframe.text(x, y, text,
                fontsize=self.layout['fontsize'],
                verticalalignment='bottom', horizontalalignment='center',
                rotation='horizontal')

    def set_cylabel(self, text):
        """
        sets a common ylabel for all axes

        ===========
        Parameters:
        ===========
            text : *string*
            the text of the label. (formatting is in the layout dictionary)

        ========
        Returns:
        ========
            (None) 
                (the label is stored in <object>.cylabel)

        """
        x = (float(self.layout.get('cylabel_margin', 3.)) / self.width) 
        y = self.bottommargin + self.vaxes * .5
        #print 'x = ', x, 'y = ', y
        self.cylabel = self.mainframe.text(x, y, text,
                fontsize=self.layout['fontsize'],
                verticalalignment='center', horizontalalignment='left',
                rotation='vertical')

    def add_to_legend(self, linefmt, textstr):
        """
        adds an item to the legend and draws the legend

        note: the legend itself is stored in <object>.legend_entries

        Parameters:
        -----------
            linefmt : *dict*
                properties of the line to draw
            textstr : *str*
                text of the legend entry

        Returns:
        --------
            (None)

        
        """
        # compute position for legend_entry:
        le_row = mod(len(self.legend_entries), self.layout['legend_rows'])
        le_col = len(self.legend_entries) // self.layout['legend_rows']
        # print 'row / col: ', le_row, le_col
        # legend margins
        lm = self.layout.get('legend_margins', (2, 2, 10, 10))
        inner_width = ( float(self.haxes * self.width - lm[2] - lm[3]) /
                (self.haxes * self.width))
        #print 'inner width:', inner_width
        inner_height = float(self.legendheight * self.height - lm[0] -
                lm[1]) / (self.legendheight * self.height)

        btmline = float(lm[0]) / (self.legendheight * self.height)
        lmargin = float(lm[2]) / (self.haxes * self.width)
        #lm
        #print 'btmline: ', btmline
        #print 'lmargin: ', lmargin
        fontheight = (float(self.layout.get('legend_fontsize', 10.)) * 25.4 /
                72. / (self.legendheight * self.height)) # abs.
        #print 'fontheight:', fontheight

        line_x = (lmargin + (float(le_col) / float(self.layout['legend_cols'] +
            1.)) / inner_width)

        lsp = self.layout.get('legend_textvspace', 1.2)
        times_fh = lsp * float(self.layout['legend_rows'] - le_row) 
        line_y = btmline + fontheight * times_fh - fontheight / 2.

        line_length = (float(self.layout.get('legend_linelength', 8.)) /
                (self.haxes * self.width))

        line_text_sep = (float(self.layout.get('legend_line_text_sep', 1.)) /
                (self.haxes * self.width))

        self.legline = [[line_x, line_x + line_length], [line_y, line_y]]
        self.legendax.plot(*self.legline, **linefmt)
        self.legend_entries.append(self.legendax.text(
            line_x + line_length + line_text_sep, line_y - fontheight / 4.,
            textstr, va='baseline', ha='left',
            fontsize=self.layout.get('legend_fontsize', 10)))

        self.legendax.set_xlim(0, 1)
        self.legendax.set_ylim(0, 1)
        self.legendax.set_xticks([])
        self.legendax.set_yticks([])



    def clear_legend(self):
        """
        clears the legend frame
        """
        self.legend_entries = []
        self.legendax.cla()
        self.legendax.set_xticks([])
        self.legendax.set_yticks([])



