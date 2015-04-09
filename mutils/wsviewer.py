# -*- coding: utf-8 -*-
"""
Created on Thursday, Feb 26, 12:25

@file

@brief Provides a QT visualization of a savable object for IPython in QT mode.

@author: Moritz Maus

Version 0.0.1
    very basic first draft. Works with IPython. Missing features:
        - nice layout of text
        - adapt label lbl1 to resize operations
        - add option to recursively open "saveable" (sub-)objects
        - delete object (and stop update loop) if "close" is pressed
"""

from PyQt4 import QtGui, QtCore
import mutils.io as mio

class NoQTError(Exception):
    """
    Error thrown if no running Qt is deteced
    """
    pass


class WsView(object):
    """
    successor of Visualizer

    Usage: vis = Visualizer(ws, title), where s is the workspace object and
    title is the title string.
    """

    def __init__(self, workspace=None, title='<set title!>'):
        """
        Creates the object

        @param workspace (mutils.io.workspace) The base workspace object
        @param title (string) the window title

        """

        self.ws = workspace
        self.old_display_str = ""
        
        self.w = QtGui.QTreeWidget()
        self.w.setColumnCount(3)
        
        self.baseItem = self.w.invisibleRootItem()
        self.baseItem.setExpanded(False)
        
        self._addWorkspace(self.baseItem, self.ws)

        self.w.resize(500, 450)
        self.w.setColumnWidth(0, 200)
        self.w.setColumnWidth(1, 100)
        self.w.setColumnWidth(2, 400)
        
        self.w.setHeaderLabels(['variable name', 'type', 'info'])
        self.w.setWindowTitle(title)
        self.w.show()
        self.update()


    def _addWorkspace(self, widget, ws):
    
        for line in ws.display(returnString=True, extra_sep="|").splitlines():
            cells = [cell.strip() for cell in line.split('|')]
            newItem = QtGui.QTreeWidgetItem()
            newItem.setText(0, str(cells[0]))
            newItem.setText(1, str(cells[1]))
            if len(cells) > 2:
                newItem.setText(2, str(cells[2]))
                
            widget.addChild(newItem)
            # recursively add workspaces objects
            if cells[1] == '<workspace object>':                
                self._addWorkspace(newItem, ws[cells[0]])

    def _getWsStringRec(self, ws):
        all_lines = []
        for line in ws.display(returnString=True, extra_sep="|").splitlines():
            cells = [cell.strip() for cell in line.split('|')]
            all_lines.append('|'.join(cells))
            if cells[1] == '<workspace object>':                
                all_lines.extend(
                        self._getWsStringRec(ws[cells[0]]).splitlines())

        return '\n'.join(all_lines)


    def update(self):
        """
        updates the drawing window and enter an update loop
        """
        if self._getWsStringRec(self.ws) != self.old_display_str:
            self.old_display_str = self._getWsStringRec(self.ws)
            self.w.clear()
            self._addWorkspace(self.baseItem, self.ws)

        # repeate the update
        QtCore.QTimer.singleShot(1000, self.update)


class Visualizer(object):
    """
    A simple visualization of a mutils.io.saveable object. It keeps itself
    updated. If updating breaks, call the "update" method manually.

    **NOTE** This class requires that a Qt Code Application is already running.
    It is designed to run inside IPython with --pylab=qt option.

    Usage: vis = Visualizer(s, title), where s is the saveable instance and
    title is the title string.

    To delete the window afterwards, run "del vis".
    """
    def __init__(self, workspace = None, title='WsViewer'):
        app = QtCore.QCoreApplication.instance()
        if app == None:
            raise NoQTError('pylab QT required - start IPython with --pylab=qt')
        self.w = QtGui.QWidget()
        self.w.resize(300, 600)
        self.w.move(300, 200)
        self.w.setWindowTitle(title)
        self.w.show()

        self.lbl1 = QtGui.QLabel('<empty>', self.w)
        self.lbl1.move(10, 50)
        self.lbl1.resize(290,500)
        self.lbl1.setAlignment(QtCore.Qt.AlignTop)
        self.lbl1.show()

        if workspace is None:
            self.ws = mio.saveable()
        else:
            self.ws = workspace
            self.update()

        print "NOTE: THIS CLASS IS REPLACED BY WsView!!"

        #self._timer = QtCore.QTimer()

    def setSaveable(self, dat):
        """
        exchange the saveable object
        """
        self.ws = dat
        self.update()

    def update(self):
        """
        updates the drawing window and enter an update loop
        """
        self.lbl1.setText(self.ws.__str__())
        # repeate the update
        QtCore.QTimer.singleShot(1000, self.update)




