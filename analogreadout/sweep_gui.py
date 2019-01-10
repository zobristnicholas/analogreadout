import sys
from pymeasure.display.Qt import QtGui
from analogreadout.daq import DAQ
from analogreadout.procedures import Sweep2
from mkidplotter import (SweepGUI, SweepGUIProcedure2, SweepPlotWidget, NoisePlotWidget,
                         get_image_icon)


def sweep_window():
    x_list = (('i1', 'i1_bias'), ('f_psd', 'f_psd'),
              ('i2', 'i2_bias'), ('f_psd', 'f_psd'))
    y_list = (('q1', 'q1_bias'), ("i1_psd", "q1_psd"),
              ('q2', 'q2_bias'), ("i2_psd", "q2_psd"))
    x_label = ("I [V]", "frequency [Hz]", "I [V]", "frequency [Hz]")
    y_label = ("Q [V]", "PSD [V² / Hz]", "Q [V]", "PSD [V² / Hz]")
    legend_list = (('sweep', 'bias point'), ('I Noise', 'Q Noise'),
                   ('sweep', 'bias point'), ('I Noise', 'Q Noise'))
    widgets_list = (SweepPlotWidget, NoisePlotWidget, SweepPlotWidget, NoisePlotWidget)
    names_list = ('Channel 1: Sweep', 'Channel 1: Noise',
                  'Channel 2: Sweep', 'Channel 2: Noise')
    
    Sweep2.connect_daq(DAQ("UCSB"))
    w = SweepGUI(Sweep2, base_procedure_class=SweepGUIProcedure2, x_axes=x_list,
                 y_axes=y_list, x_labels=x_label, y_labels=y_label,
                 legend_text=legend_list, plot_widget_classes=widgets_list,
                 plot_names=names_list)
    return w


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    app.setWindowIcon(get_image_icon("loop.png"))
    window = sweep_window()
    window.show()
    ex = app.exec_()
    del app  # prevents unwanted segfault on closing the window
    sys.exit(ex)