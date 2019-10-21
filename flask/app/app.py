import math
import numpy as np
import pandas as pd

from bokeh.embed import components
from bokeh.layouts import column, gridplot, layout, row
from bokeh.models import ColumnDataSource, HoverTool, PrintfTickFormatter
from bokeh.plotting import figure
from bokeh.transform import factor_cmap

from flask import Flask, render_template, request

df = pd.read_csv('../../data/preprocess/data/complete_train.csv')


palette = ['#ba32a0', '#f85479', '#f8c260', '#00c2ba']

chart_font = 'Helvetica'
chart_location_font_size = '16pt'
chart_location_alignment = 'center'
axis_label_size = '14pt'
axis_ticks_size = '12pt'
default_padding = 30
chart_inner_left_padding = 0.015
chart_font_style_location = 'bold italic'


# def update(p_class):
#     Salaire_chart = Salaire_bar_chart(df, p_class)
#     locat_chart = class_locations_bar_chart(df, p_class)
#     hist_métier_sc = métier_sc_chart(df, p_class)
#     return (
#         Salaire_chart,
#         locat_chart,
#         hist_métier_sc
#     )

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def chart():
    selected_class = request.form.get('dropdown-select')

    # if selected_class == 0 or selected_class == None:
    #     Salaire_chart, locat_chart, hist_métier_sc = update(1)
    # else:
    #     Salaire_chart, locat_chart, hist_métier_sc = update(selected_class)

    # script_Salaire_chart, div_Salaire_chart = components(Salaire_chart)
    # script_locat_chart, div_locat_chart = components(locat_chart)
    # script_hist_métier_sc, div_hist_métier_sc = components(hist_métier_sc)

    return render_template(
        'index.html'
    )


if __name__ == '__main__':
    app.run(debug=True)