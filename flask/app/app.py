import math
import numpy as np
import pandas as pd

from bokeh.plotting import figure
from bokeh.embed import file_html
from bokeh.embed import components
from bokeh.models import HoverTool
import pygal

from flask import Flask, render_template, request


# À changer pour le csv final
data = pd.read_csv('../../data/preprocess/data/complete_train.csv')

chart_font = 'Helvetica'
chart_location_font_size = '16pt'
chart_location_alignment = 'center'
axis_label_size = '14pt'
axis_ticks_size = '12pt'
default_padding = 30
chart_inner_left_padding = 0.015
chart_font_style_location = 'bold italic'

app = Flask(__name__)

def update_plots(dom, loc):
    count_domain_plot,dev_job_count, data_job_count = count_domains(loc)
    avg_salary_per_job_plot = avg_salary_per_job(dom, loc)
    return (
        count_domain_plot,
        avg_salary_per_job_plot,
    )


@app.route('/', methods=['GET', 'POST'])
def global_view():
    list_jobs = data["métier_sc"].unique()
    list_regions = data["loc_sc"].unique()
    total_jobs = data["métier_sc"].count()
    avg_salary = int(data["Salaire_avg"].mean())
    return render_template(
        'index.html', 
        jobs = list_jobs,
        regions = list_regions,
        total_jobs = total_jobs,
        avg_salary = avg_salary
    )

@app.route('/model', methods=['GET', 'POST'])
def view_model():
    return render_template(
        'model.html', 
    )

@app.route('/graph', methods=['GET', 'POST'])
def chart():
    selected_domain = request.form.get('dropdown-select-dom')
    selected_loc = request.form.get('dropdown-select-loc')
    variance_avg_sal_loc_plot = variance_avg_sal_loc()
    count_domain_plot, avg_salary_per_job_plot  = update_plots('all', 'All')
    # if selected_domain == 'all' or selected_domain == None or selected_loc == 'All' or selected_domain == None:
    #     count_domain_plot = update_plots('all', 'All')
    #     avg_salary_per_job_plot = update_plots('all', 'All')
    # else:
    #     count_domain_plot = update_plots(selected_domain, selected_loc)
    #     avg_salary_per_job_plot = update_plots(selected_domain, selected_loc)


    script_count_domain_plot, div_count_domain_plot = components(count_domain_plot)
    script_avg_salary_per_job_plot, div_avg_salary_per_job_plot = components(avg_salary_per_job_plot)
    script_variance_avg_sal_loc_plot, div_variance_avg_sal_loc_plot = components(variance_avg_sal_loc_plot)

    return render_template(
        'graphs.html',
        script_count_domain_plot = script_count_domain_plot, 
        div_count_domain_plot = div_count_domain_plot,
        script_avg_salary_per_job_plot  = script_avg_salary_per_job_plot, 
        div_avg_salary_per_job_plot = div_avg_salary_per_job_plot,
        script_variance_avg_sal_loc_plot = script_variance_avg_sal_loc_plot, 
        div_variance_avg_sal_loc_plot = div_variance_avg_sal_loc_plot, 
    )

@app.route('/data', methods=['GET'])
def data_plots():
    p, dev_job_count, data_job_count = count_domains("All")
    return render_template(
        'data.html',
        data_job_count = data_job_count
    )

@app.route('/dev', methods=['GET'])
def dev_plots():
    p, dev_job_count, data_job_count = count_domains("All")

    return render_template(
        'dev.html',
        dev_job_count = dev_job_count
    )

@app.route('/region', methods=['GET'])
def loc_plots():
    fr_chart = pygal.maps.fr.Regions(human_readable=True)
    fr_chart.title = 'French regions'
    fr_chart.add('Métropole', ['69', '92', '13'])
    map = fr_chart.render(is_unicode=True)
    return render_template(
        'region.html',
        map = map
    )

from datetime import datetime, timedelta


@app.route('/test')
def test():
    date_chart = pygal.Line(x_label_rotation=20)
    date_chart.x_labels = map(lambda d: d.strftime('%Y-%m-%d'), [
    datetime(2013, 1, 2),
    datetime(2013, 1, 12),
    datetime(2013, 2, 2),
    datetime(2013, 2, 22)])
    date_chart.add("Visits", [300, 412, 823, 672])
    chart = date_chart.render()
    return render_template(
        'region.html',
        map = chart
    )

def count_domains(loc):
    if loc == 'All':
        df_ite = data["métier_sc"].value_counts()
    else :
        df_ite = data.loc[data['loc_sc'] == loc]["métier_sc"].value_counts()
    dev_job_count = sum(df_ite.loc[['developer', 'devops', 'software_engineer']])
    data_job_count = sum(df_ite.loc[['Data_scientist', 'Data_architect', 'Data_analyst','Big_data','BI', 'Autres_metiers_data']])

    domain = ['dev', 'data']
    counts = [dev_job_count, data_job_count]

    p = figure(x_range=domain, plot_height=500, title="Nombre de poste par domaine d'activité",
            toolbar_location=None, tools="")

    p.vbar(x=domain, top=counts, width=0.9)

    p.xgrid.grid_line_color = None
    p.y_range.start = 0
    return p, dev_job_count, data_job_count

def avg_salary_per_job(dom, loc):
    if dom == 'all' or loc == 'All' :
        df_ite = data["métier_sc"].value_counts()
    else :
        df_ite = data.loc[data['loc_sc'] == loc]["métier_sc"].value_counts()
    
    developer_avg = sum(data.loc[data['métier_sc'] == 'developer']['Salaire_avg']) / (df_ite.loc['developer'])
    devops_avg = sum(data.loc[data['métier_sc'] == 'devops']['Salaire_avg']) / (df_ite.loc['devops'])
    se_avg = sum(data.loc[data['métier_sc'] == 'software_engineer']['Salaire_avg']) / (df_ite.loc['software_engineer'])

    data_scientist_avg = sum(data.loc[data['métier_sc'] == 'Data_scientist']['Salaire_avg']) / (df_ite.loc['Data_scientist'])
    data_arch_avg = sum(data.loc[data['métier_sc'] == 'Data_architect']['Salaire_avg']) / (df_ite.loc['Data_architect'])
    data_analyst_avg = sum(data.loc[data['métier_sc'] == 'Data_analyst']['Salaire_avg']) / (df_ite.loc['Data_analyst'])
    big_data_avg = sum(data.loc[data['métier_sc'] == 'Big_data']['Salaire_avg']) / (df_ite.loc['Big_data'])
    BI_avg = sum(data.loc[data['métier_sc'] == 'BI']['Salaire_avg']) / (df_ite.loc['BI'])
    autres_data_avg = sum(data.loc[data['métier_sc'] == 'Autres_metiers_data']['Salaire_avg']) / (df_ite.loc['Autres_metiers_data'])

    jobs = ['developer', 'devops', 'software_engineer', 'Data_scientist', 'Data_architect', 'Data_analyst', 'Big_data', 'BI', 'Autres_metiers_data']
    counts = [developer_avg, devops_avg, se_avg, data_scientist_avg, data_arch_avg, data_analyst_avg, big_data_avg, BI_avg, autres_data_avg]

    p = figure(x_range=jobs, plot_height=350, plot_width=900, title="Salaire moyen par poste",
            toolbar_location=None, tools="")

    p.vbar(x=jobs, top=counts, width=0.2)

    p.xgrid.grid_line_color = None
    p.y_range.start = 0
    return p

def variance_avg_sal_loc():
    # Format the tooltip
    tooltips = [
                ('job','@x'),
                ('salaire average', '@y'),
                ('label region', '@x'),
    ]
    p = figure(plot_width = 800, plot_height = 500, title = 'Variance entre les salaires par région')
    # 3. Add Glyphs
    num_val_loc = list(pd.factorize(data["loc_sc"])[0])
    x = num_val_loc
    y = data['Salaire_avg']

    p.xaxis.ticker = list(set(num_val_loc))
    p.xaxis.major_label_overrides = {0: 'Paris', 1: 'Bordeaux', 2: 'Lyon', 3: 'Nantes', 4: 'Toulouse'}

    # Add the HoverTool to the figure
    p.circle(x, y, color = 'red', alpha = 0.5, size = 15)
    p.add_tools(HoverTool(tooltips=tooltips))

    return p

def map_france() :
    fr_chart = pygal.maps.fr.Departments()
    fr_chart.title = 'Some departments'
    fr_chart.add('Métropole', ['69', '92', '13'])
    fr_chart.add('Corse', ['2A', '2B'])
    fr_chart.add('DOM COM', ['971', '972', '973', '974'])
    return fr_chart.render_response()

if __name__ == '__main__':
	app.run(port=5000, debug=True)