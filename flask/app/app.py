import math
import numpy as np
import pandas as pd

from bokeh.plotting import figure
from bokeh.embed import file_html
from bokeh.embed import components

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
    count_domain_plot = count_domains(loc)
    avg_salary_per_job_plot = avg_salary_per_job(dom, loc)
    return (
        count_domain_plot,
        avg_salary_per_job_plot
    )

@app.route('/', methods=['GET', 'POST'])
def chart():
    selected_domain = request.form.get('dropdown-select-dom')
    selected_loc = request.form.get('dropdown-select-loc')

    if selected_domain == 'all' or selected_domain == None or selected_loc == 'All' or selected_domain == None:
        count_domain_plot = update_plots('all', 'All')
        avg_salary_per_job_plot = update_plots('all', 'All')
    else:
        count_domain_plot = update_plots(selected_domain, selected_loc)
        avg_salary_per_job_plot = update_plots(selected_domain, selected_loc)

    script_count_domain_plot, div_count_domain_plot = components(count_domain_plot)
    script_avg_salary_per_job_plot, div_avg_salary_per_job_plot = components(avg_salary_per_job_plot)

    return render_template(
        'index.html', 
        script_count_domain_plot = script_count_domain_plot, 
        div_count_domain_plot = div_count_domain_plot,
        script_avg_salary_per_job_plot  = script_avg_salary_per_job_plot, 
        div_avg_salary_per_job_plot = div_avg_salary_per_job_plot
    )


def count_domains(loc):
    if loc == 'All':
        df_ite = data["métier_sc"].value_counts()
    else :
        df_ite = data.loc[data['loc_sc'] == loc]["métier_sc"].value_counts()
    dev_job_count = sum(df_ite.loc[['developer','devops', 'software_engineer']])
    data_job_count = sum(df_ite.loc[['Data_scientist', 'Data_architect', 'Data_analyst','Big_data','BI', 'Autres_metiers_data']])

    domain = ['dev', 'data']
    counts = [dev_job_count, data_job_count]

    p = figure(x_range=domain, plot_height=500, title="Nombre de poste par domaine d'activité",
            toolbar_location=None, tools="")

    p.vbar(x=domain, top=counts, width=0.9)

    p.xgrid.grid_line_color = None
    p.y_range.start = 0
    return p

def avg_salary_per_job(dom, loc):
    if dom == 'all' or loc == 'All' :
        df_ite = data["métier_sc"].value_counts()
    else :
        df_ite = data.loc[data['loc_sc'] == loc]["métier_sc"].value_counts()

    data['Salaire_avg'] = data['Salaire_Min']+data['Salaire_Max']/2
    
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

if __name__ == '__main__':
	app.run(port=5000, debug=True)