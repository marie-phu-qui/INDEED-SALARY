import math
import numpy as np
import pandas as pd

import bokeh
from bokeh.plotting import figure
from bokeh.embed import file_html
from bokeh.embed import components
from bokeh.models import HoverTool
from bokeh.palettes import Category10
from bokeh.transform import cumsum
import bokeh_catplot

import folium

import holoviews as hv
from holoviews import dim
from bokeh.models import GraphRenderer
hv.extension('bokeh')
renderer = hv.renderer('bokeh')

from math import pi

from flask import Flask, render_template, request


# À changer pour le csv final sous model :
# data = pd.read_csv('../../model/data/data_final.csv')
data = pd.read_csv('../../data/preprocess/data/data_final.csv')


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
    pie_metier_loc_plot, loc_part = pie_metier_loc()
    pie_contract_plot, contract_part = pie_contract()
    pie_job_plot = pie_job()
    pie_domains_plot, repartition_domain = pie_domains()
    whiskers_plot_salaires = box_plot_salaires()

    script_count_domain_plot, div_count_domain_plot = components(count_domain_plot)
    script_avg_salary_per_job_plot, div_avg_salary_per_job_plot = components(avg_salary_per_job_plot)
    script_variance_avg_sal_loc_plot, div_variance_avg_sal_loc_plot = components(variance_avg_sal_loc_plot)
    script_pie_metier_loc_plot, div_pie_metier_loc_plot = components(pie_metier_loc_plot)
    script_pie_contract_plot, div_pie_contract_plot = components(pie_contract_plot)
    script_pie_job_plot, div_pie_job_plot = components(pie_job_plot)
    script_pie_domains_plot, div_pie_domains_plot = components(pie_domains_plot)
    script_box_plot, div_box_plot = components(whiskers_plot_salaires)
    return render_template(
        'graphs.html',
        script_count_domain_plot = script_count_domain_plot, 
        div_count_domain_plot = div_count_domain_plot,
        script_avg_salary_per_job_plot  = script_avg_salary_per_job_plot, 
        div_avg_salary_per_job_plot = div_avg_salary_per_job_plot,
        script_variance_avg_sal_loc_plot = script_variance_avg_sal_loc_plot, 
        div_variance_avg_sal_loc_plot = div_variance_avg_sal_loc_plot,
        script_pie_metier_loc_plot = script_pie_metier_loc_plot, 
        div_pie_metier_loc_plot = div_pie_metier_loc_plot,
        script_pie_contract_plot = script_pie_contract_plot, 
        div_pie_contract_plot = div_pie_contract_plot,
        script_pie_job_plot = script_pie_job_plot, 
        div_pie_job_plot = div_pie_job_plot,
        script_pie_domains_plot = script_pie_domains_plot, 
        div_pie_domains_plot = div_pie_domains_plot,
        script_box_plot = script_box_plot, 
        div_box_plot = div_box_plot,
        max_repartition_domain = (list(repartition_domain.keys())[list(repartition_domain.values()).index(max(repartition_domain.values()))]), 
        max_repartition_domain_pct = round(max(repartition_domain.values())),
        type_max_contract_part = contract_part.loc[contract_part.values == (max(contract_part.values))].keys()[0],
        max_contract_part = round(max(contract_part)),
        max_job_city = loc_part.loc[loc_part.values == (max(loc_part.values))].keys()[0],
        max_job_city_pct = round(max(loc_part.values))
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
    map = map_france()
    return render_template(
        'region.html',
        map = map
    )

# def word_cloud_dev():


def count_domains(loc):
    if loc == 'All':
        df_ite = data["métier_sc"].value_counts()
    else :
        df_ite = data.loc[data['loc_sc'] == loc]["métier_sc"].value_counts()
    dev_job_count = sum(df_ite.loc[['developer', 'devops']]) # Add software engineer when rescrapped
    data_job_count = sum(df_ite.loc[['Data_scientist', 'Data_architect', 'Data_analyst','Big_data','BI', 'Autres_metiers_data']])

    domain = ['développement', 'data']
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
    # se_avg = sum(data.loc[data['métier_sc'] == 'software_engineer']['Salaire_avg']) / (df_ite.loc['software_engineer'])
    se_avg= 0 # don't have any SE in final csv

    data_scientist_avg = sum(data.loc[data['métier_sc'] == 'Data_scientist']['Salaire_avg']) / (df_ite.loc['Data_scientist'])
    data_arch_avg = sum(data.loc[data['métier_sc'] == 'Data_architect']['Salaire_avg']) / (df_ite.loc['Data_architect'])
    data_analyst_avg = sum(data.loc[data['métier_sc'] == 'Data_analyst']['Salaire_avg']) / (df_ite.loc['Data_analyst'])
    big_data_avg = sum(data.loc[data['métier_sc'] == 'Big_data']['Salaire_avg']) / (df_ite.loc['Big_data'])
    BI_avg = sum(data.loc[data['métier_sc'] == 'BI']['Salaire_avg']) / (df_ite.loc['BI'])
    autres_data_avg = sum(data.loc[data['métier_sc'] == 'Autres_metiers_data']['Salaire_avg']) / (df_ite.loc['Autres_metiers_data'])

    jobs = ['developer', 'devops', 'software_engineer', 'Data_scientist', 'Data_architect', 'Data_analyst', 'Big_data', 'BI', 'Autres_metiers_data']
    counts = [developer_avg, devops_avg, se_avg, data_scientist_avg, data_arch_avg, data_analyst_avg, big_data_avg, BI_avg, autres_data_avg]

    p = figure(x_range=jobs, plot_height=350, plot_width=900, title="Salaire moyen par poste",  background_fill_color='#feb236', toolbar_location=None, tools="")

    p.vbar(x=jobs, top=counts, width=0.2)

    p.xgrid.grid_line_color = None
    p.y_range.start = 0
    return p

def get_percents():
    # get amounts
    job_count = data['métier_sc'].value_counts()
    loc_count = data['loc_sc'].value_counts()
    # xp_count = data['Experiences'].value_counts() # no XP in final CSV
    contract_count = data['contrat'].value_counts()
    # get percents
    job_part = ((job_count)/len(data))*100
    # xp_part = ((xp_count)/len(data))*100  # no XP in final CSV
    loc_part = ((loc_count)/len(data))*100
    contract_part = ((contract_count)/len(data))*100
    return job_count, loc_count, contract_count, job_part, loc_part, contract_part  # no XP in final CSV

def pie_metier_loc():
    ''' 
    Fonction permettant de faire un pie chart de la répartition des métiers selon les localisations
        
    '''
    job_count, loc_count, contract_count, job_part, loc_part, contract_part = get_percents()
    bokeh_settings = pd.Series(loc_part).reset_index(name='value').rename(columns={'index':'city'})
    bokeh_settings['value'] = loc_part.values
    bokeh_settings['angle'] = bokeh_settings['value']/bokeh_settings['value'].sum() * 2*pi
    bokeh_settings['color'] = Category10[len(loc_part)]

    p = figure(plot_height=350, title="Repartition offres par localisation (en %)", toolbar_location=None,
            tools="hover", tooltips="@city: @value", x_range=(-0.5, 1.0))

    p.wedge(x=0, y=1, radius=0.4,
            start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
            line_color="white", fill_color='color', legend='city', source=bokeh_settings)

    p.axis.axis_label=None
    p.axis.visible=False
    p.grid.grid_line_color = None
    return p, loc_part

def pie_contract():
    ''' 
    Fonction permettant de faire un pie chart de la répartition des contrats sur toutes les offres
        
    '''
    job_count, loc_count, contract_count, job_part, loc_part, contract_part = get_percents()

    bokeh_settings = pd.Series(contract_part).reset_index(name='value').rename(columns={'index':'contract'})
    bokeh_settings['angle'] = bokeh_settings['value']/bokeh_settings['value'].sum() * 2*pi
    bokeh_settings['color'] = Category10[len(contract_part)]

    p = figure(plot_height=350, title="Repartition par contrat (en %)", toolbar_location=None,
            tools="hover", tooltips="@contract: @value", x_range=(-0.5, 1.0))

    p.wedge(x=0, y=1, radius=0.4,
            start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
            line_color="white", fill_color='color', legend='contract', source=bokeh_settings)

    p.axis.axis_label=None
    p.axis.visible=False
    p.grid.grid_line_color = None

    return p, contract_part

def pie_job():
    ''' 
    Fonction permettant de faire un pie chart de la part des différents jobs sur toutes les offres
        
    '''
    job_count, loc_count, contract_count, job_part, loc_part, contract_part = get_percents()

    bokeh_settings = pd.Series(job_part).reset_index(name='value').rename(columns={'index':'city'})
    bokeh_settings['angle'] = bokeh_settings['value']/bokeh_settings['value'].sum() * 2*pi
    bokeh_settings['color'] = Category10[len(job_part)]

    p = figure(plot_height=350, title="Repartition par metiers (en %)", toolbar_location=None,
            tools="hover", tooltips="@city: @value", x_range=(-0.5, 1.0))

    p.wedge(x=0, y=1, radius=0.4,
            start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
            line_color="white", fill_color='color', legend='city', source=bokeh_settings)

    p.axis.axis_label=None
    p.axis.visible=False
    p.grid.grid_line_color = None

    return p    

def pie_domains():
    ''' 
    Fonction permettant de faire un pie chart de la part des domaines tech vs. data
        
    '''
    df_ite = data["métier_sc"].value_counts()
    dev_job_count = sum(df_ite.loc[['developer', 'devops']]) # Add software_engineer when rescrapped
    data_job_count = sum(df_ite.loc[['Data_scientist','Data_engineer', 'Data_architect', 'Data_analyst','Big_data','BI', 'Autres_metiers_data']])
    dev_job_pct = (dev_job_count/len(data))*100
    data_job_pct = (data_job_count/len(data))*100

    chart_colors = ['#ada397', '#feb236']

    repartition_domain = {'développement' : dev_job_pct, 'data' : data_job_pct} 

    bokeh_settings = pd.Series(repartition_domain).reset_index(name='value').rename(columns={'index':'jobs'})
    bokeh_settings['angle'] = bokeh_settings['value']/bokeh_settings['value'].sum() * 2*pi
    bokeh_settings['color'] = chart_colors[:len(repartition_domain)]

    p = figure(plot_height=350, title="Repartition offres par branche (en %)", toolbar_location=None,
            tools="hover", tooltips="@jobs: @value", x_range=(-0.5, 1.0))

    p.wedge(x=0, y=1, radius=0.4,
            start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
            line_color="white", fill_color='color', legend='jobs', source=bokeh_settings)

    p.axis.axis_label=None
    p.axis.visible=False
    p.grid.grid_line_color = None
    return p, repartition_domain

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

def metiers_per_loc():
    df_ite = data["loc_sc"].value_counts()
    return df_ite

def map_france() :
    map_osm = folium.Map(location=[47, 1.3], zoom_start=5.55, width=750,height=750)

    job_loc = metiers_per_loc()
    geo = {'Paris' : {'coords' : [48.864716, 2.349014], 'pop' : 7026.765, 'tx_chom': 7.6}, 
            'Bordeaux': {'coords' : [44.8333, -0.5667], 'pop' : 783.081, 'tx_chom': 8}, 
            'Lyon' : {'coords':[45.75, 4.85], 'pop': 1381.349, 'tx_chom': 7.5}, 
            'Nantes' : {'coords': [47.2173, -1.5534], 'pop': 638.931, 'tx_chom': 7.2}, 
            'Toulouse' : {'coords': [43.6043, 1.4437], 'pop': 762.956, 'tx_chom': 10.3}}

    for city in geo:
        nb_chom= (geo[city]["pop"]/geo[city]['tx_chom'])*100
        job_density = int(job_loc[city]/(geo[city]["pop"]*1000)*10000)
        job_per_chom = int(job_loc[city]/(nb_chom)*10000)
        opacity = job_density/2.5
        map_osm.add_child(folium.RegularPolygonMarker(location=geo[city]['coords'],  number_of_sides=70, fill_opacity=opacity, color = '#feb236', tooltip=f'Région de {city}. <br> Métropôle comprenant {geo[city]["pop"]} habitants. <br> Densité de job {(job_density)} pour 10.000 habitants', fill_color='#feb236', radius=geo[city]["pop"]/45))
        map_osm.add_child(folium.RegularPolygonMarker(location=geo[city]['coords'], rotation=45, number_of_sides=4, fill_opacity=0.7, color = '#feb236', tooltip=f'Région de {city}. <br> Métropôle comprenant {geo[city]["pop"]} habitants. <br> Densité de job {(job_per_chom)} pour 10.000 chômeurs', fill_color='red', radius=nb_chom/1000))
    map_osm.save('templates/map.html')
    return map_osm


def box_plot_salaires():
    bkp = bokeh.palettes.d3['Category20c'][20]
    palette = bkp[:3] + bkp[4:7] + bkp[8:11]

    p = bokeh_catplot.box(data=data, cats='métier_sc', val='Salaire_avg', whisker_caps=True, outlier_marker='diamond', palette=palette, width=900, height=400)

    return p


if __name__ == '__main__':
	app.run(port=5000, debug=True)